//! A bounded HTTP adapter around a single, thread-owned Jet engine.

use std::collections::BTreeMap;
use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use actix_web::error::{InternalError, JsonPayloadError};
use actix_web::http::{StatusCode, header};
use actix_web::{App, HttpRequest, HttpResponse, HttpServer, web};
use jet_core::http::{HttpDecisionRequest, HttpDecisionResponse, HttpErrorResponse};
use jet_core::{DecisionRequest, DecisionResponse, JetError, NoulQuestion, Question, Result};
use jet_engine::{Engine, EngineConfig};
use tokio::sync::{mpsc, oneshot};

#[derive(Clone)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    pub queue_capacity: usize,
    /// Maximum queue and inference wait after HTTP decoding, not a native abort deadline.
    pub request_timeout: Duration,
    pub http_workers: usize,
    pub max_body_bytes: usize,
    /// Optional bearer token. This value is never included in server logs.
    pub api_key: Option<String>,
}

impl ServerConfig {
    fn validate(&self) -> Result<()> {
        if self.queue_capacity == 0
            || self.http_workers == 0
            || self.max_body_bytes == 0
            || self.request_timeout.is_zero()
            || Instant::now().checked_add(self.request_timeout).is_none()
        {
            return Err(JetError::InvalidRequest(
                "server queue, worker, body and timeout limits must be positive and representable"
                    .to_owned(),
            ));
        }
        if self.api_key.as_ref().is_some_and(|key| key.is_empty()) {
            return Err(JetError::InvalidRequest(
                "server API key must not be empty".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Load and warm up the engine once, then serve until Ctrl-C or SIGTERM.
///
/// The engine is created, used, and dropped on its own OS thread. HTTP timeouts
/// discard queued work but do not interrupt an inference that has already begun.
pub async fn serve(config: ServerConfig, engine_config: EngineConfig) -> Result<()> {
    config.validate()?;
    // Reserve the address before loading weights, so bind failures are immediate.
    let listener = TcpListener::bind(config.bind)?;
    listener.set_nonblocking(true)?;
    let address = listener.local_addr()?;
    let model_id = engine_config.model_id.clone();
    let (worker, startup) = Worker::spawn(&config, model_id, move || {
        let mut engine = Engine::load(engine_config)?;
        engine.decide(warmup_request())?;
        Ok(move |request| engine.decide(request))
    })?;
    if let Err(error) = startup.await.map_err(|_| worker_stopped()).and_then(|r| r) {
        worker.shutdown().await?;
        return Err(error);
    }

    let state = worker.state.clone();
    let shutdown_state = state.clone();
    let max_body_bytes = config.max_body_bytes;
    let server = HttpServer::new(move || {
        App::new()
            .app_data(web::Data::from(state.clone()))
            .app_data(json_config(max_body_bytes))
            .configure(routes)
    })
    .workers(config.http_workers)
    .tcp_nodelay(true)
    .shutdown_signal(async move {
        if let Err(error) = termination_signal().await {
            tracing::error!(%error, "failed to receive shutdown signal");
        }
        shutdown_state.stop();
    })
    .listen(listener)?
    .run();

    tracing::info!(%address, model = %worker.state.model_id, "Jet HTTP server ready");
    let result = server.await;
    let shutdown = worker.shutdown().await;
    result?;
    shutdown
}

fn warmup_request() -> DecisionRequest {
    DecisionRequest {
        state: serde_json::json!({"warmup": true}),
        questions: BTreeMap::from([(
            "warmup".to_owned(),
            Question::Noul(NoulQuestion {
                instructions: serde_json::json!("Is this a warm-up request?"),
                criteria: None,
            }),
        )]),
    }
}

#[cfg(unix)]
async fn termination_signal() -> std::io::Result<()> {
    use std::future::{Future, poll_fn};
    use std::task::Poll;

    let mut terminate =
        actix_web::rt::signal::unix::signal(actix_web::rt::signal::unix::SignalKind::terminate())?;
    let mut interrupt = std::pin::pin!(actix_web::rt::signal::ctrl_c());
    let mut terminated = std::pin::pin!(terminate.recv());
    poll_fn(|cx| {
        if let Poll::Ready(result) = interrupt.as_mut().poll(cx) {
            return Poll::Ready(result);
        }
        terminated.as_mut().poll(cx).map(|_| Ok(()))
    })
    .await
}

#[cfg(not(unix))]
async fn termination_signal() -> std::io::Result<()> {
    actix_web::rt::signal::ctrl_c().await
}

type InferenceResult = std::result::Result<HttpDecisionResponse, ServiceError>;

struct Job {
    request: DecisionRequest,
    queued_at: Instant,
    deadline: Instant,
    reply: oneshot::Sender<InferenceResult>,
}

enum Work {
    Decide(Job),
    Stop,
}

struct State {
    sender: mpsc::Sender<Work>,
    accepting: Arc<AtomicBool>,
    model_id: String,
    request_timeout: Duration,
    api_key: Option<String>,
}

impl State {
    fn is_ready(&self) -> bool {
        self.accepting.load(Ordering::Acquire) && !self.sender.is_closed()
    }

    fn stop(&self) {
        self.accepting.store(false, Ordering::Release);
        // A full queue already wakes the worker; it checks accepting before
        // starting the next item. An empty queue needs this explicit wake-up.
        let _ = self.sender.try_send(Work::Stop);
    }
}

struct Worker {
    state: Arc<State>,
    thread: Option<JoinHandle<()>>,
}

impl Worker {
    fn spawn<F, S>(
        config: &ServerConfig,
        model_id: String,
        factory: F,
    ) -> Result<(Self, oneshot::Receiver<Result<()>>)>
    where
        F: FnOnce() -> Result<S> + Send + 'static,
        S: FnMut(DecisionRequest) -> Result<DecisionResponse> + 'static,
    {
        let (sender, mut receiver) = mpsc::channel(config.queue_capacity);
        let accepting = Arc::new(AtomicBool::new(false));
        let state = Arc::new(State {
            sender,
            accepting: accepting.clone(),
            model_id,
            request_timeout: config.request_timeout,
            api_key: config.api_key.clone(),
        });
        let (started, startup) = oneshot::channel();
        let thread = thread::Builder::new()
            .name("jet-inference".to_owned())
            .spawn(move || {
                let _reset = ResetReadiness(accepting.clone());
                let mut decide = match factory() {
                    Ok(decide) => decide,
                    Err(error) => {
                        let _ = started.send(Err(error));
                        return;
                    }
                };
                accepting.store(true, Ordering::Release);
                if started.send(Ok(())).is_err() {
                    return;
                }
                while let Some(work) = receiver.blocking_recv() {
                    if !accepting.load(Ordering::Acquire) || matches!(work, Work::Stop) {
                        receiver.close();
                        reject_work(work);
                        while let Ok(work) = receiver.try_recv() {
                            reject_work(work);
                        }
                        break;
                    }
                    let Work::Decide(job) = work else {
                        continue;
                    };
                    if job.reply.is_closed() {
                        continue;
                    }
                    let dequeued_at = Instant::now();
                    if dequeued_at >= job.deadline {
                        let _ = job.reply.send(Err(ServiceError::timeout()));
                        continue;
                    }
                    let queue_ms =
                        dequeued_at.duration_since(job.queued_at).as_secs_f64() * 1_000.0;
                    let started = Instant::now();
                    let result = decide(job.request);
                    let inference_ms = started.elapsed().as_secs_f64() * 1_000.0;
                    tracing::debug!(queue_ms, inference_ms, "decision completed");
                    let result = result
                        .map(|decision| HttpDecisionResponse {
                            decision,
                            inference_ms,
                        })
                        .map_err(ServiceError::from_engine);
                    let _ = job.reply.send(result);
                }
            })?;
        Ok((
            Self {
                state,
                thread: Some(thread),
            },
            startup,
        ))
    }

    async fn shutdown(mut self) -> Result<()> {
        self.state.stop();
        if let Some(thread) = self.thread.take() {
            actix_web::rt::task::spawn_blocking(move || thread.join())
                .await
                .map_err(|error| JetError::NativeRuntime(format!("worker join failed: {error}")))?
                .map_err(|_| worker_stopped())?;
        }
        Ok(())
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.state.stop();
        // Cancellation/early-return fallback. Normal shutdown joins off the
        // Actix runtime. Dropping the owner must never detach a native engine.
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::error!("inference worker panicked during shutdown");
        }
    }
}

struct ResetReadiness(Arc<AtomicBool>);

impl Drop for ResetReadiness {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn reject_work(work: Work) {
    if let Work::Decide(job) = work {
        let _ = job.reply.send(Err(ServiceError::unavailable()));
    }
}

fn worker_stopped() -> JetError {
    JetError::NativeRuntime("inference worker stopped unexpectedly".to_owned())
}

#[derive(Debug)]
struct ServiceError {
    status: StatusCode,
    message: String,
}

impl ServiceError {
    fn unavailable() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "inference engine is unavailable".to_owned(),
        }
    }

    fn timeout() -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: "request timed out; an inference already running cannot be interrupted"
                .to_owned(),
        }
    }

    fn from_engine(error: JetError) -> Self {
        let status = match error {
            JetError::InvalidRequest(_)
            | JetError::UnsupportedModel(_)
            | JetError::ContextExceeded { .. }
            | JetError::Json(_) => StatusCode::BAD_REQUEST,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self {
            status,
            message: error.to_string(),
        }
    }

    fn response(self) -> HttpResponse {
        error_response(self.status, self.message)
    }
}

fn error_response(status: StatusCode, message: impl Into<String>) -> HttpResponse {
    HttpResponse::build(status).json(HttpErrorResponse::new(status.as_u16(), message))
}

fn json_config(limit: usize) -> web::JsonConfig {
    web::JsonConfig::default()
        .limit(limit)
        .error_handler(|error, _request| {
            let status = match &error {
                JsonPayloadError::ContentType => StatusCode::UNSUPPORTED_MEDIA_TYPE,
                JsonPayloadError::Overflow { .. }
                | JsonPayloadError::OverflowKnownLength { .. } => StatusCode::PAYLOAD_TOO_LARGE,
                _ => StatusCode::BAD_REQUEST,
            };
            let response = error_response(status, error.to_string());
            InternalError::from_response(error, response).into()
        })
}

fn routes(config: &mut web::ServiceConfig) {
    config
        .service(
            web::resource("/api/alpha/decisions")
                .route(web::post().to(decisions))
                .default_service(web::route().to(method_not_allowed)),
        )
        .service(
            web::resource("/api/v1/models")
                .route(web::get().to(models))
                .default_service(web::route().to(method_not_allowed)),
        )
        .route("/healthz", web::get().to(health))
        .route("/readyz", web::get().to(ready))
        .default_service(web::route().to(not_found));
}

fn authorized(request: &HttpRequest, state: &State) -> bool {
    match &state.api_key {
        None => true,
        Some(expected) => request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.split_once(' '))
            .is_some_and(|(scheme, provided)| {
                scheme.eq_ignore_ascii_case("Bearer")
                    && provided.trim_start_matches(' ') == expected
            }),
    }
}

fn unauthorized() -> HttpResponse {
    HttpResponse::Unauthorized()
        .insert_header((header::WWW_AUTHENTICATE, "Bearer"))
        .json(HttpErrorResponse::new(
            401,
            "invalid or missing bearer token",
        ))
}

async fn decisions(
    request: HttpRequest,
    body: web::Json<HttpDecisionRequest>,
    state: web::Data<State>,
) -> HttpResponse {
    if !authorized(&request, &state) {
        return unauthorized();
    }
    if !state.is_ready() {
        return ServiceError::unavailable().response();
    }
    let request = match body.into_inner().into_decision(&state.model_id) {
        Ok(request) => request,
        Err(error) => return ServiceError::from_engine(error).response(),
    };
    let queued_at = Instant::now();
    let Some(deadline) = queued_at.checked_add(state.request_timeout) else {
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, "invalid request timeout");
    };
    let (reply, receiver) = oneshot::channel();
    match state.sender.try_send(Work::Decide(Job {
        request,
        queued_at,
        deadline,
        reply,
    })) {
        Ok(()) => {}
        Err(mpsc::error::TrySendError::Full(_)) => {
            return HttpResponse::TooManyRequests()
                .insert_header((header::RETRY_AFTER, "1"))
                .json(HttpErrorResponse::new(429, "inference queue is full"));
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            return ServiceError::unavailable().response();
        }
    }
    match tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), receiver).await {
        Ok(Ok(Ok(response))) => HttpResponse::Ok().json(response),
        Ok(Ok(Err(error))) => error.response(),
        Ok(Err(_)) => ServiceError::unavailable().response(),
        Err(_) => ServiceError::timeout().response(),
    }
}

async fn models(request: HttpRequest, state: web::Data<State>) -> HttpResponse {
    if !authorized(&request, &state) {
        return unauthorized();
    }
    HttpResponse::Ok().json(serde_json::json!({
        "data": [{
            "id": state.model_id,
            "name": state.model_id,
            "architecture": {
                "input_modalities": ["text"],
                "output_modalities": ["decisions"]
            }
        }]
    }))
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

async fn ready(state: web::Data<State>) -> HttpResponse {
    if state.is_ready() {
        HttpResponse::Ok().json(serde_json::json!({"status": "ready"}))
    } else {
        ServiceError::unavailable().response()
    }
}

async fn method_not_allowed() -> HttpResponse {
    error_response(StatusCode::METHOD_NOT_ALLOWED, "method not allowed")
}

async fn not_found() -> HttpResponse {
    error_response(StatusCode::NOT_FOUND, "route not found")
}

#[cfg(test)]
mod tests;
