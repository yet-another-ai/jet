use std::cell::Cell;
use std::future::{Future, poll_fn};
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc as sync_mpsc;
use std::task::Poll;

use actix_web::test as awtest;
use jet_core::{Answer, NoulAnswer, Usage};
use serde_json::{Value, json};

use super::*;

fn config() -> ServerConfig {
    ServerConfig {
        bind: SocketAddr::from(([127, 0, 0, 1], 0)),
        queue_enabled: true,
        queue_capacity: 1,
        request_timeout: Duration::from_secs(5),
        http_workers: 1,
        max_body_bytes: 1_024,
        api_key: None,
    }
}

fn body() -> Value {
    json!({
        "model": "test-model",
        "state": "A test request",
        "questions": {"q": {"type": "noul", "instructions": "Accept?"}}
    })
}

fn response() -> DecisionResponse {
    DecisionResponse {
        model: "test-model".to_owned(),
        answers: BTreeMap::from([("q".to_owned(), Answer::Noul(NoulAnswer { noul: 0.75 }))]),
        usage: Usage {
            input_tokens: 12,
            output_tokens: 2,
        },
    }
}

async fn start<F, S>(config: &ServerConfig, factory: F) -> Result<Worker>
where
    F: FnOnce() -> Result<S> + Send + 'static,
    S: FnMut(DecisionRequest) -> Result<DecisionResponse> + 'static,
{
    let (worker, startup) = Worker::spawn(config, "test-model".to_owned(), factory)?;
    startup.await.map_err(|_| worker_stopped())??;
    Ok(worker)
}

fn job(deadline: Instant) -> (Work, oneshot::Receiver<InferenceResult>) {
    let (reply, result) = oneshot::channel();
    (
        Work::Decide(Job {
            request: warmup_request(),
            queued_at: Instant::now(),
            deadline,
            reply,
            slot: None,
        }),
        result,
    )
}

async fn call(state: Arc<State>) -> Result<HttpResponse> {
    Ok(decisions(
        awtest::TestRequest::post().to_http_request(),
        web::Json(serde_json::from_value(body())?),
        web::Data::from(state),
    )
    .await)
}

async fn read_response(response: HttpResponse) -> Result<Value> {
    let bytes = actix_web::body::to_bytes(response.into_body())
        .await
        .map_err(|error| JetError::NativeRuntime(error.to_string()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

async fn call_after_active_work_finishes(state: Arc<State>) -> Result<HttpResponse> {
    tokio::time::timeout(Duration::from_secs(5), async move {
        loop {
            let response = call(state.clone()).await?;
            if response.status() != StatusCode::TOO_MANY_REQUESTS {
                return Ok(response);
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .map_err(|_| JetError::NativeRuntime("fixture inference slot was not released".to_owned()))?
}

#[actix_web::test]
async fn serves_decisions_with_thread_owned_non_send_engine() -> Result<()> {
    let main_thread = thread::current().id();
    let owner = Arc::new(std::sync::Mutex::new(None));
    let dropped = Arc::new(std::sync::Mutex::new(None));
    let worker_owner = owner.clone();
    let worker_dropped = dropped.clone();
    struct ThreadOwned {
        calls: Rc<Cell<usize>>,
        dropped: Arc<std::sync::Mutex<Option<thread::ThreadId>>>,
    }
    impl Drop for ThreadOwned {
        fn drop(&mut self) {
            if let Ok(mut dropped) = self.dropped.lock() {
                *dropped = Some(thread::current().id());
            }
        }
    }
    let worker = start(&config(), move || {
        if let Ok(mut owner) = worker_owner.lock() {
            *owner = Some(thread::current().id());
        }
        let engine = ThreadOwned {
            calls: Rc::new(Cell::new(0)),
            dropped: worker_dropped,
        };
        Ok(move |_request| {
            engine.calls.set(engine.calls.get() + 1);
            Ok(response())
        })
    })
    .await?;
    let app = awtest::init_service(
        App::new()
            .app_data(web::Data::from(worker.state.clone()))
            .app_data(json_config(1_024))
            .configure(routes),
    )
    .await;
    let request = awtest::TestRequest::post()
        .uri("/api/alpha/decisions")
        .set_json(body())
        .to_request();
    let result = awtest::call_service(&app, request).await;
    assert_eq!(result.status(), StatusCode::OK);
    let result: Value = awtest::read_body_json(result).await;
    assert_eq!(result["model"], "test-model");
    assert_eq!(result["answers"]["q"]["noul"], 0.75);
    assert_eq!(result["usage"]["input_tokens"], 12);
    assert!(result["inference_ms"].as_f64().is_some_and(|ms| ms >= 0.0));
    assert!(result["queue_ms"].as_f64().is_some_and(|ms| ms >= 0.0));
    worker.shutdown().await?;
    let owner = *owner.lock().map_err(|_| worker_stopped())?;
    let dropped = *dropped.lock().map_err(|_| worker_stopped())?;
    assert_ne!(owner, Some(main_thread));
    assert_eq!(dropped, owner);
    Ok(())
}

#[actix_web::test]
async fn json_auth_models_and_validation_share_http_contract() -> Result<()> {
    let mut configuration = config();
    configuration.api_key = Some("secret-test-key".to_owned());
    let calls = Arc::new(AtomicUsize::new(0));
    let recorded = calls.clone();
    let worker = start(&configuration, move || {
        Ok(move |_request| {
            recorded.fetch_add(1, Ordering::SeqCst);
            Ok(response())
        })
    })
    .await?;
    let app = awtest::init_service(
        App::new()
            .app_data(web::Data::from(worker.state.clone()))
            .app_data(json_config(512))
            .configure(routes),
    )
    .await;
    let mut wrong_model = body();
    wrong_model["model"] = json!("another-model");
    let cases = [
        (body().to_string(), Some("application/json"), None, 401),
        (
            "{".to_owned(),
            Some("application/json"),
            Some("Bearer secret-test-key"),
            400,
        ),
        (
            body().to_string(),
            Some("text/plain"),
            Some("Bearer secret-test-key"),
            415,
        ),
        (
            "x".repeat(513),
            Some("application/json"),
            Some("Bearer secret-test-key"),
            413,
        ),
        (
            wrong_model.to_string(),
            Some("application/json"),
            Some("Bearer secret-test-key"),
            400,
        ),
    ];
    for (payload, content_type, authorization, code) in cases {
        let mut request = awtest::TestRequest::post().uri("/api/alpha/decisions");
        if let Some(content_type) = content_type {
            request = request.insert_header((header::CONTENT_TYPE, content_type));
        }
        if let Some(authorization) = authorization {
            request = request.insert_header((header::AUTHORIZATION, authorization));
        }
        let result = awtest::call_service(&app, request.set_payload(payload).to_request()).await;
        assert_eq!(result.status().as_u16(), code);
        let result: Value = awtest::read_body_json(result).await;
        assert_eq!(result["error"]["code"], code);
        assert!(result["error"]["message"].is_string());
    }
    for route in ["/api/v1/models", "/api/alpha/decisions"] {
        for (authorization, expected_status) in [
            ("Bearer secret-test-key", StatusCode::OK),
            ("bearer secret-test-key", StatusCode::OK),
            ("BEARER   secret-test-key", StatusCode::OK),
            ("bearer SECRET-TEST-KEY", StatusCode::UNAUTHORIZED),
        ] {
            let request = if route.ends_with("models") {
                awtest::TestRequest::get().uri(route)
            } else {
                awtest::TestRequest::post().uri(route).set_json(body())
            };
            let request = request
                .insert_header((header::AUTHORIZATION, authorization))
                .to_request();
            let result = awtest::call_service(&app, request).await;
            assert_eq!(result.status(), expected_status);
            let result: Value = awtest::read_body_json(result).await;
            if expected_status == StatusCode::UNAUTHORIZED {
                assert_eq!(result["error"]["code"], 401);
            } else if route.ends_with("models") {
                assert_eq!(result["data"][0]["id"], "test-model");
                assert_eq!(
                    result["data"][0]["architecture"]["output_modalities"],
                    json!(["decisions"])
                );
            }
        }
    }
    let unauthenticated_models = awtest::call_service(
        &app,
        awtest::TestRequest::get()
            .uri("/api/v1/models")
            .to_request(),
    )
    .await;
    assert_eq!(unauthenticated_models.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    worker.shutdown().await
}

#[actix_web::test]
async fn readiness_tracks_startup_failure_and_shutdown() -> Result<()> {
    let (release, gate) = sync_mpsc::channel();
    let (worker, startup) = Worker::spawn(&config(), "test-model".to_owned(), move || {
        gate.recv_timeout(Duration::from_secs(5))
            .map_err(|_| worker_stopped())?;
        Ok(|_| Ok(response()))
    })?;
    assert_eq!(health().await.status(), StatusCode::OK);
    assert_eq!(
        ready(web::Data::from(worker.state.clone())).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    release.send(()).map_err(|_| worker_stopped())?;
    startup.await.map_err(|_| worker_stopped())??;
    assert_eq!(
        ready(web::Data::from(worker.state.clone())).await.status(),
        StatusCode::OK
    );
    let state = worker.state.clone();
    worker.shutdown().await?;
    assert_eq!(
        ready(web::Data::from(state.clone())).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(call(state).await?.status(), StatusCode::SERVICE_UNAVAILABLE);

    let (worker, startup) = Worker::spawn(&config(), "test-model".to_owned(), || {
        Err::<fn(DecisionRequest) -> Result<DecisionResponse>, _>(JetError::InvalidRequest(
            "load failed".to_owned(),
        ))
    })?;
    assert!(startup.await.map_err(|_| worker_stopped())?.is_err());
    assert!(!worker.state.is_ready());
    worker.shutdown().await
}

#[actix_web::test]
async fn full_queue_rejects_and_abandoned_or_expired_items_never_run() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let recorded = calls.clone();
    let (began, running) = oneshot::channel();
    let (release, gate) = sync_mpsc::channel();
    let worker = start(&config(), move || {
        let mut began = Some(began);
        Ok(move |_| {
            recorded.fetch_add(1, Ordering::SeqCst);
            if let Some(began) = began.take() {
                let _ = began.send(());
                gate.recv_timeout(Duration::from_secs(5))
                    .map_err(|_| worker_stopped())?;
            }
            Ok(response())
        })
    })
    .await?;
    let (first, first_result) = job(Instant::now() + Duration::from_secs(5));
    worker
        .state
        .sender
        .send(first)
        .await
        .map_err(|_| worker_stopped())?;
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .map_err(|_| JetError::NativeRuntime("fixture inference did not start".to_owned()))?
        .map_err(|_| worker_stopped())?;
    let (abandoned, abandoned_result) = job(Instant::now() + Duration::from_secs(5));
    worker
        .state
        .sender
        .send(abandoned)
        .await
        .map_err(|_| worker_stopped())?;
    drop(abandoned_result);
    assert_eq!(
        call(worker.state.clone()).await?.status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    release.send(()).map_err(|_| worker_stopped())?;
    assert!(first_result.await.map_err(|_| worker_stopped())?.is_ok());
    let (expired, expired_result) = job(Instant::now());
    worker
        .state
        .sender
        .send(expired)
        .await
        .map_err(|_| worker_stopped())?;
    let expired_result = expired_result.await.map_err(|_| worker_stopped())?;
    assert!(matches!(
        expired_result,
        Err(ServiceError {
            status: StatusCode::GATEWAY_TIMEOUT,
            ..
        })
    ));
    assert_eq!(call(worker.state.clone()).await?.status(), StatusCode::OK);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    worker.shutdown().await
}

#[actix_web::test]
async fn disabled_queue_rejects_busy_work_and_releases_successes_and_errors() -> Result<()> {
    let mut configuration = config();
    configuration.queue_enabled = false;
    configuration.queue_capacity = 8;
    let calls = Arc::new(AtomicUsize::new(0));
    let recorded = calls.clone();
    let (began, running) = oneshot::channel();
    let (release, gate) = sync_mpsc::channel();
    let worker = start(&configuration, move || {
        let mut began = Some(began);
        Ok(move |_| {
            let count = recorded.fetch_add(1, Ordering::SeqCst) + 1;
            if let Some(began) = began.take() {
                let _ = began.send(());
                gate.recv_timeout(Duration::from_secs(5))
                    .map_err(|_| worker_stopped())?;
            }
            match count {
                2 => Err(JetError::InvalidRequest(
                    "fixture invalid request".to_owned(),
                )),
                3 => Err(JetError::NativeRuntime(
                    "fixture inference error".to_owned(),
                )),
                _ => Ok(response()),
            }
        })
    })
    .await?;
    let first = actix_web::rt::spawn(call(worker.state.clone()));
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .map_err(|_| JetError::NativeRuntime("fixture inference did not start".to_owned()))?
        .map_err(|_| worker_stopped())?;
    let busy = tokio::time::timeout(Duration::from_secs(1), call(worker.state.clone()))
        .await
        .map_err(|_| JetError::NativeRuntime("busy request was queued".to_owned()))??;
    assert_eq!(busy.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        busy.headers().get(header::RETRY_AFTER),
        Some(&header::HeaderValue::from_static("1"))
    );
    assert_eq!(read_response(busy).await?["error"]["code"], 429);
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    release.send(()).map_err(|_| worker_stopped())?;
    let first = first.await.map_err(|_| worker_stopped())??;
    assert_eq!(first.status(), StatusCode::OK);
    let first = read_response(first).await?;
    for field in ["inference_ms", "queue_ms"] {
        assert!(
            first[field]
                .as_f64()
                .is_some_and(|ms| ms.is_finite() && ms >= 0.0)
        );
    }
    // Each response must free admission before an immediately following request.
    for expected in [
        StatusCode::BAD_REQUEST,
        StatusCode::INTERNAL_SERVER_ERROR,
        StatusCode::OK,
    ] {
        assert_eq!(call(worker.state.clone()).await?.status(), expected);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 4);
    worker.shutdown().await
}

#[actix_web::test]
async fn disabled_queue_keeps_active_slot_after_timeout_or_handler_cancellation() -> Result<()> {
    for cancel_handler in [false, true] {
        let mut configuration = config();
        configuration.queue_enabled = false;
        if !cancel_handler {
            configuration.request_timeout = Duration::from_millis(250);
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let recorded = calls.clone();
        let (began, running) = oneshot::channel();
        let (release, gate) = sync_mpsc::channel();
        let worker = start(&configuration, move || {
            let mut began = Some(began);
            Ok(move |_| {
                recorded.fetch_add(1, Ordering::SeqCst);
                if let Some(began) = began.take() {
                    let _ = began.send(());
                    gate.recv_timeout(Duration::from_secs(5))
                        .map_err(|_| worker_stopped())?;
                }
                Ok(response())
            })
        })
        .await?;
        let first = actix_web::rt::spawn(call(worker.state.clone()));
        tokio::time::timeout(Duration::from_secs(5), running)
            .await
            .map_err(|_| JetError::NativeRuntime("fixture inference did not start".to_owned()))?
            .map_err(|_| worker_stopped())?;
        if cancel_handler {
            first.abort();
            assert!(first.await.is_err_and(|error| error.is_cancelled()));
        } else {
            assert_eq!(
                first.await.map_err(|_| worker_stopped())??.status(),
                StatusCode::GATEWAY_TIMEOUT
            );
        }
        // The handler has ended, but the native call still owns the only slot.
        assert_eq!(
            call(worker.state.clone()).await?.status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        release.send(()).map_err(|_| worker_stopped())?;
        assert_eq!(
            call_after_active_work_finishes(worker.state.clone())
                .await?
                .status(),
            StatusCode::OK
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        worker.shutdown().await?;
    }
    Ok(())
}

#[actix_web::test]
async fn enabled_queue_reports_time_spent_waiting_for_active_inference() -> Result<()> {
    let (began, running) = oneshot::channel();
    let (release, gate) = sync_mpsc::channel();
    let worker = start(&config(), move || {
        let mut began = Some(began);
        Ok(move |_| {
            if let Some(began) = began.take() {
                let _ = began.send(());
                gate.recv_timeout(Duration::from_secs(5))
                    .map_err(|_| worker_stopped())?;
            }
            Ok(response())
        })
    })
    .await?;
    let first = actix_web::rt::spawn(call(worker.state.clone()));
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .map_err(|_| JetError::NativeRuntime("fixture inference did not start".to_owned()))?
        .map_err(|_| worker_stopped())?;
    let mut waiting = std::pin::pin!(call(worker.state.clone()));
    // Poll through admission before measuring the interval held behind the first request.
    assert!(poll_fn(|cx| Poll::Ready(waiting.as_mut().poll(cx).is_pending())).await);
    let held_since = Instant::now();
    assert_eq!(
        call(worker.state.clone()).await?.status(),
        StatusCode::TOO_MANY_REQUESTS
    );
    tokio::time::sleep(Duration::from_millis(20)).await;
    let held_ms = held_since.elapsed().as_secs_f64() * 1_000.0;
    release.send(()).map_err(|_| worker_stopped())?;
    assert_eq!(
        first.await.map_err(|_| worker_stopped())??.status(),
        StatusCode::OK
    );
    let result = waiting.await?;
    assert_eq!(result.status(), StatusCode::OK);
    let result = read_response(result).await?;
    assert!(
        result["queue_ms"]
            .as_f64()
            .is_some_and(|ms| ms.is_finite() && ms >= held_ms)
    );
    assert!(
        result["inference_ms"]
            .as_f64()
            .is_some_and(|ms| ms.is_finite() && ms >= 0.0)
    );
    worker.shutdown().await
}

#[actix_web::test]
async fn request_timeout_returns_before_running_inference_finishes() -> Result<()> {
    let mut configuration = config();
    configuration.request_timeout = Duration::from_millis(250);
    let (began, running) = oneshot::channel();
    let (release, gate) = sync_mpsc::channel();
    let completed = Arc::new(AtomicBool::new(false));
    let worker_completed = completed.clone();
    let worker = start(&configuration, move || {
        let mut began = Some(began);
        Ok(move |_| {
            if let Some(began) = began.take() {
                let _ = began.send(());
            }
            gate.recv_timeout(Duration::from_secs(5))
                .map_err(|_| worker_stopped())?;
            worker_completed.store(true, Ordering::SeqCst);
            Ok(response())
        })
    })
    .await?;
    let result = actix_web::rt::spawn(call(worker.state.clone()));
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .map_err(|_| JetError::NativeRuntime("fixture inference did not start".to_owned()))?
        .map_err(|_| worker_stopped())?;
    assert_eq!(
        result.await.map_err(|_| worker_stopped())??.status(),
        StatusCode::GATEWAY_TIMEOUT
    );
    assert!(!completed.load(Ordering::SeqCst));
    release.send(()).map_err(|_| worker_stopped())?;
    worker.shutdown().await?;
    assert!(completed.load(Ordering::SeqCst));
    Ok(())
}

#[actix_web::test]
async fn shutdown_rejects_waiting_work_and_waits_for_active_work() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let recorded = calls.clone();
    let (began, running) = oneshot::channel();
    let (release, gate) = sync_mpsc::channel();
    let worker = start(&config(), move || {
        let mut began = Some(began);
        Ok(move |_| {
            recorded.fetch_add(1, Ordering::SeqCst);
            if let Some(began) = began.take() {
                let _ = began.send(());
            }
            gate.recv_timeout(Duration::from_secs(5))
                .map_err(|_| worker_stopped())?;
            Ok(response())
        })
    })
    .await?;
    let (first, first_result) = job(Instant::now() + Duration::from_secs(5));
    worker
        .state
        .sender
        .send(first)
        .await
        .map_err(|_| worker_stopped())?;
    tokio::time::timeout(Duration::from_secs(5), running)
        .await
        .map_err(|_| JetError::NativeRuntime("fixture inference did not start".to_owned()))?
        .map_err(|_| worker_stopped())?;
    let (queued, queued_result) = job(Instant::now() + Duration::from_secs(5));
    worker
        .state
        .sender
        .send(queued)
        .await
        .map_err(|_| worker_stopped())?;
    let state = worker.state.clone();
    state.stop();
    let shutdown = actix_web::rt::spawn(worker.shutdown());
    assert_eq!(call(state).await?.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(!shutdown.is_finished());
    release.send(()).map_err(|_| worker_stopped())?;
    assert!(first_result.await.map_err(|_| worker_stopped())?.is_ok());
    assert!(matches!(
        queued_result.await.map_err(|_| worker_stopped())?,
        Err(ServiceError {
            status: StatusCode::SERVICE_UNAVAILABLE,
            ..
        })
    ));
    shutdown.await.map_err(|_| worker_stopped())??;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[actix_web::test]
async fn engine_errors_preserve_http_status_and_envelope() -> Result<()> {
    let worker = start(&config(), || {
        let mut count = 0;
        Ok(move |_| {
            count += 1;
            if count == 1 {
                Err(JetError::ContextExceeded {
                    required: 20,
                    limit: 10,
                })
            } else {
                Err(JetError::NativeRuntime("fixture error".to_owned()))
            }
        })
    })
    .await?;
    for expected in [StatusCode::BAD_REQUEST, StatusCode::INTERNAL_SERVER_ERROR] {
        let response = call(worker.state.clone()).await?;
        assert_eq!(response.status(), expected);
        let bytes = actix_web::body::to_bytes(response.into_body())
            .await
            .map_err(|error| JetError::NativeRuntime(error.to_string()))?;
        let body: Value = serde_json::from_slice(&bytes)?;
        assert_eq!(body["error"]["code"], expected.as_u16());
    }
    worker.shutdown().await
}
