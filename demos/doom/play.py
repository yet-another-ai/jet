# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["vizdoom==1.3.1", "Pillow==12.3.0", "pygame-ce==2.5.8"]
# ///
"""Play Doom through Jet's image decision API while the game runs in real time."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import vizdoom as vzd
from dashboard import Dashboard
from jet_client import JetClient, JetClientError
from PIL import Image
from run_output import RunOutput

ROOT = Path(__file__).resolve().parents[2]
FRAME_SIZE = (160, 100)
BUTTON_NAMES = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT_RIGHT_DELTA",
    "ATTACK",
    "USE",
)
RELEASE = (0.0,) * len(BUTTON_NAMES)
OBSERVATION_SETTLE_SECONDS = 0.15


@dataclass(frozen=True)
class ActionPhase:
    buttons: tuple[float, ...]
    duration: float


@dataclass(frozen=True)
class Action:
    label: str
    phases: tuple[ActionPhase, ...]

    @property
    def buttons(self) -> tuple[float, ...]:
        return self.phases[0].buttons

    @property
    def duration(self) -> float:
        return sum(phase.duration for phase in self.phases)


# Delta turn values are degrees per game tic: negative left, positive right.
ACTIONS = {
    "forward": Action("Move forward", (ActionPhase((1, 0, 0, 0, 0, 0, 0), 0.5),)),
    "backward": Action("Move backward", (ActionPhase((0, 1, 0, 0, 0, 0, 0), 0.5),)),
    "strafe_left": Action("Strafe left", (ActionPhase((0, 0, 1, 0, 0, 0, 0), 0.5),)),
    "strafe_right": Action("Strafe right", (ActionPhase((0, 0, 0, 1, 0, 0, 0), 0.5),)),
    "turn_left": Action("Turn left", (ActionPhase((0, 0, 0, 0, -3, 0, 0), 0.25),)),
    "turn_right": Action("Turn right", (ActionPhase((0, 0, 0, 0, 3, 0, 0), 0.25),)),
    "fire": Action("Fire weapon", (ActionPhase((0, 0, 0, 0, 0, 1, 0), 0.5),)),
    "use": Action(
        "Use door",
        (
            ActionPhase((1, 0, 0, 0, 0, 0, 0), 0.5),
            ActionPhase((0, 0, 0, 0, 0, 0, 1), 0.1),
        ),
    ),
    "wait": Action("Stay still", (ActionPhase(RELEASE, 0.5),)),
}


@dataclass(frozen=True)
class PendingFrame:
    request_id: str
    episode: int
    captured_at: float
    game_tic: int
    observation: dict | None = None


@dataclass
class Controls:
    episode: int = 1
    previous_action: str = "wait"
    deadline: float | None = None
    remaining_phases: list[ActionPhase] = field(default_factory=list)
    observe_after: float = 0.0

    def reset(self) -> tuple[float, ...]:
        self.episode += 1
        self.previous_action = "wait"
        self.deadline = None
        self.remaining_phases.clear()
        self.observe_after = 0.0
        return RELEASE

    def release_due(self, now: float) -> tuple[float, ...] | None:
        """Advance a macro or release its last input, without blocking the game."""
        if self.deadline is not None and now >= self.deadline:
            if self.remaining_phases:
                phase = self.remaining_phases.pop(0)
                # Time each phase from when it is actually sent. A late game
                # tic must not skip the USE press at the end of an approach.
                self.deadline = now + phase.duration
                return phase.buttons
            self.deadline = None
            self.observe_after = now + OBSERVATION_SETTLE_SECONDS
            return RELEASE
        return None

    def ready_for_observation(self, now: float) -> bool:
        return self.deadline is None and now >= self.observe_after

    def apply(
        self,
        frame: PendingFrame,
        choice: str,
        now: float,
        max_age: float,
    ) -> tuple[float, ...] | None:
        if frame.episode != self.episode or now - frame.captured_at > max_age:
            return None
        if not isinstance(choice, str) or choice not in ACTIONS:
            raise ValueError(f"Jet returned an unknown action: {choice!r}")
        action = ACTIONS[choice]
        self.previous_action = choice
        self.remaining_phases = list(action.phases[1:])
        self.deadline = now + action.phases[0].duration
        return action.buttons


@dataclass
class VisualProgress:
    baseline: bytes | None = None
    action: str | None = None
    unchanged_moves: int = 0
    unchanged_uses: int = 0

    @staticmethod
    def signature(png: bytes) -> bytes:
        # Compare scenery above the weapon/HUD, using only pixels sent to Jet.
        with Image.open(io.BytesIO(png)) as image:
            return image.crop((0, 0, 160, 60)).resize((32, 12)).convert("L").tobytes()

    def begin(self, png: bytes, action: str) -> None:
        # Baseline is the live frame just before execution, not the older frame
        # on which the model based its decision.
        self.baseline = self.signature(png)
        self.action = action

    def observe(self, png: bytes) -> dict:
        if self.baseline is not None:
            signature = self.signature(png)
            change = sum(abs(a - b) for a, b in zip(signature, self.baseline)) / len(
                signature
            )
            moving = self.action in {
                "forward",
                "backward",
                "strafe_left",
                "strafe_right",
            }
            self.unchanged_moves = (
                self.unchanged_moves + 1 if moving and change < 3 else 0
            )
            self.unchanged_uses = (
                self.unchanged_uses + 1 if self.action == "use" and change < 3 else 0
            )
            # Count each executed action once, even if later responses are stale.
            self.baseline = None
            self.action = None
        return {
            "unchanged_moves": self.unchanged_moves,
            "unchanged_uses": self.unchanged_uses,
        }


@dataclass
class DashboardState:
    episode: int = 1
    decision: dict | None = None

    def reset(self, episode: int) -> None:
        self.episode = episode
        self.decision = None

    def complete(self, record: dict) -> None:
        # An old in-flight response must never restore the previous episode's scores.
        if record["episode"] == self.episode:
            self.decision = record

    def snapshot(
        self, state, *, thinking: bool, executing: bool = False, settling: bool = False
    ) -> dict:
        decision = self.decision or {}
        pixels = state.screen_buffer
        return {
            "frame_rgb": pixels.tobytes(),
            "frame_size": (pixels.shape[1], pixels.shape[0]),
            "episode": self.episode,
            "game_tic": state.tic,
            "frame_id": decision.get("request_id"),
            "decision_game_tic": decision.get("game_tic"),
            "choice": decision.get("choice"),
            "probabilities": decision.get("probabilities"),
            "applied": decision.get("applied", False),
            "latency_seconds": decision.get("latency_seconds"),
            "thinking": thinking,
            "executing": executing,
            "settling": settling,
            "health": float(state.game_variables[0]),
            "ammo": float(state.game_variables[1]),
        }


def encode_frame(image: Image.Image) -> bytes:
    """The transmitted PNG is exactly 160x100 RGB, including the HUD."""
    image = image.convert("RGB").resize(FRAME_SIZE, Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def decision_request(
    request_id: str, png: bytes, state: dict, *, warmup: bool = False
) -> dict:
    instructions = (
        "The game has not started. Choose Stay still."
        if warmup
        else "Reach the Doom exit. Move through open paths, turn when blocked, use nearby doors, "
        "and aim/fire at enemies. "
        "If unchanged_moves or unchanged_uses >= 2, turn or strafe instead of repeating. "
        "Keep exploring; avoid waiting."
    )
    return {
        "request_id": request_id,
        "state": state,
        "images": [
            {
                "id": "screen",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(png).decode("ascii"),
                },
            }
        ],
        "questions": {
            "action": {
                "type": "choice",
                "instructions": instructions,
                "criteria": {key: action.label for key, action in ACTIONS.items()},
            }
        },
    }


def selected_action(response: dict) -> str:
    answers = response.get("answers")
    answer = answers.get("action") if isinstance(answers, dict) else None
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("Jet response must contain an action answer of type choice")
    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in ACTIONS:
        raise ValueError(f"Jet returned an unknown action: {choice!r}")
    return choice


def configure_game(args: argparse.Namespace) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.set_doom_config_path(str((args.output_dir / "doom.ini").resolve()))
    if args.iwad:
        game.set_doom_game_path(str(args.iwad.resolve()))
    else:
        # Use the freely redistributable assets bundled with the pinned wheel.
        game.set_doom_game_path(str(Path(vzd.__file__).parent / "freedoom2.wad"))
    game.set_doom_map(args.map)
    game.set_doom_skill(args.skill)
    game.set_mode(vzd.Mode.ASYNC_PLAYER)
    game.set_ticrate(35)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X400)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_render_hud(True)
    game.set_render_crosshair(True)
    game.set_render_all_frames(True)
    # The dashboard displays the RGB buffer alongside model probabilities.
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_available_buttons([getattr(vzd.Button, name) for name in BUTTON_NAMES])
    game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, 3.0)
    game.set_available_game_variables(
        [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.SELECTED_WEAPON_AMMO,
        ]
    )
    game.set_episode_start_time(10)
    game.set_episode_timeout(0)
    game.set_seed(args.seed)
    return game


def default_binary() -> Path:
    name = "jet.exe" if os.name == "nt" else "jet"
    target = Path(os.environ.get("CARGO_TARGET_DIR", str(ROOT / "target")))
    for profile in ("release", "debug"):
        path = target / profile / name
        if path.is_file():
            return path
    return target / "release" / name


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=default_binary())
    parser.add_argument(
        "--model-path", type=Path, default=ROOT / "models/Qwen3.6-35B-A3B-Q4_K_M.gguf"
    )
    parser.add_argument(
        "--mmproj-path",
        type=Path,
        default=ROOT / "models/mmproj-Qwen3.6-35B-A3B-Q8_0.gguf",
    )
    parser.add_argument("--backend", choices=("vulkan", "cpu"), default="vulkan")
    parser.add_argument(
        "--iwad", type=Path, help="Own Doom IWAD; defaults to bundled Freedoom2"
    )
    parser.add_argument(
        "--map", default="map01", help="Use E1M1 for Doom, map01 for Doom II/Freedoom2"
    )
    parser.add_argument("--skill", type=int, choices=range(1, 6), default=1)
    parser.add_argument(
        "--decision-hz",
        type=positive_float,
        default=2.0,
        help="Maximum request rate; slow inference never queues old frames",
    )
    parser.add_argument(
        "--max-frame-age",
        type=positive_float,
        default=3.0,
        help="Discard decisions older than this many seconds",
    )
    parser.add_argument("--request-timeout", type=positive_float, default=120.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Seconds of gameplay after model warmup; 0 runs until closed",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Stop after this many episodes; 0 restarts indefinitely",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "target/doom" / datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f"),
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration < 0 or args.episodes < 0:
        parser.error("--duration and --episodes must be non-negative")
    for name in ("binary", "model_path", "mmproj_path", "iwad"):
        path = getattr(args, name)
        if path is not None and not path.is_file():
            parser.error(f"--{name.replace('_', '-')}: file not found: {path}")
    return args


def jet_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.binary.resolve()),
        "judge-multimodal",
        "--model-path",
        str(args.model_path.resolve()),
        "--mmproj-path",
        str(args.mmproj_path.resolve()),
        "--backend",
        args.backend,
        "--context-tokens",
        "2048",
        "--max-sequences",
        "2",
        "--micro-batch",
        "256",
        "--threads",
        "8",
        "--max-images",
        "1",
        "--image-max-tokens",
        "256",
        "--no-mmap",
    ]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = jet_command(args)
    write_json(
        output / "config.json",
        {
            "command": command,
            "iwad": str(args.iwad) if args.iwad else "bundled Freedoom2",
            "map": args.map,
            "skill": args.skill,
            "seed": args.seed,
            "frame_size": FRAME_SIZE,
            "decision_hz_limit": args.decision_hz,
            "max_frame_age_seconds": args.max_frame_age,
            "observation_settle_seconds": OBSERVATION_SETTLE_SECONDS,
            "vizdoom_version": vzd.__version__,
            "buttons": BUTTON_NAMES,
            "actions": {
                key: {
                    "label": value.label,
                    "buttons": value.buttons,
                    "duration_seconds": value.duration,
                    "phases": [
                        {"buttons": phase.buttons, "duration_seconds": phase.duration}
                        for phase in value.phases
                    ],
                }
                for key, value in ACTIONS.items()
            },
        },
    )
    print(f"Loading Jet; logs: {output}", flush=True)
    client = JetClient(
        command, output / "jet.log", timeout_seconds=args.request_timeout
    )
    game = None
    dashboard = None
    dashboard_state = DashboardState()
    run_output = None
    started = None
    warmup_seconds = None
    controls = Controls()
    progress = VisualProgress()
    pending = None
    submitted = applied = discarded = deaths = 0
    latencies = []
    status = "completed"
    try:
        before = time.perf_counter()
        client.submit(
            decision_request(
                "warmup",
                encode_frame(Image.new("RGB", FRAME_SIZE)),
                {},
                warmup=True,
            )
        )
        while True:
            result = client.poll()
            if result is not None:
                selected_action(result.response)
                break
            time.sleep(0.02)
        warmup_seconds = time.perf_counter() - before
        print(
            f"Model ready in {warmup_seconds:.2f}s. Starting {args.map}; Ctrl+C to stop.",
            flush=True,
        )
        game = configure_game(args)
        game.init()  # Starts the map directly, without opening menus.
        game.set_action(list(RELEASE))
        run_output = RunOutput(output, save_frames=args.save_frames)
        if not args.headless:
            dashboard = Dashboard(
                {key: action.label for key, action in ACTIONS.items()},
                screenshot_path=output / "dashboard.png",
            )
        started = time.perf_counter()
        next_request_at = started
        next_ui_at = started
        if run_output is not None:
            while game.is_running():
                now = time.perf_counter()
                if dashboard is not None and dashboard.stop_requested:
                    if dashboard.error is not None:
                        raise RuntimeError(dashboard.error)
                    break
                if args.duration and now - started >= args.duration:
                    break
                if game.is_episode_finished() or game.is_player_dead():
                    dead = game.is_player_dead()
                    deaths += int(dead)
                    game.set_action(list(RELEASE))
                    if args.episodes and controls.episode >= args.episodes:
                        break
                    game.new_episode()
                    game.set_action(list(controls.reset()))
                    progress = VisualProgress()
                    dashboard_state.reset(controls.episode)
                    next_ui_at = 0.0
                    # Keep any old request in flight; drain and discard its eventual result.
                    next_request_at = time.perf_counter()
                    run_output.message(
                        f"Episode {controls.episode} started ({'death' if dead else 'map ended'})."
                    )
                    continue

                release = controls.release_due(now)
                if release is not None:
                    game.set_action(list(release))
                # In ASYNC_PLAYER the engine advances independently. This refreshes its
                # cached state and paces the control loop at roughly one game tic.
                game.advance_action(1, True)
                if game.is_episode_finished() or game.is_player_dead():
                    continue
                now = time.perf_counter()
                release = controls.release_due(now)
                if release is not None:
                    game.set_action(list(release))

                if dashboard is not None and now >= next_ui_at:
                    state = game.get_state()
                    if state is not None:
                        dashboard.publish(
                            dashboard_state.snapshot(
                                state,
                                thinking=client.busy,
                                executing=controls.deadline is not None,
                                settling=now < controls.observe_after,
                            )
                        )
                    next_ui_at = now + 1.0 / 20

                result = client.poll()
                if result is not None:
                    if pending is None or result.request_id != pending.request_id:
                        raise ValueError("Decision does not match the in-flight frame")
                    choice = selected_action(result.response)
                    before_action = game.get_state()
                    baseline = (
                        encode_frame(Image.fromarray(before_action.screen_buffer))
                        if before_action is not None
                        else None
                    )
                    # Start the input deadline after preparing the baseline so
                    # image processing cannot shorten the actual button hold.
                    now = time.perf_counter()
                    buttons = controls.apply(pending, choice, now, args.max_frame_age)
                    frame_age = now - pending.captured_at
                    latencies.append(result.latency_seconds)
                    if buttons is None:
                        discarded += 1
                    else:
                        game.set_action(list(buttons))
                        if baseline is not None:
                            progress.begin(baseline, choice)
                        applied += 1
                    record = {
                        "request_id": pending.request_id,
                        "episode": pending.episode,
                        "game_tic": pending.game_tic,
                        "choice": choice,
                        "applied": buttons is not None,
                        "latency_seconds": result.latency_seconds,
                        "frame_age_seconds": frame_age,
                        "usage": result.response.get("usage"),
                        "observation": pending.observation,
                        "probabilities": result.response["answers"]["action"].get(
                            "probabilities"
                        ),
                    }
                    run_output.record(record)
                    dashboard_state.complete(record)
                    next_ui_at = 0.0
                    pending = None
                    # Observe only after the full action and settling interval.
                    continue

                if (
                    not client.busy
                    and now >= next_request_at
                    and controls.ready_for_observation(now)
                ):
                    state = game.get_state()
                    if state is None:
                        continue
                    captured_at = time.perf_counter()
                    png = encode_frame(Image.fromarray(state.screen_buffer))
                    submitted += 1
                    request_id = f"episode-{controls.episode}-frame-{submitted}"
                    hud = {
                        "health": float(state.game_variables[0]),
                        "ammo": float(state.game_variables[1]),
                        "previous_action": controls.previous_action,
                        **progress.observe(png),
                    }
                    pending = PendingFrame(
                        request_id, controls.episode, captured_at, state.tic, hud
                    )
                    client.submit(decision_request(request_id, png, hud))
                    run_output.frame(submitted, png)
                    next_request_at = captured_at + 1.0 / args.decision_hz
    except KeyboardInterrupt:
        status = "interrupted"
    except Exception:
        status = "failed"
        raise
    finally:
        elapsed = time.perf_counter() - started if started is not None else 0.0
        try:
            if game is not None:
                try:
                    if game.is_running():
                        game.set_action(list(RELEASE))
                finally:
                    game.close()
        finally:
            try:
                client.close()
            finally:
                try:
                    if dashboard is not None:
                        dashboard.close()
                finally:
                    if run_output is not None:
                        run_output.close()
            write_json(
                output / "summary.json",
                {
                    "status": status,
                    "warmup_seconds": warmup_seconds,
                    "play_seconds": elapsed,
                    "episodes_started": controls.episode if started else 0,
                    "deaths": deaths,
                    "frames_submitted": submitted,
                    "decisions_applied": applied,
                    "stale_decisions_discarded": discarded,
                    "median_latency_seconds": statistics.median(latencies)
                    if latencies
                    else None,
                    "decisions_per_second": applied / elapsed if elapsed else 0.0,
                    "frame_size": FRAME_SIZE,
                    "diagnostic_events_dropped": run_output.dropped_events
                    if run_output
                    else 0,
                    "diagnostic_error": str(run_output.error)
                    if run_output and run_output.error
                    else None,
                },
            )
            print(
                f"Stopped: {applied} actions, {discarded} stale decisions. Results: {output}",
                flush=True,
            )


if __name__ == "__main__":
    try:
        run(parse_args())
    except (JetClientError, ValueError, OSError, RuntimeError) as error:
        print(f"doom demo: {error}", file=sys.stderr)
        raise SystemExit(1) from error
