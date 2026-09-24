# Doom demo

Jet chooses a short action from a 160×100 RGB screenshot while Doom runs continuously.
The default game is the Freedoom2 campaign bundled with ViZDoom. `init()` enters `map01`
directly and skips the opening weapon animation; no menu automation is needed.

## Run

Build Jet with both vision and Vulkan, and obtain the matching language model and projector:

```sh
./scripts/download-accuracy-models.sh --qwen36-vision
mise exec -- cargo build --release -p jet-cli --features vision,vulkan
mise exec uv@0.12.13 -- uv run --locked --script demos/doom/play.py
```

For CUDA, build with `--features vision,cuda` and run the demo with `--backend cuda`.
Build with `--features vision,cuda,vulkan` to let the default `auto` backend choose CUDA,
then Vulkan, then CPU.

The script declares Python 3.12 and pins ViZDoom, Pillow, and pygame-ce. `play.py.lock` also locks the transitive
dependencies; uv installs the runtime and dependencies on first use. The model files are the
same ones used by `judge-multimodal`. Run from the repository root.

On Windows, use PowerShell 7, configure the Vulkan SDK, and keep Cargo's target path short:

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.357.0' # Adjust to your installed SDK.
$env:PATH = "$env:VULKAN_SDK\Bin;$env:PATH"
$env:CARGO_TARGET_DIR = 'E:\jet-target'
mise exec -- cargo build --release -p jet-cli --features vision,vulkan
mise exec uv@0.12.13 -- uv run --locked --script demos/doom/play.py
```

`--binary PATH` selects an existing Jet binary; otherwise the script searches
`$CARGO_TARGET_DIR/{release,debug}` or the repository's `target/{release,debug}`.
The dashboard opens after a real warmup request finishes. Stop with Esc, the window close
button, or Ctrl+C. `--headless` runs without the dashboard.

## Live dashboard

The live 640×400 game view appears beside nine action confidence bars. Every completed
decision updates all candidate probabilities, highlights the model's choice, and shows
the inference latency and source frame ID. Health, ammo, and the episode are also visible.
The status distinguishes inference, action execution, and waiting to observe the result.

Confidence means relative probability **among these nine candidates**, not a calibrated
probability of success. While the model is thinking, the panel retains its last decision;
the highlight does not mean the corresponding button is still held. Missing scores appear
as `--`. A result discarded because its frame expired is marked as not applied. A new
episode clears the scores, and responses from earlier episodes cannot restore them.

Rendering and window events run in a separate process. The game publishes complete snapshots
at up to 20 Hz through a bounded, nonblocking queue, so a stalled UI cannot extend a button
hold or build a frame backlog. Closing the dashboard also stops the hidden Doom engine and Jet.

To use your own original Doom assets:

```sh
mise exec uv@0.12.13 -- uv run --locked --script demos/doom/play.py --iwad /path/to/doom.wad --map E1M1
```

Use `--map map01` for Doom II or Freedoom2. Original Doom WADs are not included in this repository.

## Controls and timing

| Choice | Input | Approximate maximum hold |
| --- | --- | --- |
| Forward / Backward | Move forward / backward | 0.5 s |
| Strafe left / Strafe right | Move sideways | 0.5 s |
| Turn left / Turn right | −3° / +3° per game tic | 0.25 s, about 27° |
| Fire | Attack | 0.5 s |
| Use door | Walk forward, then press USE | 0.5 s forward + 0.1 s USE |
| Wait | Release all inputs | Immediate |

The game uses `ASYNC_PLAYER` at 35 tic/s. A single Jet process stays loaded. The main thread
updates Doom and releases buttons independently of the inference worker; at most one image is
in flight. After all action phases finish, the controller releases the inputs and waits
0.15 seconds before capturing the next observation. This prevents the model from judging
the result of a movement that has only just started. There is no backlog
of screenshots and no catch-up burst after slow inference.

`--decision-hz 2` limits submissions to two per second; actual throughput depends on inference
latency plus action execution and observation settling. The nine action candidates cost
more than a two-choice image test. Button deadlines
are checked each game tic, so they have normal operating-system scheduling jitter.

The game renders at 640×400, then Pillow downsamples to an exact **160×100** PNG. Jet's model-specific
visual processor may subsequently resize or pad that image to its patch grid. The request also
contains HUD health/ammo, the previous action, and counts of consecutive moves or door uses
that barely changed the scenery pixels. Each comparison uses the image immediately before
the executed action and the image after it settles, counting that action only once. This
helps the model notice blocked paths. No enemy positions,
map geometry, or other hidden game state is supplied. This is a visual control demo, not a trained Doom policy or a
claim that the model can complete the level.

The model and dashboard retain the simple candidate label `Use door`. Its controller action
walks closer before pressing USE, so selecting a visible door outside interaction range
does not repeatedly press USE from the same position. The prompt does not describe this
candidate as an alternative movement action. This does not guarantee correct handling of
locked doors or a door the player is not facing.

Death and map completion restart the selected map. Each restart changes the episode ID;
decisions from the previous episode are discarded. Decisions older than `--max-frame-age 3`
seconds are also discarded. This age limit is independent of the button hold duration.

## Results and checks

Useful options:

```sh
# Run a bounded, headless model smoke test and save each submitted image.
mise exec uv@0.12.13 -- uv run --locked --script demos/doom/play.py --headless --duration 30 --save-frames
```

`--output-dir` chooses a new results directory; the default is `target/doom/<timestamp>`.
It contains configuration, `jet.log`, `decisions.jsonl`, `last-frame.png`, and `summary.json`.
Visible runs also save `dashboard.png`, the latest rendered decision panel.
`--save-frames` also retains numbered input PNGs. Diagnostic writes and console output run on
a separate bounded worker; slow outputs can drop diagnostic events without delaying game
control. The summary reports dropped diagnostics, decision latency, actual action rate,
restarts, and stale responses.

The summary records model latency separately from applied decisions per second. The latter
also includes action execution and the observation settling interval; `--decision-hz 2`
remains a scheduling limit, not a measured throughput guarantee.

Fast tests use fake Jet subprocesses and verify real pipe behavior, deadlines, stale episode
rejection, image dimensions, nonblocking diagnostics, and dashboard rendering/shutdown:

```sh
mise exec uv@0.12.13 -- uv run --python 3.12 --with vizdoom==1.3.1 --with Pillow==12.3.0 --with pygame-ce==2.5.8 python -m unittest discover -s demos/doom -v
```

Set `JET_DOOM_ENGINE_SMOKE=1` to also run the real headless Doom engine against a deliberately
slow fake model, plus a real Freedoom2 door regression: five USE presses at a distance fail,
while the controller's forward/USE phases open the same door. These verify engine behavior,
not the model's ability to choose useful actions or complete a level. The smoke tests need a
graphical display on platforms where the ViZDoom
binary requires one, even with `--headless`.

Upstream references: [ViZDoom API](https://vizdoom.farama.org/api/python/doom_game/),
[pinned ViZDoom release](https://pypi.org/project/vizdoom/1.3.1/),
[Freedoom](https://freedoom.github.io/).
