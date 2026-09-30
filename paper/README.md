# JET: Justification Evaluation in Transformer

An English technical report draft using the single-column
[arxiv-style preprint template](https://github.com/kourgeorge/arxiv-style), with a
`mise + TinyTeX + latexmk` workflow and split-section layout. The unmodified
`arxiv.sty` is vendored from upstream commit
[`920514696f6e7270cab0558fd97c44515c63b4c4`](https://github.com/kourgeorge/arxiv-style/tree/920514696f6e7270cab0558fd97c44515c63b4c4);
its MIT license is included in `arxiv-LICENSE.txt`.

## Build

From the repository root:

```sh
mise install vfox:tinytex
mise run paper
```

Output: `paper/build/main.pdf`. `latexmk` runs pdfLaTeX, BibTeX, and the required
cross-reference passes automatically. Build errors stop the task. Generated files
stay in the ignored `paper/build/` directory. The toolchain does not require a Rust
build, a model download, or a GPU. TinyTeX installation requires network access;
once the TeX packages are installed, ordinary builds work offline.

The build has been verified on Linux x86-64. The current mise TinyTeX plugin
selects Linux x86-64 or macOS binaries; its lock entries are limited to those
platforms. On Windows or Linux ARM, use a native TeX Live installation and run
`latexmk main.tex` from `paper/` instead.

A portable Tectonic 0.17.0 executable is available locally for Windows at
`target/length-audit/tools/tectonic-0.17.0/tectonic.exe`. It requires no global
installation. From the repository root:

```powershell
& ./target/length-audit/tools/tectonic-0.17.0/tectonic.exe --keep-logs --keep-intermediates --outdir paper/build paper/main.tex
```

Tectonic downloads its TeX bundle on first use and then reuses its local cache.
The portable executable and generated build files are ignored by Git.

If a minimal TinyTeX installation reports a missing package, install the template's
package set and rebuild:

```sh
mise exec -- tlmgr install latexmk collection-latexrecommended collection-fontsrecommended
mise run paper
```

The template uses numeric citations with `natbib` and BibTeX.
TinyTeX is resolved through `mise.lock`; additional `tlmgr` package installations
are not pinned by that lockfile. Archive the TeX environment for archival builds.

To remove the PDF and intermediate build files:

```sh
mise run paper:clean
```

For live rebuilding, run `mise exec -- latexmk -pvc main.tex` from `paper/`.
An existing TeX Live installation with the same packages can instead run
`latexmk main.tex` directly from that directory.

## Edit

- `main.tex`: title, author metadata, section order, and bibliography.
- `arxiv.sty`: vendored preprint template; controls fonts, page geometry, and headings.
- `preamble.tex`: typography, math, tables, figures, listings, links, and draft notes.
- `sections/*.tex`: independently editable report sections.
- `sections/statements.tex`: author statements, including AI-assisted writing disclosure.
- `references.bib`: BibTeX references; cite with `\cite{key}`.
- `figures/workflow.tex`: editable TikZ vector diagram of the local decision workflow.
- `figures/mmlu-results.json`: accuracy and throughput data for Figure 2.
- `figures/a770-results.json`: validated Arc A770 measurements used by the
  full-MMLU table and the accuracy--throughput figure.
- `figures/rtx4090-results.tex`: generated numeric macros for tables and prose.
- `figures/a770-results.tex`: generated numeric macros for the A770 table.
- `figures/rtx4090-evidence.json`: validated aggregate results and paired comparisons.
- `update_results.py`: validate a complete rerun campaign and regenerate its
  manuscript macros, evidence, and figure data; `--pending` initializes placeholders.
- `figures/generate_mmlu_chart.py`: regenerate the TikZ accuracy--throughput
  comparison with `python3 paper/figures/generate_mmlu_chart.py`, then rebuild
  with `mise run paper`.

Use `\label` and `\ref` for equations, sections, figures, and tables. Add figures
with `\includegraphics`; keep source figures outside `build/`.

The report is organized as Introduction, Related Work, Decision Method,
Efficient Inference, Experimental Evaluation, Task-Aligned Fine-Tuning,
Discussion and Limitations, and Conclusion, followed by Statements, references,
and a short experimental protocol.
Local deployment on consumer GPUs is a central evaluation theme.

The report now uses one candidate-scoring method: mean conditional token
log-probability followed by candidate softmax. Every active JET measurement
comes from a completed fresh run with that method; historical measurements
are not reused.

The completed RTX 4090 campaign covers six pretrained models with both CUDA and Vulkan,
plus the existing Qwen3.5-4B LoRA export with CUDA: thirteen full runs of the
same 14,042 MMLU questions. Eighteen controls on a fixed 570-question subset
check paired backend differences, selected Vulkan options, and fresh-process
repetition. All 31 cases completed on September 29--30, 2026 (UTC).
The LoRA adapter and training provenance remain unchanged; both reference
and adapted inference were rerun.

Manuscript numbers come from generated TeX macros rather than hand-maintained
copies. The result exporter validates the completed campaign before updating
the tables, derived comparisons, and figure. See [evidence.md](evidence.md)
for the artifact paths, validation requirements, and provenance boundaries.
Each displayed JET figure point combines accuracy and complete-process
throughput from the same full run. No partial measurement supplies a point.

The Arc A770 full-set comparison for Qwen3.5-0.8B, 2B, 4B, and 9B is complete
and appears in the table and figure. The shared-prefix study remains pending.

On the Linux A770 host, synchronize this working tree (including the scoring
change) and rebuild the local Vulkan executable before running the four full
models. From the repository root, with the Vulkan build dependencies and model
files available:

```sh
mise exec rust@1.98 -- cargo build --release --locked -p jet-cli --features vulkan
JET_A770_RUN_TAG=mean-20260930-v3 JET_A770_BINARY=target/release/jet \
  bash scripts/run-a770-full-mmlu.sh 08b 2b 4b 9b
```

The completed campaign used this tag. Choose a fresh tag for any new run. The
541-request prefix comparison described in the appendix is a separate follow-up.

Jev 1.13 (author-supplied 89.06% and 2.86 requests/s via API) and published
larger-model scores remain external references with their original protocols.
The report does not claim measured parity with Jev, calibrated confidence,
LoRA-only causality, a repeatable speedup from one run, or cross-device bitwise
reproducibility.
