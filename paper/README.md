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
mise install tinytex
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
- `figures/generate_mmlu_chart.py`: regenerate the TikZ accuracy--throughput
  comparison with `python3 paper/figures/generate_mmlu_chart.py`, then rebuild
  with `mise run paper`.

Use `\label` and `\ref` for equations, sections, figures, and tables. Add figures
with `\includegraphics`; keep source figures outside `build/`.

The report is organized as Introduction, Related Work, Decision Method,
Efficient Inference, Experimental Evaluation, Discussion and Limitations,
and Conclusion, followed by Statements, references, and a short experimental protocol.
Local deployment on consumer CPU/GPU configurations is a central evaluation theme.

The draft compares two JET models on an Arc A770 and six on an RTX 4090
using full MMLU. The RTX 4090 comparison ranges from
Qwen3.5-0.8B (32.42%, 9.79 req/s) to Qwen3.5-27B (85.09%, 2.21 req/s), and
Qwen3.6-35B-A3B (82.62%, 4.17 req/s). Jev 1.13 (89.06%, 2.86 req/s via API)
and published larger-model scores provide context. The report formalizes sampler-free candidate
scoring, describes prefix sharing and state isolation, and consolidates
existing accuracy and performance measurements. See [evidence.md](evidence.md) for
the mapping from quantitative claims to source records and the remaining experiments.
The full-set Arc A770 and RTX 4090 rates are each measured on the same 14,042 requests as their
accuracy scores. The separate 3.693 req/s preparation result uses a 285-question
subset. Engineering details, commands, and artifact
provenance belong in the repository documentation rather than the manuscript.
A later Qwen3.6-35B-A3B CUDA full-set check measured 82.57% and 4.96 req/s
with a newer executable; it is reported separately from the uniform Vulkan
model comparison.

The primary full-MMLU runs were performed September 23--24, 2026, with a
separate CUDA check on September 28. Results retain their original
sample sizes, hardware, executor versions, and timing boundaries. The report does not
claim measured parity with Jev, calibrated probabilities, or cross-device bitwise
reproducibility.
