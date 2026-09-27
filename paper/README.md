# JET: Justification Evaluation in Transformer

An English technical report draft, following the reference project's `mise + TinyTeX +
latexmk` workflow and split-section layout. The style uses standard LaTeX packages;
there is no conference-specific or copied `arxiv.sty` dependency.

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
- `preamble.tex`: typography, math, tables, figures, listings, links, and draft notes.
- `sections/*.tex`: independently editable report sections.
- `references.bib`: BibTeX references; cite with `\cite{key}`.
- `figures/workflow.tex`: editable TikZ vector diagram of the local decision workflow.
- `figures/mmlu-results.json`: accuracy and throughput measurements for the comparison figure.
- `figures/generate_mmlu_chart.py`: regenerate the native TikZ plot after changing its data:
  `python3 paper/figures/generate_mmlu_chart.py`, then `mise run paper`.

Use `\label` and `\ref` for equations, sections, figures, and tables. Add figures
with `\includegraphics`; keep source figures outside `build/`. The template contains
a scoring equation, a table, a code listing, and a citation to exercise the toolchain.

The report is organized as Introduction, Background and Related Work, Decision
Algorithm, System Architecture, Experimental Evaluation, Discussion and Limitations,
and Conclusion, followed by reproduction details. Local deployment on consumer
CPU/GPU configurations is a central evaluation theme.

The draft compares full-MMLU results for JET / Qwen3.6-35B-A3B (87.48%) and
Jev 1.13 (89.06%, 2.86 req/s via API) with published larger-model scores, formalizes sampler-free candidate
scoring, describes cache/state scheduling and the multimodal path, and consolidates
existing accuracy and performance measurements. See [evidence.md](evidence.md) for
the mapping from quantitative claims to source records and the remaining experiments.

No inference benchmarks were rerun to write this draft. Results retain their original
sample sizes, hardware, executor versions, and timing boundaries. The report does not
claim measured parity with Jev, calibrated probabilities, or cross-device bitwise
reproducibility.
