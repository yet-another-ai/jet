# Accuracy evaluation

This evaluation maps public labelled datasets onto Jet's decision protocol:

- BoolQ validation examples become `noul` questions.
- MMLU test examples become `choice` questions with `A` through `D` candidates.

The source archives, extracted data, generated requests, gold labels, responses, and reports are
intentionally ignored by Git. The preparation script verifies the normalized BoolQ JSONL and the
MMLU archive with pinned SHA-256 digests before using them.

Run the reproducible default sample (256 balanced BoolQ examples and 5 examples from each of the
57 MMLU subjects):

```sh
./scripts/run-accuracy-eval.sh
```

Override sample sizes with `JET_BOOLQ_LIMIT` and `JET_MMLU_PER_SUBJECT`. For example, the complete
MMLU test set has unequal subject sizes, so use the preparation script directly if a sampling rule
other than a fixed count per subject is needed.

The primary metric is top-1 accuracy. Mean negative log-likelihood of the gold label is included as
a diagnostic. These results are not directly comparable to the official MMLU leaderboard because
Jet uses its own Tera-rendered zero-shot instruction prompt and semantic candidate scoring.

See [the recorded baseline](../../docs/accuracy.md) for the latest checked-in result and its
interpretation.

## Sources and licenses

- [BoolQ](https://github.com/google-research-datasets/boolean-questions), CC BY-SA 3.0. The labelled
  validation split is retrieved through the Hugging Face dataset server because the original GCS
  link no longer permits anonymous downloads.
- [MMLU](https://github.com/hendrycks/test), MIT. The data archive is downloaded from the URL in the
  upstream repository.

Retain these attributions when redistributing derived fixtures.
