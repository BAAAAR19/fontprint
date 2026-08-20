# Step-by-step guide

Every command below runs locally on CPU. A GPU or Apple Silicon device is used
automatically when PyTorch finds one, but nothing here requires it.

---

## Step 0 — Requirements

| Requirement | Notes |
|---|---|
| Python 3.11 – 3.14 | `python --version` |
| ~2 GB disk | PyTorch wheels dominate the install |
| Internet (once) | Only to install packages and fetch the pinned font set |

---

## Step 1 — Install

```bash
git clone https://github.com/BAAAAR19/fontprint.git
cd fontprint
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e '.[dev,demo]'
```

Check that the CLI is on your path:

```bash
fontprint --version
```

Every command has help text: `fontprint --help`, `fontprint train --help`.

---

## Step 2 — Get fonts

Fontprint learns from font files on your machine and never copies them into the
repository. Fetch the pinned 20-face Open Font License benchmark:

```bash
python scripts/fetch_ofl_fonts.py --destination data/fonts
```

This writes `data/fonts/manifest.json` with the upstream commit, source URL,
SHA-256 digest, and license file for every face, so a run can be reproduced later.

Confirm the faces are readable:

```bash
fontprint fonts --root data/fonts --limit 20
```

You should see 20 rows. Any directory of `.ttf`/`.otf` files works instead — pass
`--root` as many times as you like. With no `--root`, system font directories are used.

---

## Step 3 — Train a model

```bash
fontprint train --config configs/base.yaml --font-root data/fonts
```

What happens, in order:

1. Faces are discovered and the last four are **reserved before any optimizer step**.
2. Word and phrase crops are rendered on the fly with print/scan augmentation.
3. The encoder trains for 30 epochs with supervised contrastive loss plus an auxiliary
   classifier.
4. Retrieval quality is measured on seen faces, pair verification on the unseen faces.
5. Conformal calibration runs through the real inference path on 48 clean synthetic
   pages, so the stored threshold matches how inference actually scores a region.
6. Two files are written: `artifacts/fontprint.pt` and `artifacts/run.json`.

Expect roughly 25 minutes on an Apple M1. For a two-minute smoke run instead:

```bash
fontprint train --quick --font-root data/fonts
```

Tune anything in [`configs/base.yaml`](../configs/base.yaml) — `epochs`,
`embedding_dim`, `max_fonts`, `calibration_alpha`, and the rest are validated on load,
so a typo fails immediately instead of halfway through training.

---

## Step 4 — Create a test document

```bash
fontprint synthesize --output specimen.png --tampered
```

A fictional invoice is rendered with exactly one line set in a different face. Add
`--show-truth` to outline the substituted line, or `--clean` to render a consistent
page for a false-positive check. `--seed N` changes the layout noise.

---

## Step 5 — Analyze it

```bash
fontprint analyze specimen.png \
  --checkpoint artifacts/fontprint.pt \
  --overlay evidence.png
```

Evidence JSON goes to stdout and an annotated image to `evidence.png`:

```json
{
  "verdict": "typographic outlier detected",
  "review_recommended": true,
  "reference_region_id": 4,
  "threshold": 0.563,
  "correction": "bh",
  "regions": [
    {
      "region_id": 7,
      "box": [88, 337, 268, 374],
      "anomaly_score": 0.62,
      "p_value": 0.002,
      "adjusted_p_value": 0.022,
      "is_anomaly": true,
      "nearest_style": "Space Mono Regular",
      "style_similarity": 0.71
    }
  ],
  "caveat": "Fontprint measures typographic inconsistency; it does not establish that a document is authentic or fraudulent."
}
```

How to read it:

- **`anomaly_score`** — cosine distance from this region to the document's dominant style. Larger means less typographically similar.
- **`p_value`** — the conformal probability of seeing a distance this extreme among style-consistent regions. It is calibrated, not a softmax score.
- **`adjusted_p_value`** — the same p-value after correcting for the fact that a page tests many regions at once. This is what the decision uses.
- **`is_anomaly`** — true when `adjusted_p_value <= alpha` (0.05 by default) and the region is not itself the reference.
- **`correction`** — `bh` (default, bounds the share of flagged regions that are false), `holm` (no false flag anywhere on the page), or `none` (raw per-region level, more sensitive and noisier). Pass `--correction` to change it.
- **`reference_region_id`** — the medoid region every distance is measured against.
- **`nearest_style`** — descriptive context from the prototype index. It never drives the decision.

---

## Step 6 — Measure detection quality

```bash
fontprint benchmark \
  --checkpoint artifacts/fontprint.pt \
  --font-root data/fonts \
  --documents 40 --markdown
```

Alternating tampered and clean pages are scored end to end. The summary reports
document recall, false-positive rate, region precision/recall/F1, region AUROC, and
how the empirical flag rate on consistent text compares to the nominal alpha. Full
per-document rows land in `artifacts/benchmark.json`.

Expect the corrected default to be precise and quiet: on faces the model has never
seen it flags about one substitution in five, and roughly five in six of its flags are
right. Add `--correction none` to trade precision for sensitivity.

Two modes matter:

- `--oracle-boxes` (default) supplies ground-truth line boxes and isolates the encoder. Note that a full invoice line is far wider than the crops the encoder is trained on, so treat this mode as a ranking diagnostic rather than a decision benchmark.
- `--proposals` runs the OCR-free region proposer too, which is what deployment does.

For an honest open-set number, benchmark only on faces the model never trained on —
`artifacts/run.json` lists them under `holdout_fonts`:

```bash
fontprint benchmark -m artifacts/fontprint.pt --proposals --documents 40 \
  --font-root data/fonts/playfairdisplay --font-root data/fonts/rubik \
  --font-root data/fonts/spacemono --font-root data/fonts/spectral
```

---

## Step 7 — Review documents interactively

```bash
fontprint demo --checkpoint artifacts/fontprint.pt
```

Opens the Gradio evidence desk at `http://127.0.0.1:7860`: generate a specimen or
upload your own image, then read the overlay and JSON side by side.

---

## Step 8 — Serve the API

```bash
pip install -e '.[api]'
fontprint serve --checkpoint artifacts/fontprint.pt
```

```bash
curl -X POST 'http://localhost:8000/v1/analyze?include_overlay=false' \
  -F 'image=@specimen.png'
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness; always 200, reports whether a model is loaded |
| `GET /ready` | Readiness; 503 until a checkpoint is mounted |
| `GET /v1/model` | Embedding dimension, crop size, alpha, threshold, indexed styles |
| `POST /v1/analyze` | Multipart image in, evidence JSON out (`?include_overlay=true` adds a base64 PNG) |
| `GET /docs` | Interactive OpenAPI documentation |

Uploads are capped at 10 MB and 25 megapixels. No checkpoint means 503, never silent
random weights.

Containerized:

```bash
docker compose up --build      # expects artifacts/fontprint.pt to exist
```

---

## Step 9 — Export for deployment

```bash
pip install -e '.[export]'
fontprint export-onnx --checkpoint artifacts/fontprint.pt --output artifacts/fontprint.onnx
```

Produces an ONNX encoder with a dynamic batch axis plus a JSON sidecar holding crop
size, prototypes, calibration scores, and metrics — enough to reimplement inference in
a runtime with no Python.

---

## Step 10 — Develop

```bash
make install     # editable install with dev and demo extras
make lint        # ruff check, ruff format --check, mypy --strict
make test        # pytest
make coverage    # pytest with a coverage report
make benchmark   # detection benchmark against artifacts/fontprint.pt
```

Install the hooks once with `pre-commit install` so formatting runs before each commit.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `found N usable fonts, but training needs at least M` | Too few Latin faces. Run `python scripts/fetch_ofl_fonts.py`, then pass `--font-root data/fonts`. |
| `at least three text regions are required` | The proposer found fewer than three regions. Use a larger or higher-contrast image, or pass your own boxes through the Python API. |
| `unsupported checkpoint schema` | Checkpoint from a different `SCHEMA_VERSION`. Retrain, or check out the matching tag. |
| API returns 503 | No checkpoint mounted. Start with `--checkpoint`, or set `FONTPRINT_CHECKPOINT`. |
| Everything is flagged on a real scan | Calibration assumes exchangeability. Recalibrate on pages from your own capture pipeline, or raise `calibration_alpha`. |
| Tests skip with "fewer than two Latin fonts" | Headless machine without fonts. Install `fonts-dejavu` and `fonts-liberation`. |
