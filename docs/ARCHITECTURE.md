# Architecture

Fontprint separates representation learning from document-level decision logic. This keeps the learned component small and makes the final evidence traceable.

## Training path

1. `fonts.py` discovers local OpenType/TrueType faces and rejects non-Latin or decorative system resources.
2. `synthesis.py` renders lexically varied crops with rotation, blur, sensor noise, contrast changes, and JPEG degradation. Generation is deterministic per sample index. Phrases are one to three words with mixed casing and occasional currency amounts, and the canvas tracks the measured ink extent, so training crops carry the same width and aspect statistics as regions cut out of a real page.
3. `PKBatchSampler` supplies multiple examples from every selected font in a batch, which guarantees valid positives for supervised contrastive loss.
4. `TrainingModel` optimizes normalized embeddings with supervised contrastive loss and a lower-weight classification loss. The classifier is discarded at inference.
5. Validation embeddings from seen faces produce the known-style prototype index.
6. Faces reserved before optimization provide the open-set pair-verification evaluation. Separately seeded samples from those faces are shuffled into disjoint reference/query groups; each query's distance to its group's medoid becomes a word-level split-conformal calibration score.
7. Calibration then runs a second time *through the deployed inference path*: clean pages are rendered from the reserved faces, the OCR-free proposer cuts them into regions, and each region's distance to the page medoid becomes a calibration score. This page-matched distribution is the one stored in the checkpoint; the word-level threshold is kept in `metrics` only as a diagnostic.

### Why calibration is page-matched

Split-conformal validity requires calibration scores and test scores to be exchangeable. Word-crop groups are not exchangeable with inference: a page contributes upper-case fragments, currency, headers, and proposer artifacts that the crop sampler never produces. Calibrating on word crops made the nominal 5% level behave like 36% in the benchmark. Calibrating through `render_document -> propose_regions -> encoder -> medoid` brings the empirical flag rate on style-consistent text back in line with alpha.

## Inference path

The OCR-free proposer joins nearby dark connected components into word-like boxes. Each normalized crop passes through the encoder. Pairwise cosine distances identify the median-distance medoid, a real region representing the document's dominant typography.

Each region's distance to that medoid is converted into a conformal p-value:

```text
p = (1 + number of calibration scores >= observed score) / (n + 1)
```

A page carries one hypothesis per region, so those p-values are Holm-adjusted across the page before any decision is taken; the medoid is excluded from the family because it is the reference, not a hypothesis. A region is marked for review when its **adjusted** p-value is at or below alpha. Without the correction, eleven regions at a nominal 5% level produce a false flag on roughly 40% of clean pages; Holm holds the error rate at the page level and needs no independence assumption, which matters when every region is compared against one shared medoid. `--correction none` restores the uncorrected per-region behaviour for triage.

Note that a conformal p-value can never fall below `1 / (n + 1)`, so the size of the calibration set bounds how much evidence the correction can ever see. That is why calibration renders a few dozen pages rather than a handful.

The model also returns its nearest known prototype as descriptive context; that label does not control the anomaly decision.

## Evaluation path

`benchmark.py` scores the pipeline the way it is deployed rather than the way it is trained. It renders alternating tampered and clean pages, runs the analyzer, and reports document recall, document false-positive rate, region precision/recall/F1, region AUROC, and the empirical flag rate on style-consistent regions next to the nominal alpha. Boxes come either from ground truth (`--oracle-boxes`, isolating the encoder) or from the proposer (`--proposals`, measuring the whole system). Both modes report `proposal_coverage`, the fraction of ground-truth lines the OCR-free proposer recovered.

## Intentional design choices

- **Exact search over vector infrastructure:** a font catalog is normally small. NumPy cosine search is transparent and avoids an unnecessary service. FAISS/Qdrant becomes useful only at much larger catalog sizes.
- **Medoid over mean:** the reference is an observed region and is less sensitive to a minority of substitutions.
- **No OCR dependency:** this makes the core portable and prevents text content from becoming a shortcut. Production systems can replace the proposal layer with OCR/layout boxes.
- **No silent untrained fallback:** the API reports degraded readiness until a checkpoint is explicitly loaded.

## Extension points

- Swap `StyleEncoder` for a ConvNeXt or ViT encoder while retaining the analyzer contract.
- Add hard-negative mining among visually similar font families.
- Calibrate by capture domain (native PDF, mobile photo, scanner) using Mondrian conformal groups.
- Replace exact prototypes with FAISS when indexing tens of thousands of type styles.
- Use layout-aware region groups so headers, body text, and totals get separate dominant styles.
