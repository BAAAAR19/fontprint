# Architecture

Fontprint separates representation learning from document-level decision logic. This keeps the learned component small and makes the final evidence traceable.

## Training path

1. `fonts.py` discovers local OpenType/TrueType faces and rejects non-Latin or decorative system resources.
2. `synthesis.py` renders lexically varied crops with rotation, blur, sensor noise, contrast changes, and JPEG degradation. Generation is deterministic per sample index.
3. `PKBatchSampler` supplies multiple examples from every selected font in a batch, which guarantees valid positives for supervised contrastive loss.
4. `TrainingModel` optimizes normalized embeddings with supervised contrastive loss and a lower-weight classification loss. The classifier is discarded at inference.
5. Validation embeddings from seen faces produce the known-style prototype index.
6. Faces reserved before optimization provide the open-set pair-verification evaluation. Separately seeded samples from those faces are shuffled into disjoint reference/query groups; each query's distance to its group's medoid becomes a split-conformal calibration score.

## Inference path

The OCR-free proposer joins nearby dark connected components into word-like boxes. Each normalized crop passes through the encoder. Pairwise cosine distances identify the median-distance medoid, a real region representing the document's dominant typography.

Each region's distance to that medoid is converted into a conformal p-value:

```text
p = (1 + number of calibration scores >= observed score) / (n + 1)
```

A region is marked for review when `p <= alpha`. The model also returns its nearest known prototype as descriptive context; that label does not control the anomaly decision.

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
