# Model card: Fontprint StyleEncoder v0.1

## Summary

StyleEncoder is a compact residual convolutional network that maps a grayscale word crop to a normalized vector. It is trained to reduce distance between different words rendered in the same font face and increase distance between faces. Document inference compares regions to a robust within-document medoid.

## Intended uses

- Research on font-style representation and open-set recognition.
- Human-in-the-loop document quality assurance.
- Prioritizing regions for a trained forensic reviewer.
- Educational demonstrations of metric learning and conformal calibration.

## Out-of-scope uses

- Authentication or fraud decisions without corroborating evidence.
- Identity, authorship, or intent attribution.
- Legal, credit, insurance, employment, or access decisions.
- Handwriting analysis, writer identification, or non-Latin typography in v0.1.

## Training objective

The primary objective is supervised contrastive loss over P×K batches. An auxiliary linear classifier adds class-separating pressure during training and is removed for inference. Output vectors are L2 normalized; cosine distance is therefore the native comparison metric.

## Evaluation

Every training run reports seen-class prototype accuracy and pair-verification AUROC on font identities excluded from all optimizer updates. Separately seeded crops from those held-out identities supply disjoint query/reference groups for calibration. Treat these as dataset-specific diagnostics, not population-level performance claims. Report the font collection commit, capture domain, configuration, and random seed with any model release.

## Risks and limitations

Font choice can correlate with language, geography, organization, document age, and software defaults. Those correlations do not establish tampering. Compression, reflow, rasterization engines, font hinting, bold/italic spans, and legitimate multi-font layouts may all change style distance. Synthetic calibration is not guaranteed to transfer to real capture pipelines.

## Human oversight

The output calls anomalous regions “review recommended” and includes the calibration p-value and caveat. A reviewer should compare equivalent layout roles, inspect document metadata and compression evidence, and seek an original source before forming a conclusion.
