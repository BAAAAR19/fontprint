# Data card: synthetic typography crops

## Source

Samples are generated at access time from font files already present on the user's system or explicitly passed with `--font-root`. Font binaries and rendered datasets are not committed by this project.

## Generation

Each sample combines a neutral document-domain word with a font face and seeded rendering parameters. Augmentations include small rotation, blur, grayscale sensor noise, contrast variation, positional jitter, font-size variation, and occasional JPEG recompression.

## Splits

- Training and seen validation share font identities but use disjoint sample seeds.
- Open-set holdout reserves entire font identities before optimization. Separate deterministic seeds are used for evaluation and conformal calibration crops.
- `PKBatchSampler` selects P identities and K samples per identity.

## Licensing

Users must verify the licenses of supplied fonts. A reproducible public experiment should use a pinned collection with redistribution/training permissions and publish a manifest containing font name, source, license, checksum, and split. Google Fonts is a convenient source of OFL and Apache-licensed faces, but each file's license should still be recorded.

## Privacy

The built-in generator uses fictional text only and contains no personal data. Real documents introduced by users may contain highly sensitive information; the local CLI and demo do not transmit them. API operators are responsible for transport security, retention, access control, and logging policy.

## Known gaps

Synthetic words do not capture every rasterizer, printer, scanner, camera, paper texture, language, or document layout. Current vocabulary is English and current font filtering requires a small Latin coverage probe.
