# Public model

The Public Edition uses one checkpoint in all detector roles:

`engine/models/comic_base.pt`

## Provenance

- Repository: [`mosesb/best-comic-panel-detection`](https://huggingface.co/mosesb/best-comic-panel-detection)
- Upstream file: `best.pt`
- Upstream commit used: `bbab11504194d0b341ac3f6099f3592aa0604ae3`
- Upstream SHA-256: `566e9ff59ec146afb695eb48a8aaae983b43bb72c3a57f63afafe12f3b349af4`
- Upstream model-card license label: Apache-2.0
- Cleaned distribution SHA-256: `4bc841f9ea33f449f97550ed5b420aa83b9997faf5d87ad6716b3e1b77001b92`

The distribution copy differs only in non-functional path metadata. Its 1,249 serialized tensor entries match the upstream checkpoint byte-for-byte, and inference parity was verified locally.

The repository's model card says it is a YOLOv12x model trained with Ultralytics on a custom Roboflow dataset. Because this application uses the Ultralytics software stack, the complete Public Edition is released under AGPL-3.0. The Apache-2.0 text and attribution are retained alongside the project license.

The two locally fine-tuned private checkpoints are deliberately excluded. Their comic-page training permissions were not documented, and the Public Edition does not require them.
