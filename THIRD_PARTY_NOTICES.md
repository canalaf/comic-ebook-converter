# Third-party notices

Comic Ebook Converter depends on third-party open-source software. Each component remains subject to its own license and copyright notices.

Important components include:

- [Ultralytics](https://github.com/ultralytics/ultralytics), version 8.4.121
- [PyTorch](https://github.com/pytorch/pytorch), version 2.13.0
- [TorchVision](https://github.com/pytorch/vision), version 0.28.0
- [OpenCV](https://github.com/opencv/opencv)
- [Pillow](https://github.com/python-pillow/Pillow)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [TkinterDnD2](https://github.com/pmgagne/tkinterdnd2)
- [PyInstaller](https://github.com/pyinstaller/pyinstaller)

## Ultralytics

This project uses Ultralytics YOLO code and model weights under the AGPL-3.0 open-source option. The complete AGPL-3.0 text is provided in `LICENSE`.

Ultralytics publishes separate terms for proprietary and commercial use. Review the [current licensing information](https://www.ultralytics.com/license) directly with Ultralytics for your distribution model.

## Content notice

The source repository and release packages must not contain comic-book archives, extracted copyrighted pages, annotation screenshots, or training datasets. Model and dataset provenance should be reviewed separately before a broad public or commercial release.


## Redistribution status

These component names are a human-readable inventory. The release archive also retains the license and copyright files found in the installed distributions, including native-library notices. The public model's repository, commit and hashes are documented in `MODELS.md`. This notice records the identified terms and is not a substitute for legal advice.

## Public Edition model

The Public Edition includes `best.pt` from [`mosesb/best-comic-panel-detection`](https://huggingface.co/mosesb/best-comic-panel-detection), whose model card is labelled Apache-2.0. The upstream file SHA-256 is documented in `MODELS.md`, and the Apache-2.0 text is included as `APACHE-2.0.txt`. Two locally fine-tuned private checkpoints are excluded from the Public Edition.
