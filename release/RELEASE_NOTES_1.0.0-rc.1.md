# Comic Ebook Converter Public Edition 1.0.0-rc.1

First public Windows CPU release candidate.

## What it does

- Converts CBR/CBZ pages into panel-first CBZ files for smaller screens.
- Retains each original page after its generated panel views.
- Runs locally and does not upload comic pages.
- Includes a drag-and-drop queue and cancellation support.

## Public model policy

This release uses only the traceable `mosesb/best-comic-panel-detection` checkpoint documented in `MODELS.md`. Two private locally fine-tuned checkpoints are excluded because their training-image permissions were not documented. Difficult layouts may therefore differ from the private development build.

## Verification

The packaged CPU executable passed startup and conversion testing on an original AI-generated four-panel demo page. It produced four panel images plus the retained original page. This is a smoke test, not a comprehensive hardware or layout benchmark.

## Installation

Download the Universal CPU ZIP, extract it completely, and run `Comic Ebook Converter.exe`. Keep the `_internal` directory beside the executable. Windows may display a SmartScreen message because this community build is unsigned.

## License

The application and complete project are published under AGPL-3.0. The public checkpoint's Hugging Face model card is labelled Apache-2.0; its attribution, exact hashes and Apache license text are included.
