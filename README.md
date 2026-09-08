# Comic Ebook Converter

Comic Ebook Converter turns CBR and CBZ comic books into panel-first CBZ files that are easier to read on small e-readers. For every source page, it generates readable panel crops and keeps the original page at the end.

The application runs locally. Comic pages are not uploaded to a server.

## Highlights

- Automatic comic-panel detection
- One traceable public comic-panel model used across the detection pipeline
- Reading-order resolution
- Speech-text-aware crop expansion
- Duplicate and low-value crop suppression
- Original page retained after its generated panels
- Drag-and-drop queue
- Conversion cancellation
- Universal CPU release for Windows

## Download

The first public release is prepared as a GitHub **Release** asset:

- **Universal CPU Edition:** works without an NVIDIA GPU; slower on long books

Download the ZIP for your edition, extract the entire archive, and run `Comic Ebook Converter.exe`. Do not separate the executable from its `_internal` directory.

## Requirements

- Windows 10 or Windows 11, 64-bit
- No Python installation for prebuilt packages
- No internet connection during conversion

CBZ files are supported directly. CBR extraction tries Windows `tar`, then installed 7-Zip or WinRAR. If extraction fails, install a compatible archive tool or supply CBZ input.

## Using the application

1. Start `Comic Ebook Converter.exe`.
2. Drag CBR or CBZ files into the conversion queue, or use the file picker.
3. Select **Start Conversion**.
4. Find the converted CBZ files in the `Comics` directory beside the application.

CPU processing can take time on long books. Keep the application open until the current conversion finishes.

## Running from source

Python 3.12 is recommended. Source execution requires the separate model pack; it is not included in this source archive.

1. Download the model pack from the Releases page.
2. Extract it into the repository root so the paths in [MODELS.md](MODELS.md) exist.
3. From the repository directory, install the CPU environment and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe .\ComicEbookConverterApp.py
```

If your chosen environment is already active, run:

```powershell
python .\ComicEbookConverterApp.py
```

The conversion engine also has a command-line entry point:

```powershell
python .\engine\run_v6_clean_pipeline.py input.cbz output.cbz
```

## Building Windows packages

After installing the model pack, use the CPU build script:

```powershell
.\scripts\build_cpu.ps1
```

Build output is written below `dist` and is excluded from Git.

## Privacy and content

- Conversion is local and offline.
- No comic books or training images are included in this repository.
- The two locally fine-tuned private checkpoints are not included in this edition.
- Users are responsible for processing only content they are legally allowed to use.
- Comic Ebook Converter is not affiliated with comic publishers, e-reader manufacturers, or their trademarks.

## License

This project is distributed under the [GNU Affero General Public License v3.0](LICENSE) because it uses Ultralytics YOLO under its AGPL-3.0 option. Review [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution or commercial use.

The included public checkpoint comes from [`mosesb/best-comic-panel-detection`](https://huggingface.co/mosesb/best-comic-panel-detection); its published SHA-256 matches the preserved upstream file. Its model card carries an Apache-2.0 label, and that license text is included as `APACHE-2.0.txt`. See [MODELS.md](MODELS.md) and [Ultralytics licensing](https://www.ultralytics.com/license).

## Status

The current Public Edition is a Windows CPU release candidate. It deliberately excludes the two private fine-tuned models, so difficult or unusual layouts may differ from the private three-model build. Reports should include the input format, page number, expected panel boundary, and a legally shareable screenshot when possible. Do not upload copyrighted comic books to issue reports.

## Release verification status

On 2026-09-08, the one-model Public Edition produced four panel views plus the original page from an original AI-generated demo. Three crops and the retained original matched the private reference pixel-for-pixel; one panel crop differed because the private fine-tuned detectors are intentionally absent. This is a one-page smoke check, not a broad layout or hardware compatibility benchmark.

The Public Edition uses Python 3.12 and the pinned CPU dependency set. Its source, traceable model, packaging specification and license notices are published with the application release.

See [PUBLIC_EDITION.md](PUBLIC_EDITION.md) for the public/private model separation.
