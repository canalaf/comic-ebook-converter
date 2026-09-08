# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project = Path(SPECPATH)
# Explicit files only: engine folders may contain private diagnostics after use.
release_data = [
    "assets/ComicEbookConverter_Logo.png",
    "assets/ComicEbookConverter.ico",
    "engine/run_v6_clean_pipeline.py",
    "engine/v55/marvel_kobo_v58_smart_merge.py",
    "engine/models/comic_base.pt",
    "LICENSE",
    "APACHE-2.0.txt",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "MODELS.md",
]
datas = [(str(project / name), str(Path(name).parent)) for name in release_data]
binaries = []
hiddenimports = [
    "cv2",
    "numpy",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "PIL.ImageOps",
    "statistics",
    "tkinterdnd2",
    "torch",
    "ultralytics",
]

for package in ("customtkinter", "tkinterdnd2", "ultralytics", "torch"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


a = Analysis(
    [str(project / "ComicEbookConverterApp.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Comic Ebook Converter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(project / "assets" / "ComicEbookConverter.ico")],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Comic Ebook Converter",
)
