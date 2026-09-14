# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for `Socratic AI.app` (onedir, macOS arm64).

Build with packaging/build_app.sh, not by hand: the script injects the version,
signs the bundle and zips it. Entry point is app_native.py (native window).
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).parent
VERSION = os.environ.get("SOCRATIC_VERSION", "0.0.0")
WITH_LOCAL = os.environ.get("SOCRATIC_BUNDLE_LOCAL", "1") == "1"  # parakeet-mlx / mlx-whisper

datas = []
binaries = []
hiddenimports = [
    # uvicorn picks these by string at runtime
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "wsproto",
    "engineio.async_drivers.asgi",
    "webview.platforms.cocoa",
    "sqlalchemy.dialects.sqlite",
    "pydantic.deprecated.decorator",
    "multipart",
    "python_multipart",
]

for pkg in ("nicegui", "chromadb", "certifi", "yt_dlp", "pymupdf", "fitz"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass
hiddenimports += collect_submodules("chromadb")
hiddenimports += collect_submodules("openai")
hiddenimports += collect_submodules("nicegui")
hiddenimports += collect_submodules("socketio")
hiddenimports += collect_submodules("engineio")

if WITH_LOCAL:
    for pkg in ("mlx", "mlx_whisper", "parakeet_mlx", "librosa", "huggingface_hub", "tokenizers", "numba", "llvmlite"):
        try:
            d, b, h = collect_all(pkg)
            datas += d
            binaries += b
            hiddenimports += h
        except Exception:
            pass

# Repo-level modules PyInstaller can't see through the FastAPI app object.
hiddenimports += collect_submodules("app") + collect_submodules("routes") + collect_submodules("frontend") + collect_submodules("prompts")
hiddenimports += ["main", "config", "run"]

# chromadb imports its OpenTelemetry exporter (and therefore grpc) at import
# time, so those cannot be excluded. onnxruntime can: we always supply vectors.
excludes = [
    "onnxruntime",
    "kubernetes",
    "tkinter", "matplotlib", "IPython", "pytest", "torch", "torchaudio", "tensorflow",
]

a = Analysis(
    [str(ROOT / "app_native.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas + [(str(ROOT / "packaging" / "icon.png"), "packaging")],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Socratic AI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    target_arch="arm64",
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Socratic AI")

app = BUNDLE(
    coll,
    name="Socratic AI.app",
    icon=str(ROOT / "packaging" / "icon.icns"),
    bundle_identifier="com.arturgrochau.socratic-ai",
    info_plist={
        "CFBundleName": "Socratic AI",
        "CFBundleDisplayName": "Socratic AI",
        "CFBundleShortVersionString": VERSION,
        "CFBundleVersion": VERSION,
        "LSMinimumSystemVersion": "14.0",
        "LSApplicationCategoryType": "public.app-category.education",
        "LSArchitecturePriority": ["arm64"],
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
        "NSAppTransportSecurity": {"NSAllowsLocalNetworking": True},
        "NSHumanReadableCopyright": "MIT License. Copyright 2026 Artur Grochau.",
    },
)
