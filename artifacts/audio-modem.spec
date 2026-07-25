# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller-спека для сборки автономного audio-modem.exe.

Сборка:
    backend\venv\Scripts\pyinstaller.exe audio-modem.spec --noconfirm

Итог: dist\audio-modem.exe (однофайловый, самодостаточный).
"""

from pathlib import Path

block_cipher = None

ROOT = Path(SPECPATH)
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

# uvicorn[standard] и starlette подгружают часть реализаций (event loop,
# протоколы HTTP/WebSocket, lifespan) динамически через importlib внутри
# функций "auto"-выбора, поэтому статический анализ PyInstaller их не видит
# и их нужно перечислить явно — иначе сервер падает в рантайме уже после
# упаковки в exe с ModuleNotFoundError.
HIDDEN_IMPORTS = [
    # uvicorn internals (auto-detected dynamically via importlib)
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    # WebSocket support
    "websockets",
    "websockets.legacy",
    "websockets.legacy.server",
    # Модем: новые модули бэкенда
    "app.config",
    "app.modem",
    "app.audio_io",
    "app.stream_decoder",
    # Зависимости бэкенда (импортируются внутри функций)
    "reedsolo",
    "python_multipart",
    "multipart",
    "scipy.io",
    "scipy.io.wavfile",
    "scipy.signal",
    "numpy",
    "numpy._core",
    "numpy._core.multiarray",
    "numpy._core.numerictypes",
    "numpy.core",
    "numpy.core.numeric",
    "numpy.core.function_base",
]

a = Analysis(
    ["run_app.py"],
    pathex=[str(BACKEND_DIR)],
    binaries=[],
    datas=[
        (str(FRONTEND_DIR), "frontend"),
    ],
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="audio-modem",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
