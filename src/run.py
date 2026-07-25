"""Единая точка запуска: поднимает FastAPI-бекенд (REST + WebSocket) и отдаёт
фронтенд по адресу http://127.0.0.1:8000/"""

import sys
import webbrowser
from pathlib import Path
from threading import Timer

# Добавляем пути для поиска модулей app/ и config/
_root = Path(__file__).resolve().parent.parent
if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(_root / "src"))
    sys.path.insert(0, str(_root))

import uvicorn  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402

HOST = "127.0.0.1"
PORT = 8000


def _open_browser() -> None:
    webbrowser.open(f"http://{HOST}:{PORT}/")


if __name__ == "__main__":
    Timer(1.2, _open_browser).start()
    print(f"Аудиомодем запущен: http://{HOST}:{PORT}/")
    uvicorn.run(fastapi_app, host=HOST, port=PORT, log_level="info")
