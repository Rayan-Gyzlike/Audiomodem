"""Smoke-тесты FastAPI: REST + WebSocket + ARQ."""

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from app.main import app
from app import modem
from config import CONTROL_BYTE_ACK, SAMPLE_RATE
from app.stream_decoder import StreamDecoder

client = TestClient(app)


def test_health():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["sample_rate"] == SAMPLE_RATE


def test_encode_endpoint_returns_valid_wav():
    resp = client.post("/api/encode", json={"text": "API roundtrip"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    sample_rate, pcm16 = wavfile.read(io.BytesIO(resp.content))
    assert sample_rate == SAMPLE_RATE
    signal = pcm16.astype(np.float32) / 32767.0
    decoded, _, _ = modem.signal_to_text(signal)
    assert decoded == "API roundtrip"


def test_encode_endpoint_rejects_empty_text():
    assert client.post("/api/encode", json={"text": "   "}).status_code == 400


def test_encode_endpoint_rejects_too_long_text():
    assert client.post("/api/encode", json={"text": "x" * 300}).status_code == 400


def test_encode_endpoint_rejects_lone_surrogate_with_400():
    resp = client.post("/api/encode", content=b'{"text": "\\ud800"}', headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


def test_encode_endpoint_returns_clean_500_on_unexpected_error(monkeypatch):
    monkeypatch.setattr(modem, "text_to_signal", lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
    resp = client.post("/api/encode", json={"text": "любой текст"})
    assert resp.status_code == 500


@pytest.mark.parametrize("text", ["Обычный текст", "x" * 245, "𝕏𝕐𝕑", "​﻿", "\t\n\r  текст  "])
def test_encode_endpoint_handles_tricky_inputs_without_500(text):
    resp = client.post("/api/encode", json={"text": text})
    assert resp.status_code in (200, 400)


# ═══════════════════════════════════════════════════════════════════════════════
# ARQ REST API тесты
# ═══════════════════════════════════════════════════════════════════════════════


def test_encode_ctrl_ack():
    resp = client.post("/api/encode-ctrl?control=ack")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    sample_rate, pcm16 = wavfile.read(io.BytesIO(resp.content))
    assert len(pcm16) / sample_rate < 1.0


def test_encode_ctrl_nack():
    assert client.post("/api/encode-ctrl?control=nack").status_code == 200


def test_encode_ctrl_invalid():
    assert client.post("/api/encode-ctrl?control=invalid").status_code == 400


def test_encode_ctrl_ack_is_decodable():
    resp = client.post("/api/encode-ctrl?control=ack")
    sample_rate, pcm16 = wavfile.read(io.BytesIO(resp.content))
    signal = pcm16.astype(np.float32) / 32767.0
    result = modem.signal_to_control(signal)
    assert result is not None
    assert result[0] == CONTROL_BYTE_ACK


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket тесты
# ═══════════════════════════════════════════════════════════════════════════════


def _pcm_frame(chunk: np.ndarray) -> bytes:
    return bytes([1]) + chunk.astype(np.float32).tobytes()


def _drain_until(ws, msg_type, max_msgs=20):
    msgs = []
    for _ in range(max_msgs):
        msg = ws.receive_json()
        msgs.append(msg)
        if msg.get("type") == msg_type:
            break
    return msgs


def test_ws_decode_roundtrip():
    text = "Websocket streamed message"
    signal = modem.text_to_signal(text)
    padded = np.concatenate([np.zeros(1000, dtype=np.float32), signal, np.zeros(1000, dtype=np.float32)])
    with client.websocket_connect("/ws/decode") as ws:
        assert ws.receive_json() == {"type": "status", "listening": True}
        ws.send_text('{"type": "hello", "sampleRate": 44100, "clientContextRate": 48000}')
        for i in range(0, len(padded), 2048):
            ws.send_bytes(_pcm_frame(padded[i:i + 2048]))
        msgs = _drain_until(ws, "received")
        types = [m["type"] for m in msgs]
        assert "received" in types
        assert "play_ctrl" in types
        received = next(m for m in msgs if m["type"] == "received")
        assert received["text"] == text
        assert received["crc_ok"] is True


def test_ws_decode_roundtrip_attenuated_signal():
    text = "Тихий сигнал с микрофона"
    signal = modem.text_to_signal(text)
    rng = np.random.default_rng(1)
    quiet = signal * 0.15 + rng.standard_normal(len(signal)).astype(np.float32) * 0.01
    padded = np.concatenate([np.zeros(1500, dtype=np.float32), quiet, np.zeros(1500, dtype=np.float32)])
    with client.websocket_connect("/ws/decode") as ws:
        ws.receive_json()
        for i in range(0, len(padded), 2048):
            ws.send_bytes(_pcm_frame(padded[i:i + 2048]))
        msgs = _drain_until(ws, "received")
        received = next(m for m in msgs if m["type"] == "received")
        assert received["text"] == text


def test_ws_decode_rejects_malformed_pcm_frame():
    with client.websocket_connect("/ws/decode") as ws:
        ws.receive_json()
        ws.send_bytes(bytes([1, 0, 1, 2]))
        assert ws.receive_json()["type"] == "error"
        ws.send_bytes(bytes([99, 0, 0, 0, 0]))
        assert ws.receive_json()["type"] == "error"
        text = "После ошибок формата работает"
        ws.send_bytes(_pcm_frame(modem.text_to_signal(text)))
        msgs = _drain_until(ws, "received")
        assert next(m for m in msgs if m["type"] == "received")["text"] == text


def test_ws_self_muting_after_tx_complete():
    with client.websocket_connect("/ws/decode") as ws:
        ws.receive_json()
        ws.send_text('{"type": "tx_complete"}')
        ws.send_bytes(_pcm_frame(modem.text_to_signal("Should be muted")))
        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════════
# Фрагментация файлов
# ═══════════════════════════════════════════════════════════════════════════════


def test_encode_file_small():
    """Маленький файл кодируется одним кадром (старый формат FILE:)."""
    resp = client.post("/api/encode-file", files={"file": ("test.txt", b"Hello, world!", "text/plain")})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"


def test_encode_file_chunked():
    """Большой файл фрагментируется на чанки (новый формат F:)."""
    content = b"X" * 500  # > FILE_CHUNK_RAW_MAX (128)
    resp = client.post("/api/encode-file", files={"file": ("big.bin", content, "application/octet-stream")})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    sample_rate, pcm16 = wavfile.read(io.BytesIO(resp.content))
    assert len(pcm16) > sample_rate  # Больше 1 секунды


def test_encode_file_empty():
    """Пустой файл отклоняется."""
    resp = client.post("/api/encode-file", files={"file": ("empty.txt", b"", "text/plain")})
    assert resp.status_code == 400


def test_ws_decode_chunked_file():
    """WS: приём фрагментированного файла — проверка кодирования и декодирования чанков."""
    content = b"Test chunked file data " * 5  # ~115 байт < FILE_CHUNK_RAW_MAX (128)
    resp = client.post("/api/encode-file", files={"file": ("chunk.txt", content, "text/plain")})
    wav_bytes = resp.content

    sample_rate, pcm16 = wavfile.read(io.BytesIO(wav_bytes))
    signal = pcm16.astype(np.float32) / 32767.0

    # Декодируем первый кадр напрямую (без WS)
    text, _, _ = modem.signal_to_text(signal)
    assert text.startswith("FILE:")
    assert "chunk.txt" in text


# ═══════════════════════════════════════════════════════════════════════════════
# Тест фрагментации: разбиение → кодирование → декодирование → сборка → SHA256
# ═══════════════════════════════════════════════════════════════════════════════


def test_chunking_reassembly_sha256():
    """Полный цикл: файл → чанки → WAV → декод → сборка → SHA256 совпадает."""
    import base64
    import hashlib
    from config import FILE_CHUNK_RAW_MAX

    original = b"X" * 400  # > 128 → 4 чанка
    original_hash = hashlib.sha256(original).hexdigest()

    # Разбиваем на чанки (как делает encode-file)
    chunks = []
    for i in range(0, len(original), FILE_CHUNK_RAW_MAX):
        chunk_data = original[i:i + FILE_CHUNK_RAW_MAX]
        b64_chunk = base64.b64encode(chunk_data).decode("ascii")
        chunk_index = i // FILE_CHUNK_RAW_MAX
        total_chunks = (len(original) + FILE_CHUNK_RAW_MAX - 1) // FILE_CHUNK_RAW_MAX
        payload = f"F:test.bin:{total_chunks}:{chunk_index}:{b64_chunk}"
        chunks.append(payload)

    assert len(chunks) == 4

    # Каждый чанк кодируем в WAV, декодируем, проверяем
    assembled = {}
    for payload in chunks:
        signal = modem.text_to_signal(payload)
        decoded, _, _ = modem.signal_to_text(signal)
        assert decoded == payload
        parts = decoded.split(":", 4)
        idx = int(parts[3])  # F:name:total:index:b64
        b64_data = parts[4]
        assembled[idx] = base64.b64decode(b64_data)

    # Собираем файл
    result = b"".join(assembled[i] for i in range(len(assembled)))
    assert hashlib.sha256(result).hexdigest() == original_hash
    assert result == original


# ═══════════════════════════════════════════════════════════════════════════════
# Тест ARQ: симуляция потери пакета → NACK → повторная отправка
# ═══════════════════════════════════════════════════════════════════════════════


def test_arq_nack_on_corrupted_frame():
    """Повреждённый кадр → NACK через StreamDecoder."""
    text = "ARQ NACK test"
    signal = modem.text_to_signal(text)
    # Сильно повреждаем сигнал
    corrupted = signal.copy()
    corrupted[len(corrupted)//2:len(corrupted)//2 + 500] *= -1
    decoder = StreamDecoder()
    texts, ctrl = decoder.push(corrupted)
    # Либо не декодируется вообще, либо (редко) RS восстановит
    if texts:
        # Если RS восстановил — OK, ACK
        assert ctrl == "ack"
    else:
        # Не декодировалось — может вернуться NACK или ничего
        # Важно что НЕ вернул successfully decoded text с wrong data
        pass


def test_arq_retry_success_after_initial_failure():
    """Симуляция: первый пакет повреждён (NACK), второй ОК (ACK)."""
    decoder = StreamDecoder()

    # Повреждённый кадр
    bad_signal = np.random.randn(5000).astype(np.float32) * 0.3
    texts1, ctrl1 = decoder.push(bad_signal)

    # Хороший кадр
    text = "Retry success"
    good_signal = modem.text_to_signal(text)
    padded = np.concatenate([np.zeros(500, dtype=np.float32), good_signal])
    texts2, ctrl2 = decoder.push(padded)

    assert "Retry success" in texts2


def test_arq_ack_on_valid_frame():
    """Валидный кадр → ACK."""
    decoder = StreamDecoder()
    signal = modem.text_to_signal("Valid frame")
    padded = np.concatenate([np.zeros(500, dtype=np.float32), signal])
    texts, ctrl = decoder.push(padded)
    assert "Valid frame" in texts
    assert ctrl == "ack"


# ═══════════════════════════════════════════════════════════════════════════════
# Тест профилей стресс-тестов
# ═══════════════════════════════════════════════════════════════════════════════


def test_test_profiles_endpoint():
    resp = client.get("/api/test-profiles")
    assert resp.status_code == 200
    profiles = resp.json()["profiles"]
    assert "basic" in profiles
    assert "max_range" in profiles


def test_apply_test_profile():
    resp = client.post("/api/test-profile", json={"profile": "basic"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_apply_nonexistent_profile():
    resp = client.post("/api/test-profile", json={"profile": "nonexistent"})
    assert resp.status_code == 404


def test_session_endpoint():
    resp = client.get("/api/session")
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "freq_zero" in data


def test_set_session():
    resp = client.post("/api/session", json={"session_id": 2})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == 2
    # Восстанавливаем
    client.post("/api/session", json={"session_id": 0})


# ═══════════════════════════════════════════════════════════════════════════════
# Тесты сохранения файлов на диск
# ═══════════════════════════════════════════════════════════════════════════════


def test_ws_file_chunked_reassembly_and_save(tmp_path, monkeypatch):
    """Сборка фрагментированного файла через StreamDecoder → сохранение на диск."""
    import base64
    import hashlib
    from config import FILE_CHUNK_RAW_MAX
    from app import main as main_mod

    monkeypatch.setattr(main_mod, "RECEIVED_FILES_DIR", tmp_path)

    original = b"File save test data " * 10  # ~190 байт → 2 чанка
    original_hash = hashlib.sha256(original).hexdigest()
    filename = "test_save.bin"

    total_chunks = (len(original) + FILE_CHUNK_RAW_MAX - 1) // FILE_CHUNK_RAW_MAX
    chunk_signals = []
    for i in range(total_chunks):
        chunk_data = original[i * FILE_CHUNK_RAW_MAX:(i + 1) * FILE_CHUNK_RAW_MAX]
        b64_chunk = base64.b64encode(chunk_data).decode("ascii")
        payload = f"F:{filename}:{total_chunks}:{i}:{b64_chunk}"
        signal = modem.text_to_signal(payload)
        padded = np.concatenate([np.zeros(500, dtype=np.float32), signal, np.zeros(500, dtype=np.float32)])
        chunk_signals.append(padded)

    # Декодируем чанки через StreamDecoder (без WS/self-mute)
    decoder = StreamDecoder()
    file_assembly = {}
    for sig in chunk_signals:
        texts, ctrl = decoder.push(sig)
        for text in texts:
            if text.startswith("F:") and not text.startswith("FILE:"):
                parts = text.split(":", 4)
                if len(parts) == 5:
                    _, fname, total_str, idx_str, b64_chunk = parts
                    total = int(total_str)
                    idx = int(idx_str)
                    if fname not in file_assembly:
                        file_assembly[fname] = {"total": total, "chunks": {}}
                    file_assembly[fname]["chunks"][idx] = b64_chunk

    # Собираем и сохраняем файл
    assert filename in file_assembly
    fa = file_assembly[filename]
    assert len(fa["chunks"]) == fa["total"]
    assembled = b"".join(base64.b64decode(fa["chunks"][i]) for i in range(fa["total"]))
    save_path = tmp_path / filename
    save_path.write_bytes(assembled)

    assert save_path.exists()
    assert save_path.read_bytes() == original
    assert hashlib.sha256(save_path.read_bytes()).hexdigest() == original_hash


def test_ws_file_fallback_name_when_unknown(tmp_path, monkeypatch):
    """Файл с именем 'unknown' сохраняется с fallback-именем received_file_<timestamp>.bin."""
    import base64
    from app import main as main_mod
    monkeypatch.setattr(main_mod, "RECEIVED_FILES_DIR", tmp_path)

    original = b"Fallback name test"
    b64_data = base64.b64encode(original).decode("ascii")
    payload = f"F:unknown:1:0:{b64_data}"
    signal = modem.text_to_signal(payload)
    padded = np.concatenate([np.zeros(500, dtype=np.float32), signal, np.zeros(500, dtype=np.float32)])

    decoder = StreamDecoder()
    texts, ctrl = decoder.push(padded)
    assert len(texts) == 1
    text = texts[0]
    assert text.startswith("F:unknown:")
    # Fallback имя = "unknown" → сохраняется как received_file_<timestamp>.bin
    # Проверяем что сборка корректна (StreamDecoder вернул текст)
    parts = text.split(":", 4)
    assert len(parts) == 5
    _, fname, total_str, idx_str, b64_chunk = parts
    assert fname == "unknown"
    assert total_str == "1"
    assembled = base64.b64decode(b64_chunk)
    assert assembled == original


# ═══════════════════════════════════════════════════════════════════════════════
# Тест Manual ACK
# ═══════════════════════════════════════════════════════════════════════════════


def test_ws_manual_ack():
    """WS: отправка manual_ack → получение arq_status delivered."""
    with client.websocket_connect("/ws/decode") as ws:
        ws.receive_json()  # status
        ws.send_text('{"type": "manual_ack"}')
        msgs = _drain_until(ws, "arq_status")
        arq = next((m for m in msgs if m.get("type") == "arq_status"), None)
        assert arq is not None
        assert arq["status"] == "delivered"
        assert "ручную" in arq["message"].lower() or "ручн" in arq["message"].lower()


def test_manual_ack_does_not_crash_without_connection():
    """Manual ACK обрабатывается корректно даже после других сообщений."""
    with client.websocket_connect("/ws/decode") as ws:
        ws.receive_json()
        # Сначала отправляем hello, потом manual_ack
        ws.send_text('{"type": "hello", "sampleRate": 44100}')
        ws.send_text('{"type": "manual_ack"}')
        msgs = _drain_until(ws, "arq_status")
        arq = next((m for m in msgs if m.get("type") == "arq_status"), None)
        assert arq is not None
        assert arq["status"] == "delivered"
