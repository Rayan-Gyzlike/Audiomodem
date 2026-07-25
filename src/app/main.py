"""FastAPI-бекенд аудиомодема.

WebSocket-эндпоинт поддерживает:
  - Асинхронный декодинг через asyncio.to_thread (не блокирует event loop)
  - Ограниченную очередь аудио-чанков (дроп старых при переполнении)
  - Таймаут декодирования (2 сек) — сброс при зависании
  - Self-muting (600мс) — защита от эха
  - ARQ: ACK/NACK воспроизводятся в браузере через WebSocket (без sounddevice)
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scipy.io import wavfile

from . import modem
from .audio_io import FORMAT_NAMES, decode_pcm_message
from config import (
    AMPLITUDE,
    ACK_SEND_DELAY_MS,
    AUDIO_QUEUE_MAXSIZE,
    BLANKING_SECONDS,
    CONTROL_BYTE_ACK,
    CONTROL_BYTE_NACK,
    DECODE_TIMEOUT_SECONDS,
    FILE_CHUNK_RAW_MAX,
    FREQ_ONE,
    FREQ_ZERO,
    FREQUENCY_STEP_HZ,
    NACK_SILENCE_MS,
    NACK_SILENCE_SAMPLES,
    SAMPLE_RATE,
    SESSION_ID,
    TRANSMIT_REPEATS,
    TEST_PROFILES,
)
from .stream_decoder import StreamDecoder

CHUNK_LOG_EVERY = 50

# Директории для сохранения файлов
RECEIVED_FILES_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "received_files"
RECEIVED_CHUNKS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts" / "received_chunks"
RECEIVED_FILES_DIR.mkdir(parents=True, exist_ok=True)
RECEIVED_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("audio_modem")
if not logger.handlers:
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):
        pass
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [audio-modem] %(message)s", "%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


# Простой объект для хранения состояния передачи
class _TxState:
    is_transmitting: bool = False

state = _TxState()


app = FastAPI(title="Аудиомодем")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EncodeRequest(BaseModel):
    text: str


class TestProfileRequest(BaseModel):
    profile: str


class SessionRequest(BaseModel):
    session_id: int  # 0-3


def _signal_to_wav(signal: np.ndarray) -> bytes:
    pcm16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, pcm16)
    return buf.getvalue()


def _generate_wav_with_repeats(text: str) -> tuple[bytes, float]:
    """Генерирует WAV с повторами. Возвращает (wav_bytes, duration_seconds)."""
    signal = modem.text_to_signal(text)
    pcm16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    pause_samples = int(0.3 * SAMPLE_RATE)
    pause = np.zeros(pause_samples, dtype=np.int16)
    frames = []
    for i in range(TRANSMIT_REPEATS):
        frames.append(pcm16)
        if i < TRANSMIT_REPEATS - 1:
            frames.append(pause)
    combined = np.concatenate(frames)
    duration_sec = len(combined) / SAMPLE_RATE
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, combined)
    return buf.getvalue(), duration_sec


def _generate_ctrl_wav(control_byte: int) -> tuple[bytes, float]:
    """Генерирует WAV контрольного кадра с 100мс тишины в начале (padding).
    Возвращает (wav_bytes, duration_seconds)."""
    signal = modem.encode_control_frame(control_byte)
    silence = np.zeros(NACK_SILENCE_SAMPLES, dtype=np.float32)
    padded_signal = np.concatenate([silence, signal])
    duration_sec = len(padded_signal) / SAMPLE_RATE
    return _signal_to_wav(padded_signal), duration_sec


def _generate_control_pcm16_bytes(control_byte: int) -> tuple[bytes, float]:
    """Генерирует PCM16 (LE, mono) контрольного кадра с тишиной в начале.
    Возвращает (pcm_bytes, duration_seconds)."""
    signal = modem.encode_control_frame(control_byte)
    silence = np.zeros(NACK_SILENCE_SAMPLES, dtype=np.float32)
    padded = np.concatenate([silence, signal])
    duration_sec = len(padded) / SAMPLE_RATE
    pcm16 = np.clip(padded * 32767, -32768, 32767).astype(np.int16)
    return pcm16.tobytes(), duration_sec


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "sample_rate": SAMPLE_RATE,
        "freq_zero": FREQ_ZERO,
        "freq_one": FREQ_ONE,
        "transmit_repeats": TRANSMIT_REPEATS,
        "amplitude": AMPLITUDE,
    }


@app.get("/api/test-profiles")
def list_test_profiles() -> dict:
    """Список доступных профилей стресс-тестов."""
    return {"profiles": TEST_PROFILES}


@app.post("/api/test-profile")
def apply_test_profile(req: TestProfileRequest):
    """Применяет профиль стресс-теста (громкость, повторы и т.д.)."""
    import config as cfg
    profile = TEST_PROFILES.get(req.profile)
    if not profile:
        return Response(status_code=404, content=f"Профиль '{req.profile}' не найден")
    cfg.AMPLITUDE = profile["amplitude"]
    cfg.TRANSMIT_REPEATS = profile["transmit_repeats"]
    # Пересоздаём волновую форму синхрослова с новой амплитудой
    modem.rebuild_waveforms()
    logger.info(
        "Профиль '%s' применён: amplitude=%.2f, repeats=%d",
        req.profile, profile["amplitude"], profile["transmit_repeats"],
    )
    return {"status": "ok", "profile": req.profile, **profile}


@app.get("/api/session")
def get_session():
    """Текущая сессия и частотные параметры."""
    import config as cfg
    freq_zero, freq_one = modem.get_session_frequencies()
    return {
        "session_id": cfg.SESSION_ID,
        "freq_zero": freq_zero,
        "freq_one": freq_one,
        "description": f"Канал {cfg.SESSION_ID}: {freq_zero:.0f}/{freq_one:.0f} Гц",
    }


@app.post("/api/session")
def set_session(req: SessionRequest):
    """Устанавливает номер сессии (0-3) для мультиплеера."""
    import config as cfg
    if req.session_id < 0 or req.session_id > 3:
        return Response(status_code=400, content="session_id должен быть 0-3")
    cfg.SESSION_ID = req.session_id
    modem.rebuild_waveforms()
    freq_zero, freq_one = modem.get_session_frequencies()
    logger.info("Сессия установлена: %d (%.0f/%.0f Гц)", req.session_id, freq_zero, freq_one)
    return {
        "status": "ok",
        "session_id": req.session_id,
        "freq_zero": freq_zero,
        "freq_one": freq_one,
    }


@app.post("/api/encode")
def encode(req: EncodeRequest):
    text = req.text.strip()
    if not text:
        return Response(status_code=400, content="Текст пуст")
    try:
        wav_bytes, duration_sec = _generate_wav_with_repeats(text)
    except modem.MessageTooLongError as exc:
        logger.info("Отклонено слишком длинное сообщение (%d символов): %s", len(text), exc)
        return Response(status_code=400, content=str(exc))
    except modem.ModemError as exc:
        logger.warning("Ошибка кодирования текста %r: %s", text, exc)
        return Response(status_code=400, content=str(exc))
    except Exception:
        logger.exception("Непредвиденная ошибка при кодировании текста %r", text)
        return Response(status_code=500, content="Внутренняя ошибка кодирования сигнала.")
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Playback-Duration": f"{duration_sec:.3f}"},
    )


@app.post("/api/encode-file")
async def encode_file(file: UploadFile = File(...)):
    """Кодирует файл в WAV с фрагментацией.

    Файлы > FILE_CHUNK_RAW_MAX байт разбиваются на чанки:
    Формат чанка: "F:<filename>:<total>:<index>:<base64>"
    Все чанки кодируются в один WAV с паузами между ними.
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            return Response(status_code=400, content="Файл пуст")

        filename = file.filename or "unknown"
        total_size = len(file_bytes)

        # Маленький файл — отправляем одним кадром (старый формат)
        if total_size <= FILE_CHUNK_RAW_MAX:
            b64_data = base64.b64encode(file_bytes).decode("ascii")
            payload = f"FILE:{filename}:{total_size}:{b64_data}"
            wav_bytes, duration_sec = _generate_wav_with_repeats(payload)
            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers={"X-Playback-Duration": f"{duration_sec:.3f}"},
            )

        # Большой файл — фрагментация на чанки
        chunks = []
        for i in range(0, total_size, FILE_CHUNK_RAW_MAX):
            chunk_data = file_bytes[i:i + FILE_CHUNK_RAW_MAX]
            b64_chunk = base64.b64encode(chunk_data).decode("ascii")
            chunk_index = i // FILE_CHUNK_RAW_MAX
            total_chunks = (total_size + FILE_CHUNK_RAW_MAX - 1) // FILE_CHUNK_RAW_MAX
            # Формат: F:filename:total:index:base64
            payload = f"F:{filename}:{total_chunks}:{chunk_index}:{b64_chunk}"
            chunks.append(payload)

        logger.info("Файл '%s' (%d байт) → %d чанков", filename, total_size, len(chunks))

        # Кодируем все чанки в один WAV с паузами
        all_signals = []
        pause_samples = int(0.3 * SAMPLE_RATE)
        pause = np.zeros(pause_samples, dtype=np.float32)

        for idx, chunk_payload in enumerate(chunks):
            signal = modem.text_to_signal(chunk_payload)
            all_signals.append(signal)
            if idx < len(chunks) - 1:
                all_signals.append(pause)

        combined = np.concatenate(all_signals)
        wav_bytes = _signal_to_wav(combined)
        duration_sec = len(combined) / SAMPLE_RATE
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={"X-Playback-Duration": f"{duration_sec:.3f}"},
        )

    except modem.MessageTooLongError as exc:
        return Response(status_code=400, content=str(exc))
    except Exception:
        logger.exception("Ошибка кодирования файла %s", file.filename)
        return Response(status_code=500, content="Ошибка кодирования файла")


@app.post("/api/encode-ctrl")
def encode_ctrl(control: str):
    """Кодирует ACK или NACK в WAV (с 100мс padding тишины в начале)."""
    if control == "ack":
        wav_bytes, duration_sec = _generate_ctrl_wav(CONTROL_BYTE_ACK)
    elif control == "nack":
        wav_bytes, duration_sec = _generate_ctrl_wav(CONTROL_BYTE_NACK)
    else:
        return Response(status_code=400, content="control должен быть 'ack' или 'nack'")
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"X-Playback-Duration": f"{duration_sec:.3f}"},
    )


@app.websocket("/ws/decode")
async def ws_decode(websocket: WebSocket) -> None:
    """WebSocket: асинхронное декодирование + ARQ + self-muting.

    Архитектура:
    1. PCM-чанки попадают в ограниченную очередь (maxsize=30).
       При переполнении — дроп самых старых.
    2. Декодирование (FSK + RS + CRC32) выполняется через asyncio.to_thread
       в отдельном потоке, чтобы не блокировать event loop.
    3. Таймаут 2 сек на декодирование — при зависании сброс.
    4. Self-muting: при передаче блокируем приём на duration + 600мс.
    5. NACK: при CRC-ошибке отправляем NACK с задержкой 50мс.
    6. ACK/NACK воспроизводятся в БРАУЗЕРЕ через WebSocket (без sounddevice).
    """
    await websocket.accept()
    client = websocket.client
    logger.info("WS-подключение открыто: %s", client)

    def _log_diagnostics(diag: dict) -> None:
        score = diag["best_score"]
        score_str = f"{score:.3f}" if score is not None else "н/д"
        logger.info(
            "RX диагностика: буфер=%d сэмплов (%.2f с), RMS=%.5f, "
            "лучшая корреляция=%s (порог=%.2f)",
            diag["buffer_len"], diag["buffer_len"] / SAMPLE_RATE,
            diag.get("rms", 0.0), score_str, diag["threshold"],
        )

    decoder = StreamDecoder(on_diagnostics=_log_diagnostics)
    await websocket.send_json({"type": "status", "listening": True})

    chunk_count = 0
    session_start = time.monotonic()
    decode_count = 0
    self_mute_until: float = 0.0

    # Ограниченная очередь аудио-чанков
    audio_queue: deque[np.ndarray] = deque(maxlen=AUDIO_QUEUE_MAXSIZE)

    # Трекер сборки фрагментов файлов:
    # {filename: {"total": N, "chunks": {0: b64, 1: b64, ...}, "size": int}}
    file_assembly: dict[str, dict] = {}

    # Manual ACK: последний принятый файл для подтверждения вручную
    last_received_file: str | None = None

    # ARQ: последняя длительность воспроизведения (из X-Playback-Duration)
    last_tx_duration: float = 0.0

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            raw_bytes = message.get("bytes")
            raw_text = message.get("text")

            if raw_bytes is not None:
                # ── Парсинг PCM-фрейма ──
                try:
                    fmt_tag = raw_bytes[0] if raw_bytes else None
                    chunk = decode_pcm_message(raw_bytes)
                except ValueError as exc:
                    logger.warning("Некорректный PCM-фрейм: %s", exc)
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                chunk_count += 1

                # ── Self-muting: полный дроп во время передачи + эхо ──
                if time.monotonic() < self_mute_until:
                    continue  # Игнорируем ВСЕ байты — не передаём в декодер

                if len(chunk) == 0:
                    continue

                # Диагностика (раз в N чанков)
                rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
                peak = float(np.max(np.abs(chunk)))
                is_milestone = chunk_count == 1 or chunk_count % CHUNK_LOG_EVERY == 0
                if is_milestone:
                    logger.info(
                        "RX чанк #%d: формат=%s, %d сэмплов, RMS=%.5f, пик=%.5f",
                        chunk_count, FORMAT_NAMES.get(fmt_tag, "?"),
                        len(chunk), rms, peak,
                    )

                # ── Очередь: если декодер не успевает, дропаем старые ──
                audio_queue.append(chunk)
                if len(audio_queue) >= AUDIO_QUEUE_MAXSIZE:
                    dropped = audio_queue.popleft()
                    logger.warning("[Audio Queue] Переполнение! Дропаем %d сэмплов", len(dropped))

                # Микро-пауза чтобы не загружать CPU на 100%
                await asyncio.sleep(0.005)

                # ── Асинхронный декодинг с таймаутом ──
                # Собираем все накопленные сэмплы из очереди
                if audio_queue:
                    combined = np.concatenate(list(audio_queue))
                    audio_queue.clear()

                    try:
                        data_texts, control_signal = await asyncio.wait_for(
                            asyncio.to_thread(decoder.push, combined),
                            timeout=DECODE_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[RX Timeout] Декодирование превысило %.1f сек — сброс",
                            DECODE_TIMEOUT_SECONDS,
                        )
                        decoder.reset()
                        continue
                    except modem.ModemError as exc:
                        logger.warning("Ошибка декодера: %s", exc)
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        continue
                    except Exception as exc:
                        logger.warning("[RX Error] Непредвиденная ошибка: %s", exc)
                        continue

                    # ── Обработка результатов: данные ──
                    for text in data_texts:
                        decode_count += 1
                        elapsed = time.monotonic() - session_start
                        ts_now = time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"

                        # ═══ МГНОВЕННАЯ ОТПРАВКА ACK ═══════════════════════════════
                        # ACK отправляется ПЕРВЫМ ДЕЛОМ, до любой обработки файла,
                        # сохранения на диск, обновления UI — чтобы передатчик не ушёл
                        # на повторную отправку из-за задержки.

                        # Пауза 50мс — чтобы передатчик успел переключиться в режим приёма
                        await asyncio.sleep(ACK_SEND_DELAY_MS / 1000.0)

                        # Генерируем ACK PCM16 и отправляем в браузер для воспроизведения
                        ack_pcm_bytes, ack_duration = _generate_control_pcm16_bytes(CONTROL_BYTE_ACK)
                        ack_b64 = base64.b64encode(ack_pcm_bytes).decode("ascii")

                        # Self-muting на время воспроизведения ACK + blanking
                        self_mute_until = time.monotonic() + ack_duration + BLANKING_SECONDS

                        print(f"\n>>> [AUDIO OUT] ОТПРАВЛЯЮ ACK В БРАУЗЕР ДЛЯ ВОСПРОИЗВЕДЕНИЯ! <<<\n", flush=True)

                        # Отправляем PCM данные браузеру — браузер воспроизведёт ACK
                        await websocket.send_json({
                            "type": "control_signal",
                            "signal": "ACK",
                            "pcm": ack_b64,
                            "duration": ack_duration,
                        })

                        # Сообщаем фронтенду заблокировать микрофон на время воспроизведения
                        await websocket.send_json({
                            "type": "play_ctrl",
                            "control": "ack",
                            "duration": ack_duration,
                        })
                        logger.info(
                            "[RX SUCCESS] CRC32 ОК → ACK отправлен в браузер "
                            "(Time: %s, mute=%.2f сек)",
                            ts_now, ack_duration + BLANKING_SECONDS,
                        )

                        # ── Фрагментированный файл (новый формат "F:") ──
                        if text.startswith("F:") and not text.startswith("FILE:"):
                            parts = text.split(":", 4)
                            if len(parts) == 5:
                                _, fname, total_str, idx_str, b64_chunk = parts
                                total = int(total_str)
                                idx = int(idx_str)

                                if fname not in file_assembly:
                                    file_assembly[fname] = {"total": total, "chunks": {}, "size": 0}
                                file_assembly[fname]["chunks"][idx] = b64_chunk

                                # Сохраняем чанк для отладки
                                try:
                                    chunk_bytes = base64.b64decode(b64_chunk)
                                    chunk_path = RECEIVED_CHUNKS_DIR / f"{fname}_chunk{idx}.bin"
                                    chunk_path.write_bytes(chunk_bytes)
                                except Exception:
                                    pass

                                received_count = len(file_assembly[fname]["chunks"])
                                progress_pct = round(100 * received_count / total)
                                # Подсчитываем байты для прогресс-бара
                                bytes_received = sum(
                                    len(base64.b64decode(file_assembly[fname]["chunks"][i]))
                                    for i in file_assembly[fname]["chunks"]
                                )
                                # Оценка общего размера: средний размер чанка * число чанков
                                avg_chunk = bytes_received / received_count if received_count else 0
                                total_size = file_assembly[fname].get("size", 0)
                                if total_size == 0:
                                    total_size = int(avg_chunk * total)
                                logger.info("Файл '%s': чанк %d/%d принят (%d%%)",
                                            fname, received_count, total, progress_pct)

                                await websocket.send_json({
                                    "type": "file_progress",
                                    "filename": fname,
                                    "chunk_index": idx,
                                    "chunks_received": received_count,
                                    "chunks_total": total,
                                    "bytes_received": bytes_received,
                                    "total_size": total_size,
                                    "progress_pct": progress_pct,
                                })

                                # Все чанки получены → собираем файл
                                if received_count >= total:
                                    try:
                                        raw_chunks = []
                                        for i in range(total):
                                            raw_chunks.append(base64.b64decode(
                                                file_assembly[fname]["chunks"][i]
                                            ))
                                        assembled = b"".join(raw_chunks)
                                        logger.info("Файл '%s' собран: %d байт", fname, len(assembled))

                                        # Сохраняем файл на диск (с fallback-именем)
                                        save_name = fname if fname and fname != "unknown" else f"received_file_{int(time.time())}.bin"
                                        save_path = RECEIVED_FILES_DIR / save_name
                                        save_path.write_bytes(assembled)

                                        # Яркий лог в консоль
                                        print(f"\n!!! [FILE SAVED] Файл успешно сохранен: {save_path} ({len(assembled)} bytes) !!!\n", flush=True)
                                        logger.info("Файл '%s' сохранён: %s", save_name, save_path)

                                        last_received_file = save_name

                                        await websocket.send_json({
                                            "type": "received",
                                            "text": f"[Файл: {fname} ({len(assembled)} байт)]",
                                            "raw_text": None,
                                            "file_info": {
                                                "filename": fname,
                                                "original_size": len(assembled),
                                                "decoded_size": len(assembled),
                                                "chunks_total": total,
                                                "saved_to": str(save_path),
                                            },
                                            "crc_ok": True,
                                            "ts": time.time(),
                                            "metrics": {"decode_count": decode_count, "session_elapsed_sec": round(elapsed, 2)},
                                        })
                                    except Exception as exc:
                                        logger.warning("Ошибка сборки файла '%s': %s", fname, exc)
                                        await websocket.send_json({"type": "error", "message": f"Ошибка сборки файла: {exc}"})
                                    finally:
                                        del file_assembly[fname]
                                continue

                        # ── Цельный файл (старый формат "FILE:") ──
                        is_file = text.startswith("FILE:")
                        display_text = text
                        file_info = None
                        if is_file:
                            parts = text.split(":", 3)
                            if len(parts) == 4:
                                _, filename, size_str, b64 = parts
                                try:
                                    fb = base64.b64decode(b64)
                                    display_text = f"[Файл: {filename} ({len(fb)} байт)]"
                                    file_info = {"filename": filename, "original_size": int(size_str), "decoded_size": len(fb)}
                                except Exception:
                                    display_text = text

                        logger.info("Кадр декодирован: %r (Time: %s)", display_text, ts_now)
                        await websocket.send_json({
                            "type": "received",
                            "text": display_text,
                            "raw_text": text if is_file else None,
                            "file_info": file_info,
                            "crc_ok": True,
                            "ts": time.time(),
                            "metrics": {"decode_count": decode_count, "session_elapsed_sec": round(elapsed, 2)},
                        })

                    # ── Обработка результатов: контрольные кадры ──
                    if control_signal is not None:
                        if control_signal == "nack":
                            # NACK с задержкой 50мс для переключения передатчика
                            await asyncio.sleep(ACK_SEND_DELAY_MS / 1000.0)

                            # Генерируем NACK PCM16 и отправляем в браузер для воспроизведения
                            nack_pcm_bytes, nack_duration = _generate_control_pcm16_bytes(CONTROL_BYTE_NACK)
                            nack_b64 = base64.b64encode(nack_pcm_bytes).decode("ascii")

                            # Self-muting на время воспроизведения NACK + blanking
                            self_mute_until = time.monotonic() + nack_duration + BLANKING_SECONDS

                            print(f"\n>>> [AUDIO OUT] ОТПРАВЛЯЮ NACK В БРАУЗЕР ДЛЯ ВОСПРОИЗВЕДЕНИЯ! <<<\n", flush=True)

                            # Отправляем PCM данные браузеру — браузер воспроизведёт NACK
                            await websocket.send_json({
                                "type": "control_signal",
                                "signal": "NACK",
                                "pcm": nack_b64,
                                "duration": nack_duration,
                            })
                            # Сообщаем фронтенду заблокировать микрофон
                            await websocket.send_json({
                                "type": "play_ctrl",
                                "control": "nack",
                                "duration": nack_duration,
                            })
                            ts_nack = time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"
                            logger.info(
                                "[RX ERROR] Кадр повреждён → NACK отправлен в браузер "
                                "(Time: %s, mute=%.2f сек)",
                                ts_nack, nack_duration + BLANKING_SECONDS,
                            )
                        else:
                            ts_ack_rx = time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}"
                            await websocket.send_json({"type": "arq_status", "status": "delivered", "message": "Доставлено успешно (ACK)"})
                            logger.info("[RX ACK] Получен ACK → доставка подтверждена (Time: %s)", ts_ack_rx)

            # ── Текстовые команды от клиента ──
            elif raw_text is not None:
                try:
                    payload = json.loads(raw_text)
                except json.JSONDecodeError:
                    continue

                if payload.get("type") == "reset":
                    decoder.reset()
                    audio_queue.clear()
                    file_assembly.clear()
                    decode_count = 0
                    session_start = time.monotonic()
                    self_mute_until = 0.0

                elif payload.get("type") == "hello":
                    logger.info("Клиент: sampleRate=%s, AudioContext=%s",
                                payload.get("sampleRate"), payload.get("clientContextRate"))

                elif payload.get("type") == "tx_complete":
                    # Self-muting: блокируем приём на duration + 600мс
                    # duration передаётся фронтендом из X-Playback-Duration
                    duration = payload.get("duration", 0.0)
                    if duration > 0:
                        last_tx_duration = duration
                    # Используем переданную длительность или последнюю известную
                    effective_duration = duration if duration > 0 else last_tx_duration
                    self_mute_until = time.monotonic() + effective_duration + BLANKING_SECONDS
                    logger.info(
                        "[Audio Mute] Микрофон заблокирован на %.2f сек "
                        "(playback=%.2f + blanking=%.1f)",
                        effective_duration + BLANKING_SECONDS,
                        effective_duration, BLANKING_SECONDS,
                    )

                elif payload.get("type") == "manual_ack":
                    # Ручное подтверждение приёма: сбрасываем ARQ-таймаут
                    # и помечаем последний принятый файл как успешно доставленный
                    logger.info("[MANUAL ACK] Получено ручное подтверждение от клиента")
                    await websocket.send_json({
                        "type": "arq_status",
                        "status": "delivered",
                        "message": "Доставлено (подтверждено вручную)",
                    })

    except WebSocketDisconnect:
        pass
    finally:
        logger.info("WS-подключение закрыто: %s", client)


def _frontend_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "frontend"
    return Path(__file__).resolve().parent.parent / "frontend"


_FRONTEND_DIR = _frontend_dir()
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
