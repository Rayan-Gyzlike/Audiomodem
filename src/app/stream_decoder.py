"""
Потоковый декодер: накапливает PCM-сэмплы и декодирует кадры.

Поддерживает два типа кадров:
  - Данные (текст/файл): преамбула 32 бита + RS + CRC32
  - Управление (ACK/NACK): укороченная преамбула 8 бит, 1 байт данных

При каждом push() возвращает (data_texts, control_signal).

Каждый этап декодирования (поиск преамбулы, RS-декодирование, CRC32)
обёрнут в try-except, чтобы повреждённый кадр не вызывал зависание
или необработанное исключение.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from . import modem
from config import CONTROL_BYTE_ACK, DEDUP_WINDOW_SECONDS, SAMPLE_RATE

logger = logging.getLogger("audio_modem")

MAX_BUFFER_SECONDS = 8
MAX_BUFFER_SAMPLES = MAX_BUFFER_SECONDS * SAMPLE_RATE
MAX_RETRIES_PER_PUSH = 25


class StreamDecoder:
    def __init__(self, on_diagnostics=None, diagnostics_every: int = 15) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._push_count = 0
        self._on_diagnostics = on_diagnostics
        self._diagnostics_every = max(1, diagnostics_every)
        self._recent_crcs: dict[int, float] = {}

    def push(self, chunk: np.ndarray) -> tuple[list[str], str | None]:
        """Добавляет чанк в буфер. Возвращает (data_texts, control_signal).

        Каждый этап декодирования защищён try-except:
        - Если поиск кадра падает → логируем и пропускаем
        - Если RS/CRC32 падает → логируем ошибку, пробуем дальше
        - Если весь push() падает → возвращаем пустой результат
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        self._buffer = np.concatenate([self._buffer, chunk])
        self._push_count += 1

        data_results: list[str] = []
        control_result: str | None = None
        retries = 0

        while True:
            # ── Шаг 1: пробуем декодировать данные ──
            try:
                text, consumed, crc32 = modem.signal_to_text(self._buffer)
            except modem.FrameNotFoundError:
                pass  # Данных в буфере нет — ищем контрольный кадр ниже
            except modem.DecodeError as exc:
                # Преамбула найдена, но данные повреждены (CRC32/RS)
                retries += 1
                logger.info("[RX Error] Кадр повреждён: %s (попытка %d/%d)",
                            exc, retries, MAX_RETRIES_PER_PUSH)
                if retries > MAX_RETRIES_PER_PUSH:
                    break
                # Отрезаем начало до найденного заголовка и пробуем дальше
                try:
                    start = modem.find_frame_start(self._buffer)
                    cut = (start or 0) + 1
                    self._buffer = self._buffer[cut:]
                except Exception:
                    # Если даже find_frame_start падает — обрезаем буфер
                    logger.warning("[RX Error] find_frame_start упал, обрезаем буфер")
                    self._buffer = self._buffer[len(self._buffer) // 2:]
                continue
            except Exception as exc:
                # Любая непредвиденная ошибка — логируем, не крашимся
                logger.warning("[RX Error] Непредвиденная ошибка декодирования: %s", exc)
                break
            else:
                # Успешно декодировали данные
                now = time.monotonic()
                self._recent_crcs = {
                    c: t for c, t in self._recent_crcs.items()
                    if now - t < DEDUP_WINDOW_SECONDS
                }
                if crc32 not in self._recent_crcs:
                    self._recent_crcs[crc32] = now
                    data_results.append(text)
                    control_result = "ack"
                    logger.info("CRC32 ОК → данные приняты")
                self._buffer = self._buffer[consumed:]
                continue

            # ── Шаг 2: данных нет — ищем контрольный кадр (ACK/NACK) ──
            if control_result is None:
                try:
                    ctrl = modem.signal_to_control(self._buffer)
                except Exception as exc:
                    logger.debug("signal_to_control ошибка: %s", exc)
                    ctrl = None
                if ctrl is not None:
                    control_byte, ctrl_consumed = ctrl
                    control_result = "ack" if control_byte == CONTROL_BYTE_ACK else "nack"
                    self._buffer = self._buffer[ctrl_consumed:]
                    continue

            # Ни данных, ни контрольного кадра
            break

        # Ограничиваем размер буфера
        if len(self._buffer) > MAX_BUFFER_SAMPLES:
            self._buffer = self._buffer[-MAX_BUFFER_SAMPLES:]

        # Периодическая диагностика
        if self._on_diagnostics and self._push_count % self._diagnostics_every == 0:
            try:
                diag = modem.sync_diagnostics(self._buffer)
                diag["chunk_len"] = len(chunk)
                diag["found_count"] = len(data_results)
                self._on_diagnostics(diag)
            except Exception:
                pass

        return data_results, control_result

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._push_count = 0
        self._recent_crcs.clear()
