"""
Ядро аудиомодема: кодирование текста в FSK-сигнал и обратное декодирование.

Формат кадра (в битах, каждый бит = SAMPLES_PER_SYMBOL сэмплов тона):

    [ PREAMBLE (32 бита, чередование 0/1) ]
    [ SYNC_WORD (16 бит, уникальный маркер) ]
    [ LEN (8 бит, длина закодированного RS-сообщения в байтах) ]
    [ DATA (LEN * 8 бит, полезная нагрузка после кодирования Рида-Соломона) ]

Кодирование Рида-Соломона (reedsolo) добавляет RS_NSYM байт избыточности,
что позволяет декодеру восстановить сообщение даже при наличии шума/помех
в канале (акустическая передача через динамик/микрофон).

Синхронизация приёмника с началом кадра выполняется через согласованный
фильтр (matched filter): волновая форма преамбулы+синхрослова коррелируется
со всем принятым буфером, и максимум корреляции даёт точную позицию начала
кадра с точностью до сэмпла.
"""

from __future__ import annotations

import binascii

import numpy as np
from scipy.signal import correlate as scipy_correlate
from reedsolo import RSCodec, ReedSolomonError

from config import (
    AMPLITUDE,
    BAUD_RATE,
    CONTROL_BYTE_ACK,
    CONTROL_BYTE_NACK,
    CONTROL_PREAMBLE_BITS,
    CRC_SIZE_BYTES,
    FREQ_ONE,
    FREQ_ZERO,
    FREQUENCY_STEP_HZ,
    MAX_ENCODED_LEN,
    PREAMBLE_BITS,
    RS_NSYM,
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
    SESSION_ID,
    SYNC_WORD_BYTES,
)


def get_session_frequencies(session_id: int | None = None) -> tuple[float, float]:
    """Возвращает (freq_zero, freq_one) для указанной сессии.

    Каждая сессия сдвигает базовые частоты на session_id * FREQUENCY_STEP_HZ,
    обеспечивая изоляцию разных пар устройств.
    """
    sid = session_id if session_id is not None else SESSION_ID
    offset = sid * FREQUENCY_STEP_HZ
    return FREQ_ZERO + offset, FREQ_ONE + offset


class ModemError(Exception):
    """Базовая ошибка модема."""


class MessageTooLongError(ModemError):
    """Сообщение слишком длинное для передачи одним кадром."""


class InvalidTextError(ModemError):
    """Текст нельзя закодировать в UTF-8 (например, содержит одиночный,
    непарный суррогат — такая строка технически проходит через JSON, но не
    является валидным Unicode-текстом)."""


class FrameNotFoundError(ModemError):
    """В буфере не найден валидный преамбула+синхрослово кадра."""


class DecodeError(ModemError):
    """Кадр найден, но Reed-Solomon не смог восстановить данные."""


_rsc = RSCodec(RS_NSYM)


def rebuild_waveforms() -> None:
    """Пересчитывает глобальные волновые формы после изменения AMPLITUDE.

    Вызывается при смене профиля стресс-теста. Пересоздаёт:
    - _HEADER_WAVEFORM, _SYNC_WAVEFORM, _SYNC_ENERGY
    - _CONTROL_HEADER_WAVEFORM, _CONTROL_SYNC_WAVEFORM, _CONTROL_SYNC_ENERGY
    """
    global _HEADER_WAVEFORM, _SYNC_WAVEFORM, _SYNC_ENERGY
    global _CONTROL_HEADER_WAVEFORM, _CONTROL_SYNC_WAVEFORM, _CONTROL_SYNC_ENERGY

    _HEADER_WAVEFORM = _modulate_bits(_HEADER_BITS)
    _SYNC_WAVEFORM = _HEADER_WAVEFORM[_PREAMBLE_SAMPLES:]
    _SYNC_ENERGY = float(np.sum(_SYNC_WAVEFORM.astype(np.float64) ** 2))

    _CONTROL_HEADER_WAVEFORM = _modulate_bits(CONTROL_PREAMBLE_BITS + SYNC_WORD_BITS)
    _CONTROL_SYNC_WAVEFORM = _CONTROL_HEADER_WAVEFORM[len(CONTROL_PREAMBLE_BITS) * SAMPLES_PER_SYMBOL:]
    _CONTROL_SYNC_ENERGY = float(np.sum(_CONTROL_SYNC_WAVEFORM.astype(np.float64) ** 2))


def _bytes_to_bits(data: bytes) -> list[int]:
    bits: list[int] = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_to_bytes(bits: list[int]) -> bytes:
    if len(bits) % 8 != 0:
        raise ValueError("Число бит должно быть кратно 8")
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i : i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


SYNC_WORD_BITS = _bytes_to_bits(SYNC_WORD_BYTES)
_HEADER_BITS = PREAMBLE_BITS + SYNC_WORD_BITS  # используется как матч.фильтр


def _modulate_bits(bits: list[int], freq_zero: float | None = None,
                   freq_one: float | None = None) -> np.ndarray:
    """Модулирует последовательность бит в аудиосэмплы с непрерывной фазой
    (чтобы избежать щелчков на границах символов).

    Если freq_zero/freq_one не указаны, используются частоты текущей сессии.
    """
    f0 = freq_zero if freq_zero is not None else get_session_frequencies()[0]
    f1 = freq_one if freq_one is not None else get_session_frequencies()[1]
    n = len(bits)
    total_samples = n * SAMPLES_PER_SYMBOL
    signal = np.empty(total_samples, dtype=np.float32)

    t_symbol = np.arange(SAMPLES_PER_SYMBOL) / SAMPLE_RATE
    phase = 0.0
    for i, bit in enumerate(bits):
        freq = f1 if bit else f0
        chunk = AMPLITUDE * np.sin(2 * np.pi * freq * t_symbol + phase)
        start = i * SAMPLES_PER_SYMBOL
        signal[start : start + SAMPLES_PER_SYMBOL] = chunk
        # непрерывная фаза для следующего символа
        phase = (phase + 2 * np.pi * freq * SAMPLES_PER_SYMBOL / SAMPLE_RATE) % (
            2 * np.pi
        )
    return signal


_HEADER_WAVEFORM = _modulate_bits(_HEADER_BITS)

# Преамбула чередует биты 0/1 с периодом ровно в 2 символа, поэтому если
# использовать для согласованного фильтра ВЕСЬ заголовок (преамбула+синхро),
# корреляция даёт ложные пики, отстоящие от истинного на чётное число
# символов (сдвинутая на 2 символа преамбула почти неотличима от исходной).
# Чтобы устранить эту неоднозначность, для поиска границы кадра используется
# только сегмент синхрослова — непериодичный и однозначно идентифицируемый.
# Он берётся как хвост _HEADER_WAVEFORM, чтобы сохранить ту же непрерывную
# фазу, с которой синхрослово реально передаётся вслед за преамбулой.
_PREAMBLE_SAMPLES = len(PREAMBLE_BITS) * SAMPLES_PER_SYMBOL
_SYNC_WAVEFORM = _HEADER_WAVEFORM[_PREAMBLE_SAMPLES:]
_SYNC_ENERGY = float(np.sum(_SYNC_WAVEFORM.astype(np.float64) ** 2))

# Порог по НОРМИРОВАННОЙ кросс-корреляции (аналог косинусного сходства,
# диапазон примерно [-1, 1]), а не по абсолютной энергии. Абсолютный порог
# был откалиброван под тестовые сигналы полной амплитуды и на практике
# проваливался на реальном сигнале с микрофона: после прохождения через
# динамик -> воздух -> микрофон уровень сигнала обычно в разы (иногда на
# порядок) тише, чем у синтетического WAV, и абсолютная корреляция падала
# ниже порога, хотя форма сигнала (и, соответственно, косинусное сходство)
# оставалась почти идеальной. Нормировка на энергию окна делает порог
# нечувствительным к общему уровню громкости/усиления микрофона.
_SYNC_MATCH_THRESHOLD = 0.82
# Игнорируем окна почти полной тишины — иначе деление корреляции на очень
# маленькую энергию окна может дать шумовой всплеск нормированного значения.
_MIN_WINDOW_ENERGY = 1e-6


def _tone_basis(freq: float) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (cos, sin) базисные векторы для однобиновой ДПФ (эквивалент
    алгоритма Гёрцеля), используемые для вычисления мощности сигнала на
    частоте freq в окне длиной SAMPLES_PER_SYMBOL."""
    n = SAMPLES_PER_SYMBOL
    k = int(0.5 + n * freq / SAMPLE_RATE)
    t = np.arange(n)
    w = 2 * np.pi * k * t / n
    return np.cos(w), np.sin(w)


_COS_ZERO, _SIN_ZERO = _tone_basis(FREQ_ZERO)
_COS_ONE, _SIN_ONE = _tone_basis(FREQ_ONE)


def _demodulate_bits(signal: np.ndarray, start: int, num_bits: int) -> list[int]:
    """Векторизованная демодуляция num_bits бит начиная с сэмпла start.

    Для каждого символьного окна вычисляется мощность на FREQ_ZERO и
    FREQ_ONE через однобиновую ДПФ (матричное умножение вместо Python-цикла
    Гёрцеля), после чего выбирается частота с большей мощностью.
    """
    end = start + num_bits * SAMPLES_PER_SYMBOL
    if end > len(signal):
        raise FrameNotFoundError("Буфер закончился раньше, чем ожидалось")

    windows = signal[start:end].reshape(num_bits, SAMPLES_PER_SYMBOL)
    power_zero = (windows @ _COS_ZERO) ** 2 + (windows @ _SIN_ZERO) ** 2
    power_one = (windows @ _COS_ONE) ** 2 + (windows @ _SIN_ONE) ** 2
    return (power_one > power_zero).astype(int).tolist()


def encode_text(text: str) -> bytes:
    """Кодирует текст в защищённые Reed-Solomon байты.

    Порядок: текст -> UTF-8 -> CRC32 (добавляется в конец) -> Reed-Solomon.
    На приёме после RS-декодирования CRC32 проверяется для подтверждения
    целостности данных.
    """
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InvalidTextError(f"Текст содержит символы, не кодируемые в UTF-8: {exc}") from exc

    # Вычисляем CRC32 от исходных данных и дописываем его в конец
    crc = binascii.crc32(raw) & 0xFFFFFFFF
    crc_bytes = crc.to_bytes(CRC_SIZE_BYTES, byteorder="big")
    payload_with_crc = raw + crc_bytes

    encoded = bytes(_rsc.encode(payload_with_crc))
    if len(encoded) > MAX_ENCODED_LEN:
        raise MessageTooLongError(
            f"Сообщение слишком длинное: {len(encoded)} байт "
            f"(максимум {MAX_ENCODED_LEN})"
        )
    return encoded


def text_to_signal(text: str) -> np.ndarray:
    """Полный конвейер: текст -> FSK-аудиосигнал (float32, диапазон [-1, 1])."""
    encoded = encode_text(text)
    len_bits = _bytes_to_bits(bytes([len(encoded)]))
    data_bits = _bytes_to_bits(encoded)
    all_bits = PREAMBLE_BITS + SYNC_WORD_BITS + len_bits + data_bits
    return _modulate_bits(all_bits)


def _normalized_sync_correlation(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Считает кросс-корреляцию сигнала с волновой формой синхрослова,
    нормированную по энергии окна сигнала (устойчиво к громкости/усилению
    входного сигнала). Возвращает (normalized, window_energy)."""
    signal64 = np.asarray(signal, dtype=np.float64)
    correlation = scipy_correlate(signal64, _SYNC_WAVEFORM, mode="valid", method="fft")

    window = len(_SYNC_WAVEFORM)
    cumsum = np.concatenate(([0.0], np.cumsum(signal64 * signal64)))
    window_energy = cumsum[window:] - cumsum[:-window]

    denom = np.sqrt(np.maximum(window_energy, 0.0) * _SYNC_ENERGY) + 1e-12
    normalized = correlation / denom
    return normalized, window_energy


def find_frame_start(signal: np.ndarray) -> int | None:
    """Находит начало кадра (конец преамбулы+синхрослова) через согласованный
    фильтр по синхрослову с нормировкой по амплитуде. Возвращает индекс
    сэмпла, с которого начинается байт LEN, либо None, если кадр не найден."""
    if len(signal) < len(_SYNC_WAVEFORM):
        return None

    normalized, window_energy = _normalized_sync_correlation(signal)

    candidates = np.flatnonzero(
        (normalized >= _SYNC_MATCH_THRESHOLD) & (window_energy >= _MIN_WINDOW_ENERGY)
    )
    if len(candidates) == 0:
        return None

    # Берём САМЫЙ РАННИЙ кадр в буфере (а не глобальный максимум по всему
    # буферу), т.к. в потоковом режиме перед декодером может быть очередь из
    # нескольких сообщений и их нужно извлекать по порядку. Индексы выше
    # порога группируются в кластеры; берём первый по времени кластер и
    # находим точный пик внутри него.
    #
    # Соседние индексы объединяются в один кластер, если разрыв между ними
    # меньше длины символа: сам пик согласованного фильтра — не идеальная
    # ступенька, а "купол" с небольшими проседаниями ниже порога у вершины,
    # из-за которых один и тот же истинный пик может расщепиться на
    # несколько формально разделённых мини-кластеров, отстоящих друг от
    # друга всего на десятки сэмплов. Наивная разбивка по gap>1 в таком
    # случае хватает самый ранний (и заведомо более слабый) фрагмент купола
    # вместо истинного максимума в паре десятков сэмплов правее. Настоящие
    # разные сообщения в потоке разделены как минимум паузой в десятки-сотни
    # миллисекунд — что значительно больше одного символа (441 сэмпл), так
    # что объединение с таким порогом не склеивает разные передачи.
    gaps = np.flatnonzero(np.diff(candidates) > SAMPLES_PER_SYMBOL)
    first_cluster_end = int(gaps[0]) if len(gaps) else len(candidates) - 1
    cluster = candidates[: first_cluster_end + 1]
    peak_idx = int(cluster[np.argmax(normalized[cluster])])

    return peak_idx + len(_SYNC_WAVEFORM)


def sync_diagnostics(signal: np.ndarray) -> dict:
    """Диагностика поиска синхрослова для логирования: даже если валидный
    кадр не найден, показывает, насколько близко буфер подошёл к порогу
    обнаружения (полезно, чтобы отличить "нет сигнала вообще" от "сигнал
    есть, но чуть-чуть не дотягивает до порога")."""
    signal = np.asarray(signal, dtype=np.float32)
    rms = float(np.sqrt(np.mean(signal ** 2))) if len(signal) else 0.0

    if len(signal) < len(_SYNC_WAVEFORM):
        return {
            "buffer_len": len(signal),
            "rms": rms,
            "best_score": None,
            "threshold": _SYNC_MATCH_THRESHOLD,
        }

    normalized, _ = _normalized_sync_correlation(signal)
    best_idx = int(np.argmax(normalized))
    return {
        "buffer_len": len(signal),
        "rms": rms,
        "best_score": float(normalized[best_idx]),
        "threshold": _SYNC_MATCH_THRESHOLD,
    }


def signal_to_text(signal: np.ndarray) -> tuple[str, int, int]:
    """Декодирует текст из аудиосигнала.

    Возвращает (текст, число_сэмплов_использованных_из_буфера, crc32).
    Бросает FrameNotFoundError, если кадр не найден, или DecodeError, если
    Reed-Solomon не смог восстановить данные или CRC32 не совпал.
    """
    signal = np.asarray(signal, dtype=np.float32)
    data_start = find_frame_start(signal)
    if data_start is None:
        raise FrameNotFoundError("Преамбула/синхрослово не найдены в буфере")

    len_bits = _demodulate_bits(signal, data_start, 8)
    encoded_len = _bits_to_bytes(len_bits)[0]

    payload_start = data_start + 8 * SAMPLES_PER_SYMBOL
    payload_bits = _demodulate_bits(signal, payload_start, encoded_len * 8)
    payload = _bits_to_bytes(payload_bits)

    try:
        decoded = _rsc.decode(payload)
    except ReedSolomonError as exc:
        raise DecodeError(f"Reed-Solomon не смог восстановить сообщение: {exc}") from exc

    decoded_bytes = decoded[0] if isinstance(decoded, tuple) else decoded
    decoded_raw = bytes(decoded_bytes)

    # Разделяем данные и CRC32: последние CRC_SIZE_BYTES байт — это хэш
    if len(decoded_raw) < CRC_SIZE_BYTES:
        raise DecodeError("Слишком короткое декодированное сообщение (нет места для CRC32)")

    data_part = decoded_raw[:-CRC_SIZE_BYTES]
    received_crc = int.from_bytes(decoded_raw[-CRC_SIZE_BYTES:], byteorder="big")

    # Проверяем целостность: вычисленный CRC32 должен совпасть с полученным
    expected_crc = binascii.crc32(data_part) & 0xFFFFFFFF
    if received_crc != expected_crc:
        raise DecodeError(
            f"CRC32 не совпал: получен 0x{received_crc:08X}, "
            f"ожидался 0x{expected_crc:08X}"
        )

    try:
        text = data_part.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DecodeError("Восстановленные байты не являются валидным UTF-8") from exc

    consumed = payload_start + encoded_len * 8 * SAMPLES_PER_SYMBOL
    return text, consumed, received_crc


def text_to_wav_bytes(text: str) -> bytes:
    """Возвращает WAV-файл (16-бит PCM) в виде bytes для отправки на фронтенд."""
    import io
    from scipy.io import wavfile

    signal = text_to_signal(text)
    pcm16 = np.clip(signal * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, pcm16)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# Контрольные кадры ARQ (ACK / NACK)
# ═══════════════════════════════════════════════════════════════════════════════
# Формат: [ CONTROL_PREAMBLE (8 бит) ][ SYNC_WORD (16 бит) ][ CONTROL_BYTE (8 бит) ]
# Итого 32 бита ≈ 0.32 сек. Без RS-кодирования для минимальной задержки.
#
# Важно: sync word в контрольном кадре имеет другую начальную фазу, чем в
# данных (короткая преамбула → другая фаза на границе), поэтому для
# обнаружения используется отдельный matched filter с правильной фазой.

# Волновая форма синхрослова для контрольных кадров (фаза от короткой преамбулы)
_CONTROL_HEADER_WAVEFORM = _modulate_bits(CONTROL_PREAMBLE_BITS + SYNC_WORD_BITS)
_CONTROL_SYNC_WAVEFORM = _CONTROL_HEADER_WAVEFORM[len(CONTROL_PREAMBLE_BITS) * SAMPLES_PER_SYMBOL:]
_CONTROL_SYNC_ENERGY = float(np.sum(_CONTROL_SYNC_WAVEFORM.astype(np.float64) ** 2))


def encode_control_frame(control_byte: int) -> np.ndarray:
    """Кодирует контрольный кадр (ACK или NACK) в аудиосигнал.

    Амплитуда контрольных кадров повышена на 25% относительно AMPLITUDE
    для лучшей устойчивости к шуму и уверенного приёма на передатчике.
    """
    bits = CONTROL_PREAMBLE_BITS + SYNC_WORD_BITS + _bytes_to_bits(bytes([control_byte]))
    signal = _modulate_bits(bits)
    # Усиление на 25% — контрольные кадры должны быть громче данных,
    # чтобы передатчик гарантированно распознал ACK/NACK даже в шуме.
    boosted = np.clip(signal * 1.25, -1.0, 1.0)
    return boosted


def encode_ack() -> np.ndarray:
    """Кодирует ACK-кадр (подтверждение успешного приёма)."""
    return encode_control_frame(CONTROL_BYTE_ACK)


def encode_nack() -> np.ndarray:
    """Кодирует NACK-кадр (запрос повторной передачи)."""
    return encode_control_frame(CONTROL_BYTE_NACK)


def signal_to_control(signal: np.ndarray) -> tuple[int, int] | None:
    """Пытается декодировать контрольный кадр (ACK/NACK) из аудиосигнала.

    Возвращает (control_byte, consumed_samples) или None, если кадр не найден.
    Использует отдельный согласованный фильтр с фазой короткой преамбулы.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) < len(_CONTROL_SYNC_WAVEFORM) + 1 * SAMPLES_PER_SYMBOL:
        return None

    # Кросс-корреляция с sync waveform контрольного кадра (нормированная)
    signal64 = np.asarray(signal, dtype=np.float64)
    correlation = scipy_correlate(signal64, _CONTROL_SYNC_WAVEFORM, mode="valid", method="fft")

    window = len(_CONTROL_SYNC_WAVEFORM)
    cumsum = np.concatenate(([0.0], np.cumsum(signal64 * signal64)))
    window_energy = cumsum[window:] - cumsum[:-window]

    denom = np.sqrt(np.maximum(window_energy, 0.0) * _CONTROL_SYNC_ENERGY) + 1e-12
    normalized = correlation / denom

    candidates = np.flatnonzero(
        (normalized >= _SYNC_MATCH_THRESHOLD) & (window_energy >= _MIN_WINDOW_ENERGY)
    )
    if len(candidates) == 0:
        return None

    gaps = np.flatnonzero(np.diff(candidates) > SAMPLES_PER_SYMBOL)
    first_cluster_end = int(gaps[0]) if len(gaps) else len(candidates) - 1
    cluster = candidates[: first_cluster_end + 1]
    peak_idx = int(cluster[np.argmax(normalized[cluster])])

    data_start = peak_idx + len(_CONTROL_SYNC_WAVEFORM)
    control_end = data_start + 8 * SAMPLES_PER_SYMBOL
    if control_end > len(signal):
        return None

    byte_bits = _demodulate_bits(signal, data_start, 8)
    control_byte = _bits_to_bytes(byte_bits)[0]

    if control_byte not in (CONTROL_BYTE_ACK, CONTROL_BYTE_NACK):
        return None

    return control_byte, control_end
