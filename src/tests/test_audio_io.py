"""Тесты разбора бинарного PCM-протокола WebSocket-канала (app/audio_io.py)."""

import numpy as np
import pytest

from app.audio_io import FORMAT_FLOAT32, FORMAT_INT16, decode_pcm_message


def test_decode_float32_roundtrip():
    samples = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    raw = bytes([FORMAT_FLOAT32]) + samples.tobytes()

    decoded = decode_pcm_message(raw)

    assert decoded.dtype == np.float32
    np.testing.assert_allclose(decoded, samples)


def test_decode_int16_roundtrip_normalizes_to_float_range():
    ints = np.array([-32768, -16384, 0, 16384, 32767], dtype="<i2")
    raw = bytes([FORMAT_INT16]) + ints.tobytes()

    decoded = decode_pcm_message(raw)

    assert decoded.dtype == np.float32
    assert decoded.min() >= -1.0
    assert decoded.max() <= 1.0
    np.testing.assert_allclose(decoded, ints.astype(np.float32) / 32768.0)


def test_decode_rejects_unknown_format_tag():
    raw = bytes([255, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="Неизвестный тег"):
        decode_pcm_message(raw)


def test_decode_rejects_misaligned_float32_length():
    raw = bytes([FORMAT_FLOAT32]) + bytes([0, 1, 2])  # 3 байта - не кратно 4
    with pytest.raises(ValueError, match="не кратна 4"):
        decode_pcm_message(raw)


def test_decode_rejects_misaligned_int16_length():
    raw = bytes([FORMAT_INT16]) + bytes([0])  # 1 байт - не кратно 2
    with pytest.raises(ValueError, match="не кратна 2"):
        decode_pcm_message(raw)


def test_decode_rejects_too_short_message():
    with pytest.raises(ValueError, match="короткое"):
        decode_pcm_message(bytes([FORMAT_FLOAT32]))
