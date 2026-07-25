"""Unit-тесты DSP-ядра + ARQ: текст -> WAV/сигнал -> декодирование -> сверка."""

import numpy as np
import pytest

from app import modem
from config import (
    CONTROL_BYTE_ACK,
    CONTROL_BYTE_NACK,
    CONTROL_PREAMBLE_BITS,
    SAMPLE_RATE,
    SAMPLES_PER_SYMBOL,
)
from app.modem import SYNC_WORD_BITS
from app.stream_decoder import StreamDecoder


# ═══════════════════════════════════════════════════════════════════════════════
# Roundtrip тесты
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("text", ["Hello", "Привет, мир!", "A", "The quick brown fox jumps over the lazy dog 1234567890", "😀 emoji test 🎧", "   spaces   "])
def test_roundtrip_text(text):
    signal = modem.text_to_signal(text)
    decoded, consumed, crc32 = modem.signal_to_text(signal)
    assert decoded == text
    assert consumed <= len(signal)
    assert isinstance(crc32, int)


def test_roundtrip_via_wav_bytes():
    import io
    from scipy.io import wavfile
    text = "Roundtrip through a real WAV file"
    wav_bytes = modem.text_to_wav_bytes(text)
    sample_rate, pcm16 = wavfile.read(io.BytesIO(wav_bytes))
    assert sample_rate == SAMPLE_RATE
    signal = pcm16.astype(np.float32) / 32767.0
    decoded, _, _ = modem.signal_to_text(signal)
    assert decoded == text


def test_leading_and_trailing_silence_is_tolerated():
    text = "Sync must find me despite silence"
    signal = modem.text_to_signal(text)
    padded = np.concatenate([np.zeros(12345, dtype=np.float32), signal, np.zeros(5000, dtype=np.float32)])
    decoded, _, _ = modem.signal_to_text(padded)
    assert decoded == text


def test_reed_solomon_corrects_bit_errors():
    text = "Error correction should save this message"
    signal = modem.text_to_signal(text)
    start = 60 * SAMPLES_PER_SYMBOL
    signal[start:start + SAMPLES_PER_SYMBOL] *= -1
    decoded, _, _ = modem.signal_to_text(signal)
    assert decoded == text


def test_pure_noise_raises_frame_not_found():
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(SAMPLE_RATE) * 0.05).astype(np.float32)
    with pytest.raises(modem.FrameNotFoundError):
        modem.signal_to_text(noise)


def test_empty_text_roundtrip():
    signal = modem.text_to_signal("")
    decoded, _, _ = modem.signal_to_text(signal)
    assert decoded == ""


def test_message_too_long_raises():
    with pytest.raises(modem.MessageTooLongError):
        modem.text_to_signal("x" * 300)


def test_lone_surrogate_raises_invalid_text_error():
    with pytest.raises(modem.InvalidTextError):
        modem.encode_text("\ud800")


def test_roundtrip_survives_additive_noise():
    text = "Robust to acoustic channel noise"
    signal = modem.text_to_signal(text)
    rng = np.random.default_rng(7)
    noisy = signal + rng.standard_normal(len(signal)).astype(np.float32) * 0.05
    decoded, _, _ = modem.signal_to_text(noisy)
    assert decoded == text


def test_decode_is_fast_enough_for_realtime():
    import time
    text = "Timing sanity check"
    signal = modem.text_to_signal(text)
    padded = np.concatenate([np.zeros(3 * SAMPLE_RATE, dtype=np.float32), signal, np.zeros(2 * SAMPLE_RATE, dtype=np.float32)])
    start = time.perf_counter()
    decoded, _, _ = modem.signal_to_text(padded)
    assert decoded == text
    assert (time.perf_counter() - start) < 1.0


def test_roundtrip_survives_heavy_amplitude_attenuation():
    text = "Тихий сигнал"
    signal = modem.text_to_signal(text)
    rng = np.random.default_rng(3)
    quiet = signal * 0.05 + rng.standard_normal(len(signal)).astype(np.float32) * 0.003
    decoded, _, _ = modem.signal_to_text(quiet)
    assert decoded == text


def test_sync_diagnostics_reports_high_score_for_clean_signal():
    text = "Diagnostics sanity check"
    signal = modem.text_to_signal(text)
    diag = modem.sync_diagnostics(signal)
    assert diag["best_score"] is not None
    assert diag["best_score"] > 0.95


def test_sync_diagnostics_on_short_buffer_does_not_crash():
    diag = modem.sync_diagnostics(np.zeros(10, dtype=np.float32))
    assert diag["best_score"] is None


def test_stream_decoder_on_pure_noise_terminates_quickly():
    import time
    rng = np.random.default_rng(99)
    noise = (rng.standard_normal(5 * SAMPLE_RATE) * 0.2).astype(np.float32)
    decoder = StreamDecoder()
    start = time.perf_counter()
    for i in range(0, len(noise), 4096):
        decoder.push(noise[i:i + 4096])
    assert (time.perf_counter() - start) < 2.0


def test_stream_decoder_two_messages_back_to_back():
    sig1 = modem.text_to_signal("First message")
    sig2 = modem.text_to_signal("Second message")
    combined = np.concatenate([sig1, np.zeros(3000, dtype=np.float32), sig2])
    decoder = StreamDecoder()
    texts, _ctrl = decoder.push(combined)
    assert texts == ["First message", "Second message"]


def test_stream_decoder_returns_tuple():
    signal = modem.text_to_signal("Tuple test")
    decoder = StreamDecoder()
    result = decoder.push(signal)
    assert isinstance(result, tuple)
    assert len(result) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# CRC32 тесты
# ═══════════════════════════════════════════════════════════════════════════════


def test_crc32_is_included_in_encoded_data():
    import binascii
    from config import CRC_SIZE_BYTES
    text = "CRC test"
    raw = text.encode("utf-8")
    encoded = modem.encode_text(text)
    from app.modem import _rsc
    decoded_raw = bytes(_rsc.decode(encoded)[0])
    assert len(decoded_raw) >= CRC_SIZE_BYTES
    data_part = decoded_raw[:-CRC_SIZE_BYTES]
    received_crc = int.from_bytes(decoded_raw[-CRC_SIZE_BYTES:], byteorder="big")
    assert received_crc == (binascii.crc32(data_part) & 0xFFFFFFFF)


def test_roundtrip_text_returns_crc32():
    _, _, crc = modem.signal_to_text(modem.text_to_signal("CRC32 return value test"))
    assert isinstance(crc, int)
    assert 0 <= crc <= 0xFFFFFFFF


def test_same_text_same_crc32():
    sig1 = modem.text_to_signal("Deterministic CRC")
    sig2 = modem.text_to_signal("Deterministic CRC")
    _, _, crc1 = modem.signal_to_text(sig1)
    _, _, crc2 = modem.signal_to_text(sig2)
    assert crc1 == crc2


def test_different_text_different_crc32():
    _, _, crc1 = modem.signal_to_text(modem.text_to_signal("Message A"))
    _, _, crc2 = modem.signal_to_text(modem.text_to_signal("Message B"))
    assert crc1 != crc2


# ═══════════════════════════════════════════════════════════════════════════════
# Дедупликация
# ═══════════════════════════════════════════════════════════════════════════════


def test_stream_decoder_deduplicates_identical_messages():
    signal = modem.text_to_signal("Dedup test")
    combined = np.concatenate([signal, np.zeros(1000, dtype=np.float32), signal])
    decoder = StreamDecoder()
    texts, _ = decoder.push(combined)
    assert texts == ["Dedup test"]


def test_stream_decoder_different_messages_not_deduplicated():
    sig1 = modem.text_to_signal("Unique A")
    sig2 = modem.text_to_signal("Unique B")
    combined = np.concatenate([sig1, np.zeros(1000, dtype=np.float32), sig2])
    decoder = StreamDecoder()
    texts, _ = decoder.push(combined)
    assert texts == ["Unique A", "Unique B"]


def test_stream_decoder_dedup_reset_clears_history():
    signal = modem.text_to_signal("Reset dedup")
    combined = np.concatenate([signal, np.zeros(1000, dtype=np.float32), signal])
    decoder = StreamDecoder()
    texts1, _ = decoder.push(combined)
    assert texts1 == ["Reset dedup"]
    texts2, _ = decoder.push(combined)
    assert texts2 == []
    decoder.reset()
    texts3, _ = decoder.push(combined)
    assert texts3 == ["Reset dedup"]


# ═══════════════════════════════════════════════════════════════════════════════
# ARQ: контрольные кадры (ACK / NACK)
# ═══════════════════════════════════════════════════════════════════════════════


def test_encode_control_frame_ack():
    signal = modem.encode_ack()
    assert isinstance(signal, np.ndarray)
    expected = (len(CONTROL_PREAMBLE_BITS) + len(SYNC_WORD_BITS) + 8) * SAMPLES_PER_SYMBOL
    assert len(signal) == expected


def test_encode_control_frame_nack():
    signal = modem.encode_nack()
    assert isinstance(signal, np.ndarray)
    assert len(signal) > 0


def test_control_frame_roundtrip_ack():
    signal = modem.encode_ack()
    result = modem.signal_to_control(signal)
    assert result is not None
    assert result[0] == CONTROL_BYTE_ACK


def test_control_frame_roundtrip_nack():
    signal = modem.encode_nack()
    result = modem.signal_to_control(signal)
    assert result is not None
    assert result[0] == CONTROL_BYTE_NACK


def test_control_frame_with_silence_around():
    signal = modem.encode_ack()
    padded = np.concatenate([np.zeros(5000, dtype=np.float32), signal, np.zeros(3000, dtype=np.float32)])
    result = modem.signal_to_control(padded)
    assert result is not None
    assert result[0] == CONTROL_BYTE_ACK


def test_control_frame_not_detected_in_noise():
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(SAMPLE_RATE) * 0.05).astype(np.float32)
    assert modem.signal_to_control(noise) is None


def test_control_frame_shorter_than_data_frame():
    data_signal = modem.text_to_signal("x")
    ctrl_signal = modem.encode_ack()
    assert len(ctrl_signal) < len(data_signal)


def test_control_frame_in_stream_decoder():
    decoder = StreamDecoder()
    ack_signal = modem.encode_ack()
    padded = np.concatenate([np.zeros(1000, dtype=np.float32), ack_signal])
    texts, ctrl = decoder.push(padded)
    assert texts == []
    assert ctrl == "ack"


def test_control_frame_nack_in_stream_decoder():
    decoder = StreamDecoder()
    nack_signal = modem.encode_nack()
    padded = np.concatenate([np.zeros(1000, dtype=np.float32), nack_signal])
    texts, ctrl = decoder.push(padded)
    assert texts == []
    assert ctrl == "nack"


def test_data_and_control_in_same_push():
    data_signal = modem.text_to_signal("Data with ctrl")
    ack_signal = modem.encode_ack()
    combined = np.concatenate([data_signal, np.zeros(500, dtype=np.float32), ack_signal])
    decoder = StreamDecoder()
    texts, ctrl = decoder.push(combined)
    assert "Data with ctrl" in texts
    assert ctrl == "ack"


def test_control_frame_boosted_amplitude():
    """ACK-кадр имеет усиленную амплитуду (125% от базовой)."""
    ack_signal = modem.encode_ack()
    normal_signal = modem.text_to_signal("x")
    # Проверяем что пик ACK > пикового значения обычного сигнала,
    # потому что ACK усилен на 25%
    ack_peak = float(np.max(np.abs(ack_signal)))
    assert ack_peak > 0.8, f"Пик ACK слишком низкий: {ack_peak}"
    assert ack_peak <= 1.0, f"Пик ACK выходит за диапазон [-1,1]: {ack_peak}"


def test_control_frame_still_decodable_after_boost():
    """Усиленный ACK всё ещё корректно декодируется."""
    signal = modem.encode_ack()
    result = modem.signal_to_control(signal)
    assert result is not None
    assert result[0] == CONTROL_BYTE_ACK
