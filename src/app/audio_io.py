"""
Разбор бинарных PCM-сообщений, приходящих от фронтенда по WebSocket.

Раньше бекенд слепо предполагал формат "сырые float32 LE сэмплы" для каждого
бинарного WS-фрейма. Любое расхождение с фактическим форматом на клиенте
(например, будущая оптимизация с отправкой int16 для экономии трафика, либо
баг в JS) молча превращалось в мусорные сэмплы и бесконечные "не найдено
преамбулы", без единой diagnostic-подсказки о причине.

Протокол теперь явный и самоописывающийся: первый байт сообщения — тег
формата, дальше — сырые сэмплы в соответствующем формате (little-endian).
Неизвестный тег или некорректная длина payload дают понятную ошибку вместо
тихого искажения сигнала.
"""

from __future__ import annotations

import numpy as np

FORMAT_FLOAT32 = 1
FORMAT_INT16 = 2

FORMAT_NAMES = {FORMAT_FLOAT32: "float32", FORMAT_INT16: "int16"}


def decode_pcm_message(raw: bytes) -> np.ndarray:
    """Разбирает бинарное WS-сообщение вида [тег формата][PCM-сэмплы] в
    массив float32 сэмплов в диапазоне [-1, 1].

    Бросает ValueError с понятным описанием при некорректном теге или длине.
    """
    if len(raw) < 2:
        raise ValueError(f"Слишком короткое PCM-сообщение: {len(raw)} байт (нужен хотя бы 1 сэмпл)")

    fmt = raw[0]
    payload = raw[1:]

    if fmt == FORMAT_FLOAT32:
        if len(payload) % 4 != 0:
            raise ValueError(
                f"Длина float32-payload не кратна 4 байтам: {len(payload)} байт"
            )
        return np.frombuffer(payload, dtype="<f4").astype(np.float32)

    if fmt == FORMAT_INT16:
        if len(payload) % 2 != 0:
            raise ValueError(
                f"Длина int16-payload не кратна 2 байтам: {len(payload)} байт"
            )
        ints = np.frombuffer(payload, dtype="<i2")
        return (ints.astype(np.float32)) / 32768.0

    raise ValueError(
        f"Неизвестный тег формата PCM-сообщения: {fmt} "
        f"(ожидался {FORMAT_FLOAT32}=float32 или {FORMAT_INT16}=int16)"
    )
