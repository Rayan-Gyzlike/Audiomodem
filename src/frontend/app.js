"use strict";

// ═══════════════════════════════════════════════════════════════════════════════
// Аудиомодем — фронтенд: кодирование текста/файлов в звук и декодирование
// обратно через WebSocket. Поддерживает FSK-модуляцию, Reed-Solomon,
// CRC32-проверку целостность, Repeat TX и ARQ (ACK/NACK).
// ═══════════════════════════════════════════════════════════════════════════════

const els = {
  micToggle: document.getElementById("micToggle"),
  connDot: document.getElementById("connDot"),
  connLabel: document.getElementById("connLabel"),
  waterfall: document.getElementById("waterfall"),
  freqAxis: document.getElementById("freqAxis"),
  chatLog: document.getElementById("chatLog"),
  sendForm: document.getElementById("sendForm"),
  textInput: document.getElementById("textInput"),
  sendBtn: document.getElementById("sendBtn"),
  fileInput: document.getElementById("fileInput"),
  fileBtn: document.getElementById("fileBtn"),
  clearChat: document.getElementById("clearChat"),
  footerStatus: document.getElementById("footerStatus"),
  manualAckBtn: document.getElementById("manualAckBtn"),
  // Прогресс-бар
  transferProgress: document.getElementById("transferProgress"),
  tpIcon: document.getElementById("tpIcon"),
  tpFilename: document.getElementById("tpFilename"),
  tpPhase: document.getElementById("tpPhase"),
  tpBarFill: document.getElementById("tpBarFill"),
  tpPct: document.getElementById("tpPct"),
  tpChunks: document.getElementById("tpChunks"),
  tpBytes: document.getElementById("tpBytes"),
};

const state = {
  listening: false,
  audioCtx: null,
  micStream: null,
  analyser: null,
  processor: null,
  ws: null,
  sampleRate: 44100, // целевая частота дискретизации бекенда (из /api/health)
  freqZero: 1200,
  freqOne: 2200,
  maxDisplayFreq: 4000,
  // ARQ: последнее отправленное сообщение для повтора при NACK/таймауте
  lastSentText: null,
  lastSentFile: null, // {formData, name, size}
  arqRetryCount: 0,
  arqMaxRetries: 3,
  arqTimeoutMs: 5000,
  arqTimer: null,
};

// ═══════════════════════════════════════════════════════════════════════════════
// Self-Muting: защита от эха (микрофон слышит динамик)
// ═══════════════════════════════════════════════════════════════════════════════
// При воспроизведении звука (данные, ACK, NACK) микрофон ДОЛЖЕН полностью
// игнорировать поступающие сэмплы, иначе устройство декодирует собственный
// сигнал и возникает зацикливание (самопередача).
//
// IS_TRANSMITTING = true → onaudioprocess дропает ВСЕ сэмплы, ничего не
// отправляет на сервер. Флаг сбрасывается через duration + ECHO_SAFETY_MS
// (600мс на затухание эха в помещении + опустошение системных буферов).

const ECHO_SAFETY_MS = 600; // мс на затухание эха после окончания звука

state.isTransmitting = false;
state._muteTimer = null;

// Тег формата бинарных PCM-сообщений, отправляемых в /ws/decode — должен
// совпадать с backend/app/audio_io.py.
const PCM_FORMAT_FLOAT32 = 1;

// ═══════════════════════════════════════════════════════════════════════════════
// Self-Muting: управление блокировкой микрофона
// ═══════════════════════════════════════════════════════════════════════════════

/** Активирует блокировку микрофона на durationMs миллисекунд.
 *  duringTransmission + safety margin после окончания звука. */
function activateMute(durationMs) {
  // Очищаем предыдущий таймер если был
  if (state._muteTimer) {
    clearTimeout(state._muteTimer);
    state._muteTimer = null;
  }
  state.isTransmitting = true;
  const totalMuteMs = durationMs + ECHO_SAFETY_MS;
  console.log(`[Audio Mute] Микрофон заблокирован на ${(totalMuteMs / 1000).toFixed(1)} сек (эхо-подавление)...`);
  state._muteTimer = setTimeout(() => {
    state.isTransmitting = false;
    state._muteTimer = null;
    console.log("[Audio Mute] Микрофон снова активен.");
  }, totalMuteMs);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ресэмплинг: линейная интерполяция AudioContext.sampleRate -> SAMPLE_RATE
// ═══════════════════════════════════════════════════════════════════════════════

// AudioContext({sampleRate}) — это лишь ПОЖЕЛАНИЕ браузеру; на части железа/ОС
// оно может быть проигнорировано. Ресэмплинг на клиенте гарантирует, что на
// сервер всегда приходит сигнал на целевой частоте.
function createResampler(fromRate, toRate) {
  if (fromRate === toRate) {
    return (input) => input;
  }
  const ratio = fromRate / toRate;
  let globalPos = 0;
  let totalConsumed = 0;
  let prevSample = null;

  return function resample(input) {
    const hasGuard = prevSample !== null;
    const extOffset = hasGuard ? -1 : 0;
    const output = [];

    let localPos = globalPos - (totalConsumed + extOffset);
    const extLength = input.length + (hasGuard ? 1 : 0);

    const sampleAt = (k) => (hasGuard ? (k === 0 ? prevSample : input[k - 1]) : input[k]);

    while (localPos + 1 < extLength) {
      const i0 = Math.floor(localPos);
      const frac = localPos - i0;
      output.push(sampleAt(i0) * (1 - frac) + sampleAt(i0 + 1) * frac);
      globalPos += ratio;
      localPos += ratio;
    }

    if (input.length > 0) {
      prevSample = input[input.length - 1];
    }
    totalConsumed += input.length;

    return Float32Array.from(output);
  };
}

// Собирает бинарное WS-сообщение: [1 байт тег формата][сырые float32 LE].
function buildPcmMessage(samples) {
  const message = new Uint8Array(1 + samples.byteLength);
  message[0] = PCM_FORMAT_FLOAT32;
  message.set(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength), 1);
  return message.buffer;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Чат-лог и UI-статусы
// ═══════════════════════════════════════════════════════════════════════════════

function addBubble(kind, text, meta) {
  document.querySelector(".hint")?.remove();
  const el = document.createElement("div");
  el.className = `bubble ${kind}`;

  const textNode = document.createElement("div");
  textNode.textContent = text;
  el.appendChild(textNode);

  if (meta) {
    const metaEl = document.createElement("span");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    el.appendChild(metaEl);
  }

  els.chatLog.appendChild(el);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " Б";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " КБ";
  return (bytes / (1024 * 1024)).toFixed(1) + " МБ";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Прогресс-бар: показ / скрытие / обновление
// ═══════════════════════════════════════════════════════════════════════════════

/** Показать прогресс-бар. direction = "tx" | "rx", phase = описание фазы */
function showProgress(direction, filename, phase) {
  els.transferProgress.style.display = "";
  els.tpIcon.textContent = direction === "tx" ? "⬆" : "⬇";
  els.tpFilename.textContent = filename;
  els.tpPhase.textContent = phase || "";
  els.tpBarFill.style.width = "0%";
  els.tpPct.textContent = "0%";
  els.tpChunks.textContent = "";
  els.tpBytes.textContent = "";
}

/** Обновить прогресс-бар */
function updateProgress(pct, chunksText, bytesText, phase) {
  els.tpBarFill.style.width = pct + "%";
  els.tpPct.textContent = pct + "%";
  if (chunksText !== undefined) els.tpChunks.textContent = chunksText;
  if (bytesText !== undefined) els.tpBytes.textContent = bytesText;
  if (phase !== undefined) els.tpPhase.textContent = phase;
}

/** Скрыть прогресс-бар */
function hideProgress() {
  els.transferProgress.style.display = "none";
  els.tpBarFill.style.width = "0%";
}

function timeLabel() {
  return new Date().toLocaleTimeString("ru-RU", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function setFooter(text) {
  els.footerStatus.textContent = text;
}

function setConnStatus(kind, label) {
  els.connDot.className = "dot" + (kind === "live" ? " live" : kind === "error" ? " error" : "");
  els.connLabel.textContent = label;
}

els.clearChat.addEventListener("click", () => {
  els.chatLog.innerHTML =
    '<div class="hint">Отправьте сообщение или включите микрофон, чтобы принимать звук.</div>';
});

// ═══════════════════════════════════════════════════════════════════════════════
// Manual ACK: кнопка ручного подтверждения приёма
// ═══════════════════════════════════════════════════════════════════════════════

els.manualAckBtn.addEventListener("click", () => {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    // Отправляем manual_ack на бэкенд
    state.ws.send(JSON.stringify({ type: "manual_ack" }));
    // Воспроизводим ACK локально через Web Audio API
    generateAndPlayAckLocally();
    // Обновляем UI: блокируем кнопку
    els.manualAckBtn.textContent = "Подтверждено вручную ✓";
    els.manualAckBtn.classList.add("confirmed");
    els.manualAckBtn.disabled = true;
    setFooter("Подтверждение приёма отправлено вручную");
    console.log("[MANUAL ACK] Ручное подтверждение отправлено");
  }
});

/** Генерирует и воспроизводит ACK локально через Web Audio API (FSK-тон). */
function generateAndPlayAckLocally() {
  const ctx = ensureAudioContext();
  if (ctx.state === "suspended") ctx.resume();
  // ACK: 8бит преамбула + 16бит sync + 8бит command = 32 бита
  // Частоты: 0→1200Гц, 1→2200Гц, 100 baud = 10мс/бит
  const preambleBits = [0,1,0,1,0,1,0,1];
  const syncBits = [0,0,1,0,1,1,0,1,1,1,0,1,0,1,0,0];
  const ackBits = [0,0,0,0,0,1,1,0]; // 0x06 = ACK
  const allBits = [...preambleBits, ...syncBits, ...ackBits];
  const baudRate = 100;
  const samplesPerBit = Math.round(ctx.sampleRate / baudRate);
  const totalSamples = allBits.length * samplesPerBit;
  const buffer = ctx.createBuffer(1, totalSamples, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  let phase = 0;
  for (let i = 0; i < allBits.length; i++) {
    const freq = allBits[i] === 1 ? 2200 : 1200;
    for (let j = 0; j < samplesPerBit; j++) {
      data[i * samplesPerBit + j] = 0.9 * Math.sin(phase);
      phase += (2 * Math.PI * freq) / ctx.sampleRate;
    }
  }
  // Блокируем микрофон на время воспроизведения + safety margin
  const durationMs = (totalSamples / ctx.sampleRate) * 1000;
  activateMute(durationMs);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  source.start();
}

// ═══════════════════════════════════════════════════════════════════════════════
// ARQ: повторная отправка при NACK или таймауте ожидания подтверждения
// ═══════════════════════════════════════════════════════════════════════════════

function startArqTimer() {
  clearArqTimer();
  state.arqTimer = setTimeout(() => {
    // Таймаут ожидания ACK/NACK — повторяем отправку
    if (state.arqRetryCount < state.arqMaxRetries) {
      state.arqRetryCount++;
      console.log(`[ARQ] Таймаут — повтор отправки (попытка ${state.arqRetryCount}/${state.arqMaxRetries})`);
      setFooter(`ARQ: повтор ${state.arqRetryCount}/${state.arqMaxRetries}...`);
      retryLastSend();
    } else {
      console.log("[ARQ] Превышен лимит повторов");
      addBubble("error", "Не удалось доставить: превышен лимит повторов");
      setFooter("Доставка не удалась (превышен лимит повторов)");
      hideProgress();
      clearArqState();
    }
  }, state.arqTimeoutMs);
}

function clearArqTimer() {
  if (state.arqTimer) {
    clearTimeout(state.arqTimer);
    state.arqTimer = null;
  }
}

function clearArqState() {
  clearArqTimer();
  state.arqRetryCount = 0;
  state.lastSentText = null;
  state.lastSentFile = null;
}

async function retryLastSend() {
  if (state.lastSentText !== null) {
    await resendText(state.lastSentText);
  } else if (state.lastSentFile !== null) {
    await resendFile(state.lastSentFile);
  }
}

async function resendText(text) {
  try {
    const resp = await fetch("/api/encode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) return;
    const arrayBuffer = await resp.arrayBuffer();
    const headerDuration = parseFloat(resp.headers.get("X-Playback-Duration")) || 0;
    setFooter(`ARQ: повторная передача...`);
    const playbackDuration = await playAudioBuffer(arrayBuffer, headerDuration);
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "tx_complete", duration: playbackDuration }));
    }
    startArqTimer();
  } catch (err) {
    console.error("[ARQ] Ошибка повторной отправки:", err);
  }
}

async function resendFile(fileData) {
  try {
    const formData = new FormData();
    formData.append("file", fileData.file);
    const resp = await fetch("/api/encode-file", {
      method: "POST",
      body: formData,
    });
    if (!resp.ok) return;
    const arrayBuffer = await resp.arrayBuffer();
    const headerDuration = parseFloat(resp.headers.get("X-Playback-Duration")) || 0;
    setFooter(`ARQ: повторная передача файла ${fileData.name}...`);
    const playbackDuration = await playAudioBuffer(arrayBuffer, headerDuration);
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "tx_complete", duration: playbackDuration }));
    }
    startArqTimer();
  } catch (err) {
    console.error("[ARQ] Ошибка повторной отправки файла:", err);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Отправка текста (TX): текст -> /api/encode -> воспроизведение через динамики
// ═══════════════════════════════════════════════════════════════════════════════

els.sendForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = els.textInput.value.trim();
  if (!text) return;

  els.sendBtn.disabled = true;
  setFooter("Кодирование...");
  try {
    // Случайная задержка для избежания коллизий (100-500 мс)
    const backoffMs = Math.floor(Math.random() * 400) + 100;
    setFooter(`Ожидание канала (${backoffMs} мс)...`);
    await new Promise(r => setTimeout(r, backoffMs));
    setFooter("Кодирование и передача сигнала...");
    const resp = await fetch("/api/encode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!resp.ok) {
      const msg = await resp.text();
      addBubble("error", `Ошибка отправки: ${msg}`);
      setFooter("Ошибка отправки");
      return;
    }

    const arrayBuffer = await resp.arrayBuffer();
    const headerDuration = parseFloat(resp.headers.get("X-Playback-Duration")) || 0;
    addBubble("sent", text, `Отправлено · ${timeLabel()}`);
    els.textInput.value = "";
    // Сохраняем для ARQ повторной отправки
    state.lastSentText = text;
    state.lastSentFile = null;
    state.arqRetryCount = 0;
    setFooter("Передача сигнала через динамики...");
    const playbackDuration = await playAudioBuffer(arrayBuffer, headerDuration);
    // Self-muting: сообщаем серверу об окончании воспроизведения с точной длительностью
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "tx_complete", duration: playbackDuration }));
    }
    setFooter("Ожидание ACK/NACK...");
    startArqTimer();
  } catch (err) {
    addBubble("error", `Сбой сети: ${err.message}`);
    setFooter("Ошибка");
    clearArqState();
  } finally {
    els.sendBtn.disabled = false;
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// Отправка файла (TX): файл -> /api/encode-file -> воспроизведение
// ═══════════════════════════════════════════════════════════════════════════════

els.fileBtn.addEventListener("click", () => {
  els.fileInput.click();
});

els.fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  els.fileBtn.disabled = true;
  showProgress("tx", file.name, "Кодирование...");
  setFooter(`Кодирование файла: ${file.name}...`);
  try {
    // Случайная задержка для избежания коллизий (100-500 мс)
    const backoffMs = Math.floor(Math.random() * 400) + 100;
    setFooter(`Ожидание канала (${backoffMs} мс)...`);
    updateProgress(0, "", "", "Ожидание канала...");
    await new Promise(r => setTimeout(r, backoffMs));

    const formData = new FormData();
    formData.append("file", file);

    setFooter(`Кодирование и передача файла: ${file.name}...`);
    updateProgress(0, "", formatBytes(file.size), "Кодирование...");
    const resp = await fetch("/api/encode-file", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      const msg = await resp.text();
      addBubble("error", `Ошибка отправки файла: ${msg}`);
      setFooter("Ошибка отправки файла");
      hideProgress();
      return;
    }

    const arrayBuffer = await resp.arrayBuffer();
    const headerDuration = parseFloat(resp.headers.get("X-Playback-Duration")) || 0;
    addBubble("sent", `[Файл: ${file.name}] (${formatBytes(file.size)})`, `Отправлено · ${timeLabel()}`);
    // Сохраняем для ARQ повторной отправки
    state.lastSentFile = { file, name: file.name, size: file.size };
    state.lastSentText = null;
    state.arqRetryCount = 0;
    setFooter(`Передача файла ${file.name} через динамики...`);
    updateProgress(5, "", formatBytes(file.size), "Передача звука...");
    const playbackDuration = await playAudioBuffer(arrayBuffer, headerDuration);
    // Анимация прогресса во время воспроизведения (от 5% до 95%)
    const txProgressEnd = () => updateProgress(95, "", formatBytes(file.size), "Ожидание подтверждения...");
    txProgressEnd();
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify({ type: "tx_complete", duration: playbackDuration }));
    }
    setFooter("Ожидание ACK/NACK...");
    startArqTimer();
  } catch (err) {
    addBubble("error", `Сбой при отправке файла: ${err.message}`);
    setFooter("Ошибка");
    hideProgress();
    clearArqState();
  } finally {
    els.fileBtn.disabled = false;
    els.fileInput.value = "";
  }
});

async function playAudioBuffer(arrayBuffer, headerDuration) {
  const ctx = ensureAudioContext();
  if (ctx.state === "suspended") await ctx.resume();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));

  // Блокируем микрофон НА ВРЕМЯ воспроизведения + safety margin
  const durationMs = audioBuffer.duration * 1000;
  activateMute(durationMs);

  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);
  source.start();

  // Возвращаем длительность воспроизведения для ARQ-протокола
  const durationSec = headerDuration || audioBuffer.duration;
  return new Promise((resolve) => {
    source.onended = () => resolve(durationSec);
  });
}

/** Воспроизводит PCM16 (base64, LE mono) контрольного кадра через Web Audio API.
 *  Используется для ACK/NACK — звук генерируется браузером, без sounddevice. */
async function playControlSignalAsync(pcmB64, durationSec) {
  const ctx = ensureAudioContext();
  if (ctx.state === "suspended") await ctx.resume();

  // Декодируем base64 → Uint8Array → Int16Array → Float32Array
  const raw = atob(pcmB64);
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);

  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) {
    float32[i] = int16[i] / 32768;
  }

  const audioBuffer = ctx.createBuffer(1, float32.length, state.sampleRate);
  audioBuffer.getChannelData(0).set(float32);

  // Блокируем микрофон на время воспроизведения + safety margin
  activateMute(audioBuffer.duration * 1000);

  const source = ctx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(ctx.destination);
  source.start();
}

function ensureAudioContext() {
  if (!state.audioCtx) {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    state.audioCtx = new Ctx({ sampleRate: state.sampleRate });
  }
  return state.audioCtx;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Приём (RX): микрофон -> WebSocket -> декодированный текст
// ═══════════════════════════════════════════════════════════════════════════════

els.micToggle.addEventListener("click", () => {
  if (state.listening) {
    stopListening();
  } else {
    startListening();
  }
});

async function startListening() {
  try {
    const ctx = ensureAudioContext();
    if (ctx.state === "suspended") await ctx.resume();

    state.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });

    const source = ctx.createMediaStreamSource(state.micStream);

    state.analyser = ctx.createAnalyser();
    state.analyser.fftSize = 2048;
    state.analyser.smoothingTimeConstant = 0.35;

    state.processor = ctx.createScriptProcessor(4096, 1, 1);

    source.connect(state.analyser);
    source.connect(state.processor);

    // ScriptProcessorNode нужно подключить к какому-либо выходу, чтобы
    // событие onaudioprocess вообще срабатывало; беззвучный GainNode.
    const silentGain = ctx.createGain();
    silentGain.gain.value = 0;
    state.processor.connect(silentGain);
    silentGain.connect(ctx.destination);

    const actualCtxRate = ctx.sampleRate;
    const needsResample = actualCtxRate !== state.sampleRate;
    console.log(
      `[audio-modem] AudioContext.sampleRate=${actualCtxRate}, целевой SAMPLE_RATE=${state.sampleRate}` +
        (needsResample ? " — включён клиентский ресэмплинг" : " — совпадают")
    );

    connectWebSocket(actualCtxRate);

    const resample = createResampler(actualCtxRate, state.sampleRate);

    state.processor.onaudioprocess = (event) => {
      if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;

      // Self-muting: во время передачи звука ПОЛНОСТЬЮ игнорируем микрофон,
      // чтобы не декодировать собственный сигнал динамика (эхо).
      if (state.isTransmitting) return;

      const input = event.inputBuffer.getChannelData(0);
      const resampled = needsResample ? resample(input) : input;
      const copy = new Float32Array(resampled);
      state.ws.send(buildPcmMessage(copy));
    };

    state.listening = true;
    els.micToggle.textContent = "Остановить";
    els.micToggle.classList.add("listening");
    setConnStatus("connecting", "Подключение...");
    setFooter("Приём кадра — ожидание сигнала...");

    requestAnimationFrame(drawWaterfall);
  } catch (err) {
    addBubble("error", `Не удалось получить доступ к микрофону: ${err.message}`);
    setFooter("Ошибка доступа к микрофону");
  }
}

function stopListening() {
  state.listening = false;
  els.micToggle.textContent = "Слушать";
  els.micToggle.classList.remove("listening");

  if (state.processor) {
    state.processor.onaudioprocess = null;
    state.processor.disconnect();
    state.processor = null;
  }
  if (state.analyser) {
    state.analyser.disconnect();
    state.analyser = null;
  }
  if (state.micStream) {
    state.micStream.getTracks().forEach((t) => t.stop());
    state.micStream = null;
  }
  if (state.ws) {
    state.ws.close();
    state.ws = null;
  }

  setConnStatus("off", "Микрофон выключен");
  setFooter("Остановлено");
}

function connectWebSocket(actualCtxRate) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/decode`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setConnStatus("live", "Прослушивание...");
    ws.send(
      JSON.stringify({
        type: "hello",
        sampleRate: state.sampleRate,
        clientContextRate: actualCtxRate,
      })
    );
  };
  ws.onclose = () => {
    if (state.listening) setConnStatus("error", "Соединение потеряно");
  };
  ws.onerror = () => setConnStatus("error", "Ошибка соединения");
  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "received") {
      const metrics = msg.metrics || {};
      const metaParts = [`Принято · ${timeLabel()}`];
      if (msg.file_info) {
        const fi = msg.file_info;
        // Показываем 100% и скрываем через момент
        updateProgress(100, `${fi.chunks_total}/${fi.chunks_total}`, formatBytes(fi.decoded_size), "Готово!");
        setTimeout(hideProgress, 1500);
        addBubble("received", `[Файл восстановлен: ${fi.filename}] (${formatBytes(fi.decoded_size)})`,
          metaParts.join(" ") + " · CRC32: ОК");
        setFooter(`Файл восстановлен: ${fi.filename} (${formatBytes(fi.decoded_size)})`);
      } else {
        if (metrics.decode_count) metaParts.push(`Пакет #${metrics.decode_count}`);
        if (metrics.session_elapsed_sec) metaParts.push(`${metrics.session_elapsed_sec} сек`);
        const crcStatus = msg.crc_ok ? "CRC32: ОК" : "CRC32: ОШИБКА";
        metaParts.push(crcStatus);
        addBubble("received", msg.text, metaParts.join(" · "));
        setFooter(`${crcStatus} · Сообщение принято`);
      }
    } else if (msg.type === "file_progress") {
      // Прогресс-бар при приёме фрагментированного файла
      const pct = msg.progress_pct || 0;
      const received = msg.chunks_received;
      const total = msg.chunks_total;
      const bytes = msg.bytes_received || 0;
      const totalBytes = msg.total_size || 0;
      const chunksText = `${received}/${total}`;
      const bytesText = totalBytes > 0 ? `${formatBytes(bytes)} / ${formatBytes(totalBytes)}` : formatBytes(bytes);
      showProgress("rx", msg.filename, "Приём...");
      updateProgress(pct, chunksText, bytesText);
      setFooter(`Приём: ${msg.filename} ${pct}% (${received}/${total})`);
    } else if (msg.type === "control_signal") {
      // Бэкенд отправляет PCM данные контрольного кадра (ACK/NACK) —
      // браузер воспроизводит звук через Web Audio API.
      // Полностью исключает фризы sounddevice на Windows/AMD.
      const b64 = msg.pcm;
      const signal = msg.signal;
      const duration = msg.duration || 0.5;
      if (b64) {
        playControlSignalAsync(b64, duration);
        console.log(`[Audio OUT] Воспроизвожу ${signal} через браузер (${duration.toFixed(2)} сек)`);
        setFooter(`Воспроизвожу ${signal}...`);
      }
    } else if (msg.type === "play_ctrl") {
      // Бэкенд запросил блокировку микрофона (ACK/NACK воспроизводится браузером)
      const ctrlDuration = (msg.duration || 0.5) * 1000;
      activateMute(ctrlDuration);
    } else if (msg.type === "arq_status") {
      if (msg.status === "delivered") {
        clearArqState();
        hideProgress();
        const tsNow = new Date().toLocaleTimeString("ru-RU");
        console.log(`[TX SUCCESS] ACK получен — доставка подтверждена (Time: ${tsNow})`);
        addBubble("arq", msg.message, `Доставлено · ${timeLabel()}`);
        setFooter("Доставлено успешно (ACK)");
      } else {
        // NACK — повторяем отправку если есть попытки
        if (state.arqRetryCount < state.arqMaxRetries) {
          state.arqRetryCount++;
          const tsNack = new Date().toLocaleTimeString("ru-RU");
          console.log(`[TX NACK] NACK получен — повтор ${state.arqRetryCount}/${state.arqMaxRetries} (Time: ${tsNack})`);
          setFooter(`ARQ: NACK — повтор ${state.arqRetryCount}/${state.arqMaxRetries}...`);
          addBubble("error", msg.message || "NACK получен, повторная отправка...");
          retryLastSend();
        } else {
          clearArqState();
          addBubble("error", msg.message || "Ошибка доставки: превышен лимит повторов");
          setFooter("Ошибка доставки (превышен лимит)");
          hideProgress();
        }
      }
    } else if (msg.type === "error") {
      setFooter(`Ошибка: ${msg.message}`);
    } else if (msg.type === "status") {
      setConnStatus("live", "Прослушивание...");
    }
  };

  state.ws = ws;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Водопадная спектрограмма (Canvas)
// ═══════════════════════════════════════════════════════════════════════════════

const waterfallCanvas = els.waterfall;
const waterfallCtx = waterfallCanvas.getContext("2d");
const axisCanvas = els.freqAxis;
const axisCtx = axisCanvas.getContext("2d");
let freqData = null;

function resizeCanvases() {
  const wrap = waterfallCanvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;

  waterfallCanvas.width = Math.max(1, Math.floor(w * dpr));
  waterfallCanvas.height = Math.max(1, Math.floor(h * dpr));
  waterfallCanvas.style.width = w + "px";
  waterfallCanvas.style.height = h + "px";

  axisCanvas.width = Math.floor(46 * dpr);
  axisCanvas.height = Math.max(1, Math.floor(h * dpr));
  axisCanvas.style.width = "46px";
  axisCanvas.style.height = h + "px";

  waterfallCtx.fillStyle = "#050810";
  waterfallCtx.fillRect(0, 0, waterfallCanvas.width, waterfallCanvas.height);
  drawFreqAxis();
}

window.addEventListener("resize", resizeCanvases);

function drawFreqAxis() {
  const h = axisCanvas.height;
  const dpr = window.devicePixelRatio || 1;
  axisCtx.clearRect(0, 0, axisCanvas.width, h);
  axisCtx.fillStyle = "#8b97ac";
  axisCtx.font = `${11 * dpr}px sans-serif`;
  axisCtx.textBaseline = "middle";

  [0, 1000, 2000, 3000, 4000].forEach((f) => {
    const y = h - (f / state.maxDisplayFreq) * h;
    axisCtx.fillText(`${f}`, 4, Math.min(Math.max(y, 10 * dpr), h - 4 * dpr));
  });
}

function freqToColor(v) {
  const t = v / 255;
  let r, g, b;
  if (t < 0.35) {
    const k = t / 0.35;
    r = 5;
    g = Math.floor(10 + k * 30);
    b = Math.floor(20 + k * 60);
  } else if (t < 0.7) {
    const k = (t - 0.35) / 0.35;
    r = Math.floor(k * 60);
    g = Math.floor(120 + k * 100);
    b = Math.floor(150 - k * 60);
  } else {
    const k = (t - 0.7) / 0.3;
    r = Math.floor(120 + k * 135);
    g = Math.floor(220 - k * 120);
    b = Math.floor(90 - k * 70);
  }
  return `rgb(${r},${g},${b})`;
}

function drawWaterfall() {
  if (!state.listening || !state.analyser) return;

  const analyser = state.analyser;
  if (!freqData || freqData.length !== analyser.frequencyBinCount) {
    freqData = new Uint8Array(analyser.frequencyBinCount);
  }
  analyser.getByteFrequencyData(freqData);

  const w = waterfallCanvas.width;
  const h = waterfallCanvas.height;

  if (w > 1) {
    const imgData = waterfallCtx.getImageData(1, 0, w - 1, h);
    waterfallCtx.putImageData(imgData, 0, 0);
  }

  const nyquist = state.sampleRate / 2;
  for (let y = 0; y < h; y++) {
    const freq = (1 - y / h) * state.maxDisplayFreq;
    const bin = Math.min(freqData.length - 1, Math.max(0, Math.round((freq / nyquist) * freqData.length)));
    waterfallCtx.fillStyle = freqToColor(freqData[bin] || 0);
    waterfallCtx.fillRect(w - 1, y, 1, 1);
  }

  drawMarkerPixel(state.freqZero, "rgba(255,107,107,0.6)");
  drawMarkerPixel(state.freqOne, "rgba(79,209,197,0.6)");

  requestAnimationFrame(drawWaterfall);
}

function drawMarkerPixel(freq, color) {
  const h = waterfallCanvas.height;
  const w = waterfallCanvas.width;
  const y = Math.round(h - (freq / state.maxDisplayFreq) * h);
  waterfallCtx.fillStyle = color;
  waterfallCtx.fillRect(w - 1, y, 1, 1);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Инициализация
// ═══════════════════════════════════════════════════════════════════════════════

async function init() {
  try {
    const resp = await fetch("/api/health");
    const health = await resp.json();
    state.sampleRate = health.sample_rate;
    state.freqZero = health.freq_zero;
    state.freqOne = health.freq_one;
  } catch {
    // бекенд ещё не готов — работаем со значениями по умолчанию
  }
  resizeCanvases();
}

init();
