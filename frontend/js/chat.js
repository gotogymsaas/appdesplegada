// chat.js - Enfoque Modular
(function () {
// 0. Validación de Visibilidad
// No mostrar si estamos en la página de login o registro, o si no hay usuario
const path = window.location.pathname;
const isAuthPage = path.includes('indexInicioDeSesion') ||
path.includes('indexRegistro') || (path.endsWith('/') &&
!localStorage.getItem('user'));
if (isAuthPage) {
// Opción: Limpiar chat previo si existe
const existing = document.getElementById('chat-widget-container');
if (existing) existing.remove();
console.log("Chat oculto: Página de autenticación.");
return;
}
// 1. Inyectar CSS
const link = document.createElement('link');
link.rel = "stylesheet";
const CHAT_CSS_VERSION = '2026-02-21-12';
link.href = `/css/chat.css?v=${CHAT_CSS_VERSION}`; // Ruta absoluta desde raíz del servidor // frontend
// Si estás en subcarpetas, esto funciona si el server sirve desde raíz.
// Si falla, intentaremos ruta relativa automática
document.head.appendChild(link);
// 2. Crear HTML estático
const container = document.createElement('div');
container.id = 'chat-widget-container';
container.innerHTML = `
<button id="chat-toggle-btn">
<svg viewBox="0 0 24 24"><path d="M20 2H4c-1.1 0-2 .9-2
2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2
2V4h16v12z"/></svg>
</button>
<div id="chat-window">
<div id="chat-header">
<h3>GoToGym Quantum Coach</h3>
<div class="chat-header-actions">
<button id="chat-reset-btn" type="button" title="Reiniciar chat">↻</button>
<button id="chat-close-btn" type="button" title="Cerrar">✕</button>
</div>
</div>
<div id="chat-messages">
</div>
<form id="chat-input-area" autocomplete="off">
  <div id="chat-attachment-preview" hidden></div>

  <div class="chat-footer-row">
    <button type="button" id="chat-plus-btn" aria-label="Más opciones" title="Más">+</button>

    <textarea id="chat-input" rows="1" placeholder="Escribe tu duda..." autocomplete="off"></textarea>

    <button type="submit" id="chat-send-btn" aria-label="Enviar" title="Enviar">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4l7 7-1.4 1.4L13 7.8V20h-2V7.8L6.4 12.4 5 11l7-7z"/></svg>
    </button>
  </div>

  <div id="chat-tools-menu" hidden>
    <button type="button" id="chat-tool-mic">🎤 Micrófono</button>
    <button type="button" id="chat-tool-camera">📷 Tomar foto</button>
    <button type="button" id="chat-tool-attach">📎 Adjuntar</button>
  </div>

  <input type="file" id="chat-file-input" hidden accept="image/*,.pdf">
  <input type="file" id="chat-camera-input" hidden accept="image/*" capture="environment">
</form>
</div>
`;
document.body.appendChild(container);
// 3. Lógica
const widgetContainer = document.getElementById('chat-widget-container');
const toggleBtn = document.getElementById('chat-toggle-btn');
const chatWindow = document.getElementById('chat-window');
const form = document.getElementById('chat-input-area');
const input = document.getElementById('chat-input');
const fileInput = document.getElementById('chat-file-input');
const cameraInput = document.getElementById('chat-camera-input');
const attachmentPreview = document.getElementById('chat-attachment-preview');
const plusBtn = document.getElementById('chat-plus-btn');
const toolsMenu = document.getElementById('chat-tools-menu');
const toolMicBtn = document.getElementById('chat-tool-mic');
const toolCameraBtn = document.getElementById('chat-tool-camera');
const toolAttachBtn = document.getElementById('chat-tool-attach');
const micBtn = toolMicBtn;
const messages = document.getElementById('chat-messages');

// --- Exp-006 Postura (pose-estimation en cliente, sin subir fotos) ---
let postureFlow = {
  active: false,
  step: 'idle', // idle | need_front | need_side | need_safety | ready
  captureTarget: null, // 'front' | 'side'
  poses: { front: null, side: null },
  userContext: { pain_neck: false, pain_low_back: false, injury_recent: false, level: 'beginner' },
};

let posturePoseEstimator = null;
let posturePoseEstimatorLoading = null;

const POSTURE_STATE_KEY = 'gtg_posture_flow_state_v0';

function savePostureState() {
  try {
    localStorage.setItem(POSTURE_STATE_KEY, JSON.stringify(postureFlow));
  } catch (e) {
    // ignore
  }
}

function loadPostureState() {
  try {
    const raw = localStorage.getItem(POSTURE_STATE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    if (!parsed.active) return;
    postureFlow = {
      ...postureFlow,
      ...parsed,
      poses: parsed.poses || postureFlow.poses,
      userContext: parsed.userContext || postureFlow.userContext,
    };
  } catch (e) {
    // ignore
  }
}

loadPostureState();

function startPostureFlow() {
  postureFlow = {
    active: true,
    step: 'need_front',
    captureTarget: null,
    poses: { front: null, side: null },
    userContext: { pain_neck: false, pain_low_back: false, injury_recent: false, level: 'beginner' },
  };
  savePostureState();

  appendMessage(
    'Para analizar tu postura necesito 2 fotos: **frontal** y **lateral** (perfil).\n\n' +
      '- Cámara a la altura del pecho, a 2–3m\n' +
      '- Cuerpo completo (pies a cabeza)\n' +
      '- Buena luz y fondo limpio\n' +
      '- Brazos relajados\n\n' +
      'Empecemos con la foto frontal.',
    'bot'
  );
  appendQuickActions([
    { label: 'Tomar foto frontal', type: 'posture_capture', view: 'front', source: 'camera' },
    { label: 'Adjuntar foto frontal', type: 'posture_capture', view: 'front', source: 'attach' },
    { label: 'Cancelar', type: 'posture_cancel' },
  ]);
}

function cancelPostureFlow() {
  postureFlow = { ...postureFlow, active: false, step: 'idle', captureTarget: null };
  try {
    localStorage.removeItem(POSTURE_STATE_KEY);
  } catch (e) {
    // ignore
  }
  appendMessage('Listo. Si quieres retomarlo, toca "Postura" o escribe: "analizar postura".', 'bot');
}

function _isImageFile(file) {
  if (!file) return false;
  const mime = String(file.type || '').toLowerCase();
  if (mime.startsWith('image/')) return true;
  const name = String(file.name || '').toLowerCase();
  return name.endsWith('.png') || name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.webp') || name.endsWith('.heic') || name.endsWith('.heif');
}

function _loadScriptOnce(src) {
  return new Promise((resolve, reject) => {
    if (!src) return reject(new Error('missing_src'));
    const existing = document.querySelector(`script[data-gtg-src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === '1') return resolve();
      existing.addEventListener('load', () => resolve());
      existing.addEventListener('error', () => reject(new Error('load_failed')));
      return;
    }
    const s = document.createElement('script');
    s.src = src;
    s.async = true;
    s.defer = true;
    s.dataset.gtgSrc = src;
    s.addEventListener('load', () => {
      s.dataset.loaded = '1';
      resolve();
    });
    s.addEventListener('error', () => reject(new Error('load_failed')));
    document.head.appendChild(s);
  });
}

async function getPosturePoseEstimator() {
  if (posturePoseEstimator) return posturePoseEstimator;
  if (posturePoseEstimatorLoading) return posturePoseEstimatorLoading;

  posturePoseEstimatorLoading = (async () => {
    const POSE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js';
    await _loadScriptOnce(POSE_CDN);

    if (typeof Pose !== 'function') {
      throw new Error('mediapipe_pose_not_available');
    }

    const estimator = new Pose({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
    });

    estimator.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      enableSegmentation: false,
      selfieMode: false,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    posturePoseEstimator = estimator;
    return estimator;
  })();

  try {
    return await posturePoseEstimatorLoading;
  } finally {
    posturePoseEstimatorLoading = null;
  }
}

function mediapipeLandmarksToKeypoints(landmarks) {
  const lm = Array.isArray(landmarks) ? landmarks : [];
  const map = {
    0: 'nose',
    7: 'left_ear',
    8: 'right_ear',
    11: 'left_shoulder',
    12: 'right_shoulder',
    23: 'left_hip',
    24: 'right_hip',
    25: 'left_knee',
    26: 'right_knee',
    27: 'left_ankle',
    28: 'right_ankle',
  };
  const out = [];
  Object.keys(map).forEach((idxStr) => {
    const idx = Number(idxStr);
    const name = map[idx];
    const p = lm[idx];
    if (!p) return;
    const x = Number(p.x);
    const y = Number(p.y);
    const score = Number(p.visibility);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    out.push({
      name,
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y)),
      score: Number.isFinite(score) ? score : 0,
    });
  });
  return out;
}

async function estimatePoseFromImageFile(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.decoding = 'async';
    img.loading = 'eager';
    img.src = url;
    await new Promise((resolve, reject) => {
      img.onload = () => resolve();
      img.onerror = () => reject(new Error('image_load_failed'));
    });

    const estimator = await getPosturePoseEstimator();

    const res = await new Promise((resolve) => {
      estimator.onResults((results) => resolve(results || {}));
      estimator.send({ image: img });
    });

    const kps = mediapipeLandmarksToKeypoints(res.poseLandmarks);
    return {
      keypoints: kps,
      image: { width: img.naturalWidth || img.width || 0, height: img.naturalHeight || img.height || 0 },
    };
  } finally {
    try {
      URL.revokeObjectURL(url);
    } catch (e) {
      // ignore
    }
  }
}

async function handlePostureCapture(file, view) {
  if (!_isImageFile(file)) {
    appendMessage('Para postura necesito una imagen (frontal o lateral).', 'bot');
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  appendMessage(view === 'front' ? 'Foto frontal' : 'Foto lateral', 'user', {
    meta: { text: view === 'front' ? 'Foto frontal' : 'Foto lateral', hasFile: true },
    attachment: { file, objectUrl },
  });

  appendMessage('Analizando postura (pose estimation local)...', 'bot');

  let pose;
  try {
    pose = await estimatePoseFromImageFile(file);
  } catch (e) {
    console.warn('Pose estimation failed:', e);
    appendMessage('No pude detectar bien tu cuerpo en la imagen. Repite con mejor luz y el cuerpo completo visible.', 'bot');
    return;
  }

  postureFlow.poses[view] = pose;
  postureFlow.captureTarget = null;
  savePostureState();

  if (view === 'front') {
    postureFlow.step = 'need_side';
    savePostureState();
    appendMessage('Perfecto. Ahora necesito la foto **lateral** (perfil).', 'bot');
    appendQuickActions([
      { label: 'Tomar foto lateral', type: 'posture_capture', view: 'side', source: 'camera' },
      { label: 'Adjuntar foto lateral', type: 'posture_capture', view: 'side', source: 'attach' },
      { label: 'Cancelar', type: 'posture_cancel' },
    ]);
    return;
  }

  postureFlow.step = 'need_safety';
  savePostureState();
  appendMessage('Antes de recomendar ejercicios: ¿tienes dolor agudo, hormigueo o lesión reciente?', 'bot');
  appendQuickActions([
    { label: 'No', type: 'posture_safety', value: 'no' },
    { label: 'Sí', type: 'posture_safety', value: 'yes' },
    { label: 'Cancelar', type: 'posture_cancel' },
  ]);
}

// --- Chat persistence (24h + reset diario) ---
const CHAT_STORAGE_KEY = 'gtg_chat_history';
const CHAT_META_KEY = 'gtg_chat_meta';
const SESSION_DAY_KEY = 'gtg_session_day';
const CHAT_TTL_MS = 24 * 60 * 60 * 1000;
const ONBOARDING_START_KEY = 'gtg_chat_onboarding_start';
const ONBOARDING_LAST_KEY = 'gtg_chat_onboarding_last';
const CONTEXT_CACHE_KEY = 'gtg_chat_context';
const GREETING_TEXT = 'Te ayudo a sacar lo mejor de ti. ¿Por dónde empezamos? ✨';

const todayKey = () => {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
};

function getUserProfile() {
  try {
    return JSON.parse(localStorage.getItem('user') || '{}');
  } catch (e) {
    return {};
  }
}

function getAuthToken() {
  return (
    (typeof window.getAccessToken === 'function' && window.getAccessToken()) ||
    localStorage.getItem('access') ||
    localStorage.getItem('token') ||
    ''
  );
}

function getLocalHour() {
  return new Date().getHours();
}

function getTimeGreeting(name) {
  const hour = getLocalHour();
  const safeName = name || 'hoy';
  if (hour >= 5 && hour <= 11) {
    return `Buenos días, ${safeName}.\nHoy comenzamos con energía limpia. Estoy aquí para ayudarte a ordenar tu mente, tu cuerpo y tus decisiones con coherencia.\n\n¿Desde dónde quieres empezar hoy?`;
  }
  if (hour >= 12 && hour <= 18) {
    return `Buenas tardes, ${safeName}.\nVamos a revisar cómo está tu sistema hoy y ajustar lo necesario con claridad.\n\n¿Qué te gustaría priorizar ahora?`;
  }
  return `Buenas noches, ${safeName}.\nEste es un buen momento para cerrar el día con calma y preparar mañana con intención.\n\n¿Quieres revisar tu progreso o preparar el día siguiente?`;
}

function getOnboardingDay(username) {
  const key = `${ONBOARDING_START_KEY}_${username || 'anon'}`;
  const start = localStorage.getItem(key);
  const today = todayKey();
  if (!start) {
    localStorage.setItem(key, today);
    return 1;
  }
  const startDate = new Date(`${start}T00:00:00`);
  const now = new Date();
  const diffDays = Math.floor((now - startDate) / (24 * 60 * 60 * 1000));
  return diffDays + 1;
}

function shouldShowOnboarding(username) {
  const day = getOnboardingDay(username);
  if (day > 3) return false;
  const lastKey = `${ONBOARDING_LAST_KEY}_${username || 'anon'}`;
  const lastShown = localStorage.getItem(lastKey);
  const today = todayKey();
  if (lastShown === today) return false;
  localStorage.setItem(lastKey, today);
  return true;
}

function getOnboardingMessage(day, hasDevice) {
  if (day === 1) {
    return `Soy tu Quantum Coach.\nMi función es acompañarte a ordenar tu energía, tu alimentación, tu entrenamiento y tus decisiones con coherencia.\n\nPuedes usarme cuando:\n– necesites claridad en una decisión,\n– quieras ajustar tu rutina de ejercicio,\n– revisar tu alimentación,\n– o simplemente entender mejor tu momento actual.\n\nNo te empujo. Te ayudo a ver con mayor claridad.\n¿Te gustaría empezar revisando tu estado actual?`;
  }
  if (day === 2) {
    if (!hasDevice) {
      return `Para trabajar con datos reales, puedes integrar tu dispositivo.\nAsí podré analizar tu energía, descanso y actividad con mayor precisión.\n\n¿Quieres conectarlo ahora o prefieres revisar tu plan primero?`;
    }
    return `Ya tengo tus datos de actividad recientes.\nPodemos usarlos para ajustar tu entrenamiento o revisar tu recuperación.\n\n¿Prefieres revisar tu rendimiento o tu descanso?`;
  }
  if (day === 3) {
    return `Los lunes realizamos una revisión profunda de tu sistema con el análisis QAF.\nSi en algún momento necesitas claridad estratégica o entender mejor tu estado general, puedes pedirme un análisis completo.\n\nEstoy aquí para ayudarte a ver más allá de lo evidente.\n¿Quieres que revisemos tu estado actual?`;
  }
  return '';
}

function getStableMessage() {
  return `Hoy podemos trabajar en tres frentes:\n– tu energía,\n– tu cuerpo,\n– o tu enfoque.\n\n¿Cuál quieres priorizar?`;
}

let speechRecognition = null;
let mediaRecorder = null;
let audioChunks = [];
let recordingTimeout = null;
let voiceCancelled = false;
let voiceHandled = false;
let activeVoiceMode = null;
let nativeSpeechPlugin = null;
let nativeSpeechListener = null;

const MAX_INPUT_HEIGHT = 120;

// --- Mobile keyboard / safe-area handling ---
let viewportRaf = 0;

function computeVisualViewportOffset() {
  const vv = window.visualViewport;
  if (!vv) return 0;
  const raw = window.innerHeight - vv.height - vv.offsetTop;
  return Math.max(0, Math.round(raw));
}

function applyViewportOffset() {
  viewportRaf = 0;
  if (!widgetContainer) return;
  const offset = computeVisualViewportOffset();
  widgetContainer.style.setProperty('--gtg-vv-offset', `${offset}px`);

  const vv = window.visualViewport;
  const height = vv ? Math.max(320, Math.round(vv.height)) : Math.max(320, Math.round(window.innerHeight));
  widgetContainer.style.setProperty('--gtg-vv-height', `${height}px`);

  scheduleFooterMetricsUpdate();
}

function scheduleViewportOffsetUpdate() {
  if (viewportRaf) return;
  viewportRaf = requestAnimationFrame(applyViewportOffset);
}

function autoResizeInput() {
  if (!input) return;
  input.style.height = 'auto';
  const nextHeight = Math.min(input.scrollHeight, MAX_INPUT_HEIGHT);
  input.style.height = `${nextHeight}px`;
  input.style.overflowY = input.scrollHeight > MAX_INPUT_HEIGHT ? 'auto' : 'hidden';

  scheduleFooterMetricsUpdate();
}

function resetInput() {
  if (!input) return;
  input.value = '';
  input.placeholder = 'Escribe tu duda...';
  autoResizeInput();
  if (window.innerWidth > 480) input.focus();
}

// --- Footer height sync (para que el input/preview/voz no tape el último mensaje) ---
let footerMetricsRaf = 0;

function computeFooterHeightPx() {
  if (!form) return 92;
  const rect = form.getBoundingClientRect();
  const height = Math.max(0, Math.round(rect.height));
  return height || 92;
}

function applyFooterHeightVar() {
  footerMetricsRaf = 0;
  if (!widgetContainer || !messages) return;

  const footerHeight = computeFooterHeightPx();
  widgetContainer.style.setProperty('--chat-footer-height', `${footerHeight}px`);

  // Mantener el último mensaje visible si el usuario estaba cerca del final.
  // (pero permitir suspenderlo durante relayout al abrir/cerrar)
  try {
    if (allowAutoScroll && shouldAutoScroll()) {
      messages.scrollTop = messages.scrollHeight;
    }
  } catch (e) {
    // ignore
  }
}

function scheduleFooterMetricsUpdate() {
  if (footerMetricsRaf) return;
  footerMetricsRaf = requestAnimationFrame(applyFooterHeightVar);
}

function formatFileSize(bytes) {
  if (typeof bytes !== 'number') return '';
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  const mb = kb / 1024;
  return `${mb.toFixed(1)} MB`;
}

let attachmentPreviewObjectUrl = '';

function revokeAttachmentPreviewUrl() {
  if (!attachmentPreviewObjectUrl) return;
  try {
    URL.revokeObjectURL(attachmentPreviewObjectUrl);
  } catch (e) {
    // ignore
  }
  attachmentPreviewObjectUrl = '';
}

function setAttachmentPreview(file, state = 'ready') {
  if (!attachmentPreview) return;
  if (!file) {
    revokeAttachmentPreviewUrl();
    attachmentPreview.hidden = true;
    attachmentPreview.innerHTML = '';
    scheduleFooterMetricsUpdate();
    return;
  }

  revokeAttachmentPreviewUrl();
  attachmentPreviewObjectUrl = URL.createObjectURL(file);

  const fileName = escapeHtml(file.name || 'adjunto');
  const size = formatFileSize(file.size);
  const mime = String(file.type || '').toLowerCase();
  const isImage = mime.startsWith('image/');
  const isPdf = mime === 'application/pdf' || String(file.name || '').toLowerCase().endsWith('.pdf');

  let previewHtml = '';
  if (isImage) {
    previewHtml = `<img class="attachment-thumb" src="${attachmentPreviewObjectUrl}" alt="Vista previa" />`;
  } else if (isPdf) {
    previewHtml = `
      <div class="attachment-thumb attachment-thumb-pdf" aria-hidden="true">
        <span class="attachment-pdf-badge">PDF</span>
      </div>
    `;
  } else {
    previewHtml = `
      <div class="attachment-thumb attachment-thumb-file" aria-hidden="true">
        <span class="attachment-file-badge">FILE</span>
      </div>
    `;
  }

  attachmentPreview.hidden = false;
  attachmentPreview.dataset.state = state;
  attachmentPreview.innerHTML = `
    ${previewHtml}
    <div class="attachment-meta">
      <span class="attachment-name">${fileName}</span>
      <span class="attachment-size">${size}</span>
    </div>
    <div class="attachment-actions">
      ${isPdf ? '<button type="button" class="attachment-open" title="Ver PDF">Ver</button>' : ''}
      <span class="attachment-state">${state === 'uploading' ? 'Subiendo...' : 'Adjunto listo'}</span>
      <button type="button" class="attachment-clear" title="Quitar">✕</button>
    </div>
  `;

  const openBtn = attachmentPreview.querySelector('.attachment-open');
  if (openBtn) {
    openBtn.disabled = state === 'uploading';
    openBtn.addEventListener('click', () => {
      if (!attachmentPreviewObjectUrl) return;
      try {
        window.open(attachmentPreviewObjectUrl, '_blank', 'noopener');
      } catch (e) {
        // ignore
      }
    });
  }

  const clearBtn = attachmentPreview.querySelector('.attachment-clear');
  if (clearBtn) {
    clearBtn.disabled = state === 'uploading';
    clearBtn.addEventListener('click', () => {
      clearSelectedFiles();
      setAttachmentPreview(null);
    });
  }

  scheduleFooterMetricsUpdate();
}

function shouldAutoScroll() {
  const threshold = 80;
  return messages.scrollHeight - messages.scrollTop - messages.clientHeight < threshold;
}

function updateScrollControls() {
  // Controles visuales de scroll removidos para una UX más limpia.
  // Mantenemos la función para no romper llamadas existentes.
  if (!messages) return;
}

function scrollMessagesByPages(direction) {
  // No-op: botones de scroll removidos.
  if (!messages) return;
}

function markMessageStatus(messageId, status) {
  const el = document.getElementById(messageId);
  if (!el) return;
  el.classList.remove('pending', 'failed');
  if (status === 'pending') el.classList.add('pending');
  if (status === 'failed') el.classList.add('failed');
}

function addRetryAction(messageId) {
  const el = document.getElementById(messageId);
  if (!el || el.querySelector('.retry-btn')) return;
  const text = el.dataset.text || '';
  const hasFile = el.dataset.hasFile === '1';
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'retry-btn';
  retry.textContent = 'Reintentar';
  retry.addEventListener('click', () => {
    if (hasFile) {
      appendMessage('Para reintentar un adjunto, vuelve a seleccionarlo.', 'bot');
      return;
    }
    sendQuickMessage(text);
  });
  el.appendChild(retry);
}

function canUseSpeechRecognition() {
  return !!(window.SpeechRecognition || window.webkitSpeechRecognition);
}

function initSpeechRecognition() {
  if (!canUseSpeechRecognition()) return null;
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = new Recognition();
  recognition.lang = 'es-ES';
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;
  return recognition;
}

function getNativeSpeechPlugin() {
  const cap = window.Capacitor;
  if (!cap || typeof cap.getPlatform !== 'function') return null;
  const platform = cap.getPlatform();
  if (platform !== 'ios' && platform !== 'android') return null;
  return cap.Plugins?.SpeechRecognition || null;
}

async function ensureNativeSpeechPermission(plugin) {
  if (!plugin?.hasPermission || !plugin?.requestPermission) return true;
  const current = await plugin.hasPermission();
  if (current?.permission === true) return true;
  const requested = await plugin.requestPermission();
  return requested?.permission === true;
}

function clearNativeSpeechListener() {
  if (nativeSpeechListener?.remove) {
    nativeSpeechListener.remove();
  }
  nativeSpeechListener = null;
}

function pickTranscript(matches) {
  if (!Array.isArray(matches)) return '';
  return String(matches[0] || '').trim();
}

function setRecordingState(isRecording) {
  if (!micBtn) return;
  micBtn.classList.toggle('recording', !!isRecording);
  micBtn.textContent = isRecording ? '■ Detener' : '🎤 Micrófono';

  scheduleFooterMetricsUpdate();
}

function setRetryVisible(visible) {
  // UX: ya no mostramos un botón fijo de reintento.
  // El usuario puede volver a abrir el menú (+) y tocar Micrófono nuevamente.
  return;
}

async function startMediaRecorder() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    appendMessage('Tu navegador no soporta grabación de voz.', 'bot');
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    voiceHandled = false;
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    mediaRecorder.onstop = async () => {
      const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
      stream.getTracks().forEach((track) => track.stop());
      setRecordingState(false);
      activeVoiceMode = null;
      if (voiceCancelled) {
        appendMessage('Grabación cancelada.', 'bot');
        return;
      }
      if (!audioChunks.length) {
        appendMessage('No se recibió audio. Intenta de nuevo.', 'bot');
        return;
      }
      const ok = await sendAudioForStt(blob);
      if (!ok) {
        // Mensaje ya mostrado en sendAudioForStt.
      }
    };
    mediaRecorder.start();
    setRecordingState(true);
    activeVoiceMode = 'media';
    recordingTimeout = setTimeout(() => {
      stopMediaRecorder();
    }, 12000);
  } catch (e) {
    appendMessage('No se pudo acceder al micrófono.', 'bot');
  }
}

function stopMediaRecorder() {
  if (recordingTimeout) {
    clearTimeout(recordingTimeout);
    recordingTimeout = null;
  }
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
}

async function stopNativeSpeech() {
  if (nativeSpeechPlugin?.stop) {
    try {
      await nativeSpeechPlugin.stop();
    } catch (e) {
      // ignore
    }
  }
  clearNativeSpeechListener();
  activeVoiceMode = null;
  setRecordingState(false);
}

async function sendAudioForStt(blob) {
  if (!blob || !window.API_URL) return;
  const token = getAuthToken();
  const formData = new FormData();
  formData.append('audio', blob, 'voice.webm');
  formData.append('language', 'es-ES');

  try {
    const res = await (window.authFetch || fetch)(`${API_URL}stt/`, {
      method: 'POST',
      headers: token ? { 'Authorization': `Bearer ${token}` } : undefined,
      body: formData
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.text) {
      appendMessage(data.error || 'No se pudo transcribir el audio.', 'bot');
      return false;
    }
    await sendQuickMessage(data.text);
    return true;
  } catch (e) {
    appendMessage('No se pudo transcribir el audio.', 'bot');
    return false;
  }
}

async function startVoiceCapture() {
  voiceCancelled = false;
  voiceHandled = false;
  setRetryVisible(false);
  if (await startNativeSpeech()) return;
  if (speechRecognition) {
    try {
      speechRecognition.start();
      setRecordingState(true);
      return;
    } catch (e) {
      // fall back to backend STT
    }
  }
  await startMediaRecorder();
}

async function startNativeSpeech() {
  const plugin = getNativeSpeechPlugin();
  if (!plugin) return false;
  nativeSpeechPlugin = plugin;
  try {
    const availability = await plugin.available?.();
    if (availability && availability.available === false) return false;
  } catch (e) {
    return false;
  }

  const permitted = await ensureNativeSpeechPermission(plugin);
  if (!permitted) {
    appendMessage('No se otorgaron permisos para el micrófono.', 'bot');
    return true;
  }

  activeVoiceMode = 'native';
  setRecordingState(true);
  let resolved = false;
  const handleMatches = async (matches) => {
    if (resolved || voiceCancelled) return;
    resolved = true;
    voiceHandled = true;
    setRecordingState(false);
    activeVoiceMode = null;
    clearNativeSpeechListener();
    const transcript = pickTranscript(matches);
    if (transcript) {
      await sendQuickMessage(transcript);
      return;
    }
    appendMessage('No pude escuchar nada claro. Intenta de nuevo.', 'bot');
  };

  if (plugin.addListener) {
    nativeSpeechListener = await plugin.addListener('result', (data) => {
      handleMatches(data?.matches || data?.results);
    });
  }

  try {
    const result = await plugin.start?.({
      language: 'es-ES',
      maxResults: 1,
      partialResults: false,
      popup: true
    });
    if (result?.matches?.length) {
      await handleMatches(result.matches);
    }
  } catch (e) {
    setRecordingState(false);
    activeVoiceMode = null;
    clearNativeSpeechListener();
    if (!voiceCancelled) {
      appendMessage('No se pudo iniciar el reconocimiento de voz.', 'bot');
    }
  }
  return true;
}

function getContextualPrompt(context) {
  if (!context) return null;
  const ifValue = context.if_snapshot?.latest_record?.value ?? context.profile?.happiness_index;
  const ifLow = typeof ifValue === 'number' && ifValue <= 6.5;
  const ifHigh = typeof ifValue === 'number' && ifValue >= 8;
  const hasPlan = Array.isArray(context.documents?.types) && context.documents.types.length > 0;
  const connectedProviders = context.devices?.connected_providers || [];
  const hasDevice = connectedProviders.length > 0;
  const fitnessProviders = context.devices?.fitness || {};
  const latestProvider = Object.values(fitnessProviders)[0];
  const metrics = latestProvider?.metrics || {};
  const hasActivity = typeof metrics.steps === 'number' && metrics.steps > 0;
  const sleepMissing = typeof metrics.sleep_minutes === 'number' && metrics.sleep_minutes === 0;
  const isMonday = new Date().getDay() === 1;

  if (ifLow) {
    return `Veo que tu energía está un poco más baja hoy.\nAntes de hacer ajustes, quiero entender cómo te sientes.\n\n¿Te gustaría que revisemos qué está generando esa carga?`;
  }
  if (ifHigh) {
    return `Tu sistema está bastante estable hoy.\nEste es un buen momento para consolidar hábitos y avanzar con intención.\n\n¿Quieres optimizar tu entrenamiento o mantener estabilidad?`;
  }
  if (sleepMissing && hasDevice) {
    return `Aún no tengo datos completos de descanso.\nSi quieres, podemos revisarlo manualmente o esperar la sincronización.\n\n¿Cómo descansaste realmente?`;
  }
  if (hasActivity) {
    return `Ya registré actividad reciente.\n¿Quieres ajustar tu entrenamiento según lo que hiciste hoy?`;
  }
  if (hasPlan) {
    return `Veo que tienes un plan de entrenamiento y nutrición cargado.\nPodemos usarlo como base para organizar tu semana.\n\n¿Quieres revisarlo juntos?`;
  }
  if (!hasDevice) {
    return `Para trabajar con datos reales, puedes integrar tu dispositivo.\nAsí podré analizar tu energía, descanso y actividad con mayor precisión.\n\n¿Quieres conectarlo ahora o prefieres revisar tu plan primero?`;
  }
  if (isMonday) {
    return `Hoy es un buen momento para un análisis profundo QAF.\n¿Quieres que revisemos tu estado completo?`;
  }
  return null;
}

function buildQuickActions(context) {
  const actions = [];
  const hasPlan = Array.isArray(context?.documents?.types) && context.documents.types.length > 0;
  const hasDevice = (context?.devices?.connected_providers || []).length > 0;

  if (hasPlan) {
    actions.push({ label: 'Revisar plan', type: 'link', href: '/pages/settings/PlanEntrenamiento.html' });
  }
  if (!hasDevice) {
    actions.push({ label: 'Sincronizar', type: 'link', href: '/pages/settings/Dispositivos.html' });
  }
  actions.push({ label: 'Postura', type: 'posture_start' });
  actions.push({ label: 'Análisis profundo', type: 'message', text: 'Quiero un análisis profundo QAF.' });
  return actions.slice(0, 3);
}

async function fetchCoachContext() {
  const user = getUserProfile();
  const username = user?.username || localStorage.getItem('username');
  const token = getAuthToken();

  // Sin username/token no podemos pedir contexto autenticado.
  if (!username || !token) return null;

  try {
    const authFetch = window.authFetch || fetch;
    const url = `${API_URL}coach_context/?username=${encodeURIComponent(username)}`;
    const res = await authFetch(url, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    });
    if (!res.ok) return null;
    const data = await res.json();
    localStorage.setItem(CONTEXT_CACHE_KEY, JSON.stringify(data));
    return data;
  } catch (e) {
    return null;
  }
}

function getCachedContext() {
  try {
    const raw = localStorage.getItem(CONTEXT_CACHE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    return null;
  }
}

const loadChatHistory = () => {
  try {
    const raw = localStorage.getItem(CHAT_STORAGE_KEY);
    const metaRaw = localStorage.getItem(CHAT_META_KEY);
    if (!raw || !metaRaw) return [];
    const meta = JSON.parse(metaRaw);
    const history = JSON.parse(raw);
    if (!meta || !meta.updatedAt || !meta.day) return [];
    const isExpired = Date.now() - Number(meta.updatedAt) > CHAT_TTL_MS;
    const isNewDay = meta.day !== todayKey();
    if (isExpired || isNewDay) {
      localStorage.removeItem(CHAT_STORAGE_KEY);
      localStorage.removeItem(CHAT_META_KEY);
      localStorage.removeItem('gtg_session_id');
      localStorage.removeItem('gtg_guest_session_id');
      localStorage.removeItem(SESSION_DAY_KEY);
      return [];
    }
    return Array.isArray(history) ? history : [];
  } catch (e) {
    return [];
  }
};

let chatHistory = loadChatHistory();

let allowAutoScroll = true;
let savedScrollState = null;

function getScrollState() {
  if (!messages) return null;
  const atBottom = (messages.scrollHeight - messages.scrollTop - messages.clientHeight) <= 10;
  return {
    top: messages.scrollTop,
    atBottom,
  };
}

function restoreScrollState(state) {
  if (!messages || !state) return;
  if (state.atBottom) {
    messages.scrollTop = messages.scrollHeight;
    return;
  }
  messages.scrollTop = state.top;
}

// --- Robust body scroll lock (mobile web + webview) ---
const bodyScrollLockState = {
  locked: false,
  scrollY: 0,
};

function lockBodyScroll() {
  if (bodyScrollLockState.locked) return;
  const y = window.scrollY || document.documentElement.scrollTop || 0;
  bodyScrollLockState.scrollY = y;
  bodyScrollLockState.locked = true;

  document.body.classList.add('gtg-chat-open');
  document.documentElement.style.overflow = 'hidden';
  document.body.style.overflow = 'hidden';
  document.body.style.overscrollBehavior = 'none';
  document.body.style.touchAction = 'none';
}

function unlockBodyScroll() {
  if (!bodyScrollLockState.locked) return;
  const y = bodyScrollLockState.scrollY || 0;
  bodyScrollLockState.locked = false;

  document.body.classList.remove('gtg-chat-open');
  document.documentElement.style.overflow = '';
  document.body.style.overflow = '';
  document.body.style.overscrollBehavior = '';
  document.body.style.touchAction = '';

  try {
    window.scrollTo(0, y);
  } catch (e) {
    // ignore
  }
}

function isToolsMenuOpen() {
  return !!(toolsMenu && !toolsMenu.hidden);
}

function closeToolsMenu() {
  if (!toolsMenu) return;
  toolsMenu.hidden = true;
  if (plusBtn) plusBtn.classList.remove('open');
}

function toggleToolsMenu() {
  if (!toolsMenu) return;
  const next = toolsMenu.hidden;
  toolsMenu.hidden = !next;
  if (plusBtn) plusBtn.classList.toggle('open', next);
  scheduleFooterMetricsUpdate();
}

function getSelectedFile() {
  return (fileInput && fileInput.files && fileInput.files[0]) || (cameraInput && cameraInput.files && cameraInput.files[0]) || null;
}

function clearSelectedFiles() {
  if (fileInput) fileInput.value = '';
  if (cameraInput) cameraInput.value = '';
}
// Toggle Function - Expanded Logic
function toggleChat() {
const containerWidget =
document.getElementById('chat-widget-container');
chatWindow.classList.toggle('open');
containerWidget.classList.toggle('expanded'); // Critical for Mobile // CSS
const isOpen = chatWindow.classList.contains('open');
containerWidget.classList.toggle('open', isOpen);

// Guardar/restaurar scroll para que al cerrar/abrir no parezca que se "pierden" mensajes/adjuntos.
if (!isOpen) {
  savedScrollState = getScrollState();
}

// En mobile web, bloquear scroll del body para que no se vea la pantalla de atrás.
try {
  if (window.matchMedia && window.matchMedia('(max-width: 768px)').matches) {
    if (isOpen) lockBodyScroll();
    else unlockBodyScroll();
  }
} catch (e) {
  // ignore
}

if (isOpen) {
if (window.innerWidth > 480) {
  try {
    input.focus({ preventScroll: true });
  } catch (e) {
    input.focus();
  }
}
// Recalcular visibilidad de los controles de scroll al abrir.
setTimeout(updateScrollControls, 50);
scheduleViewportOffsetUpdate();
allowAutoScroll = false;
scheduleFooterMetricsUpdate();

// Restaurar scroll después de que el layout se estabilice.
setTimeout(() => {
  restoreScrollState(savedScrollState);
  allowAutoScroll = true;
}, 80);
} else {
  closeToolsMenu();
  allowAutoScroll = true;
}
}
toggleBtn.addEventListener('click', toggleChat);
// Header controls
const closeBtn = document.getElementById('chat-close-btn');
if (closeBtn) closeBtn.addEventListener('click', toggleChat);

const resetBtn = document.getElementById('chat-reset-btn');
if (resetBtn) {
resetBtn.addEventListener('click', () => {
if (!confirm('¿Quieres reiniciar el chat?')) return;
localStorage.removeItem('gtg_session_id');
localStorage.removeItem('gtg_guest_session_id');
localStorage.removeItem(SESSION_DAY_KEY);
localStorage.removeItem(CHAT_STORAGE_KEY);
localStorage.removeItem(CHAT_META_KEY);
chatHistory = [];
messages.innerHTML = '';
appendMessage(GREETING_TEXT, 'bot');
});
}
if (micBtn) {
  speechRecognition = initSpeechRecognition();
  if (speechRecognition) {
    speechRecognition.onstart = () => {
      voiceHandled = false;
      setRecordingState(true);
      activeVoiceMode = 'web';
    };
    speechRecognition.onresult = async (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript || '';
      voiceHandled = true;
      setRecordingState(false);
      if (voiceCancelled) return;
      if (transcript.trim()) {
        await sendQuickMessage(transcript.trim());
      } else {
        appendMessage('No pude escuchar nada claro. Intenta de nuevo.', 'bot');
        setRetryVisible(true);
      }
    };
    speechRecognition.onerror = (event) => {
      setRecordingState(false);
      activeVoiceMode = null;
      if (voiceCancelled) return;
      if (event?.error === 'not-allowed' || event?.error === 'service-not-allowed') {
        appendMessage('No se otorgaron permisos para el micrófono.', 'bot');
        setRetryVisible(true);
        return;
      }
      startMediaRecorder();
    };
    speechRecognition.onend = () => {
      setRecordingState(false);
      activeVoiceMode = null;
      if (!voiceCancelled && !voiceHandled) {
        appendMessage('No se detectó voz. Puedes reintentar.', 'bot');
        setRetryVisible(true);
      }
    };
  }

  micBtn.addEventListener('click', async () => {
    if (micBtn.classList.contains('recording')) {
      if (activeVoiceMode === 'native') {
        await stopNativeSpeech();
      } else if (speechRecognition && activeVoiceMode === 'web') {
        speechRecognition.stop();
      } else {
        stopMediaRecorder();
      }
      return;
    }
    await startVoiceCapture();
  });
}

if (messages) {
  messages.addEventListener('scroll', () => {
    updateScrollControls();
  });
}

// Tools menu (+)
if (plusBtn) {
  plusBtn.addEventListener('click', () => {
    toggleToolsMenu();
  });
}

// Cerrar menú si se toca fuera
document.addEventListener('click', (event) => {
  if (!toolsMenu || toolsMenu.hidden) return;
  // Si estamos grabando, no cerrar el menú para permitir detener con el mismo botón.
  if (micBtn && micBtn.classList.contains('recording')) return;
  const target = event.target;
  if (!(target instanceof Element)) return;
  if (toolsMenu.contains(target) || plusBtn?.contains(target)) return;
  closeToolsMenu();
});

if (toolAttachBtn) {
  toolAttachBtn.addEventListener('click', () => {
    closeToolsMenu();
    fileInput?.click();
  });
}

if (toolCameraBtn) {
  toolCameraBtn.addEventListener('click', () => {
    closeToolsMenu();
    cameraInput?.click();
  });
}
// File Selection Feedback
if (fileInput) {
  fileInput.addEventListener('change', () => {
    const file = getSelectedFile();
    if (file && postureFlow.active && postureFlow.captureTarget) {
      const view = postureFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handlePostureCapture(file, view);
      return;
    }
    if (file) {
      setAttachmentPreview(file, 'ready');
      input.focus();
    } else {
      setAttachmentPreview(null);
    }
  });
}

if (cameraInput) {
  cameraInput.addEventListener('change', () => {
    const file = getSelectedFile();
    if (file && postureFlow.active && postureFlow.captureTarget) {
      const view = postureFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handlePostureCapture(file, view);
      return;
    }
    if (file) {
      setAttachmentPreview(file, 'ready');
      input.focus();
    } else {
      setAttachmentPreview(null);
    }
  });
}

// Mic desde el menú
if (toolMicBtn && toolMicBtn !== micBtn) {
  // no-op (micBtn ya apunta a toolMicBtn)
}
// Send Message
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const rawText = input.value || '';
  const text = rawText.trim();
  const file = getSelectedFile();
  if (!text && !file) return;
  let pendingId = null;
  if (file) {
    const objectUrl = URL.createObjectURL(file);
    pendingId = appendMessage(text || '', 'user pending', {
      meta: { text: text, hasFile: true },
      attachment: { file, objectUrl },
    });
  } else {
    pendingId = appendMessage(text, 'user pending', {
      meta: { text: text, hasFile: false }
    });
  }
  if (file) setAttachmentPreview(file, 'uploading');
  resetInput();
  clearSelectedFiles();
  await processMessage(text, file, pendingId);
});

input.addEventListener('input', autoResizeInput);
input.addEventListener('focus', () => {
  scheduleViewportOffsetUpdate();
  setTimeout(scheduleViewportOffsetUpdate, 250);
  scheduleFooterMetricsUpdate();
});
input.addEventListener('blur', () => {
  scheduleViewportOffsetUpdate();
  setTimeout(scheduleViewportOffsetUpdate, 250);
  scheduleFooterMetricsUpdate();
});
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    if (typeof form.requestSubmit === 'function') {
      form.requestSubmit();
    } else {
      form.dispatchEvent(new Event('submit', { cancelable: true }));
    }
  }
});
autoResizeInput();

// VisualViewport listeners (mobile keyboard)
if (window.visualViewport) {
  window.visualViewport.addEventListener('resize', scheduleViewportOffsetUpdate);
  window.visualViewport.addEventListener('scroll', scheduleViewportOffsetUpdate);
}
window.addEventListener('orientationchange', scheduleViewportOffsetUpdate);
window.addEventListener('resize', scheduleViewportOffsetUpdate);

// Initial compute
scheduleViewportOffsetUpdate();
scheduleFooterMetricsUpdate();

async function processMessage(text, file, pendingId, extraPayload = null) {
  const loadingId = appendMessage('Analizando...', 'bot loading', { persist: false });
  try {
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    const today = todayKey();
    const savedDay = localStorage.getItem(SESSION_DAY_KEY);
    if (savedDay !== today) {
      localStorage.removeItem('gtg_session_id');
      localStorage.removeItem('gtg_guest_session_id');
      localStorage.setItem(SESSION_DAY_KEY, today);
    }

    let sessionId = localStorage.getItem('gtg_session_id');
    if (!sessionId && user.username) {
      sessionId = `user_${user.username}_${today}`;
      localStorage.setItem('gtg_session_id', sessionId);
    }
    if (!sessionId) {
      sessionId = localStorage.getItem('gtg_guest_session_id');
      if (!sessionId) {
        sessionId = `guest_${today}_${crypto.randomUUID()}`;
        localStorage.setItem('gtg_guest_session_id', sessionId);
      }
    }

    const userData = JSON.parse(localStorage.getItem('user') || 'null');
    const username = userData?.username || localStorage.getItem('username') || null;
    const token = getAuthToken();

    let attachmentUrl = null;
    let attachmentText = null;
    let attachmentTextDiagnostic = '';
    if (file) {
      if (!username || !token) {
        document.getElementById(loadingId)?.remove();
        appendMessage('Para enviar adjuntos necesitas iniciar sesion.', 'bot');
        return;
      }
      const formData = new FormData();
      formData.append('username', username);
      formData.append('include_text', '1');
      formData.append('file', file);
      const uploadResp = await (window.authFetch || fetch)(API_URL + 'upload_chat_attachment/', {
        method: 'POST',
        headers: token ? { 'Authorization': `Bearer ${token}` } : undefined,
        body: formData
      });
      const uploadData = await uploadResp.json().catch(() => ({}));
      if (!uploadResp.ok) {
        const msg = uploadData.error || uploadData.detail || `Error subiendo archivo (HTTP ${uploadResp.status}).`;
        throw new Error(msg);
      }
      if (uploadData.success) {
        attachmentUrl = uploadData.file_url;
        attachmentText = uploadData.extracted_text;
        attachmentTextDiagnostic = uploadData.extracted_text_diagnostic || '';
      } else {
        throw new Error('Error subiendo archivo: ' + uploadData.error);
      }
    }

    const authFetch = window.authFetch || fetch;
    const response = await authFetch(API_URL + 'chat/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { 'Authorization': `Bearer ${token}` } : {})
      },
      body: JSON.stringify({
        message: text || "Analiza este documento adjunto.",
        sessionId: sessionId,
        attachment: attachmentUrl,
        attachment_text: attachmentText,
        attachment_text_diagnostic: attachmentTextDiagnostic,
        username: username,
        ...(extraPayload && typeof extraPayload === 'object' ? extraPayload : {})
      })
    });

    let data = {};
    const textResponse = await response.text();
    try {
      data = textResponse ? JSON.parse(textResponse) : {};
    } catch (e) {
      console.warn("Respuesta no JSON:", textResponse);
      data = { output: textResponse };
    }

    document.getElementById(loadingId)?.remove();
    if (pendingId) markMessageStatus(pendingId, 'sent');
    setAttachmentPreview(null);
    if (data.error) {
      appendMessage('❌ Error: ' + data.error, 'bot');
      return;
    }

    let reply = "No entendí eso.";
    if (data.output && data.output.trim() !== "") reply = data.output;
    else if (data.text) reply = data.text;
    else if (Array.isArray(data) && data[0] && data[0].text) reply = data[0].text;
    else if (typeof data === 'string') reply = data;

    if (typeof reply === "string" && reply.startsWith('<iframe')) {
      const match = reply.match(/srcdoc="([^"]*)"/);
      if (match && match[1]) {
        reply = match[1];
      }
    }
    appendMessage(reply, 'bot');

    // Quick actions (botones con estilo existente)
    try {
      if (data && typeof data === 'object' && Array.isArray(data.quick_actions) && data.quick_actions.length) {
        appendQuickActions(data.quick_actions);
      }
    } catch (e) {
      // ignore
    }

    // QAF: si el backend manda follow-ups, mostrarlos como botones (misma UX de quick-actions)
    try {
      if (data && typeof data === 'object') {
        if (data.qaf_context) lastQafContext = data.qaf_context;
        appendQafFollowUps(data.follow_up_questions, lastQafContext);
      }
    } catch (e) {
      // ignore
    }
  } catch (err) {
    console.error(err);
    document.getElementById(loadingId)?.remove();
    if (pendingId) {
      markMessageStatus(pendingId, 'failed');
      addRetryAction(pendingId);
    }
    setAttachmentPreview(null);
    appendMessage(`❌ Error: ${err.message} `, 'bot');
  }
}

function escapeHtml(input) {
  return String(input)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function buildUserAttachmentHtml(file, objectUrl) {
  if (!file || !objectUrl) return '';
  const name = escapeHtml(file.name || 'adjunto');
  const mime = String(file.type || '').toLowerCase();
  const isImage = mime.startsWith('image/');
  const isPdf = mime === 'application/pdf' || String(file.name || '').toLowerCase().endsWith('.pdf');

  if (isImage) {
    return `
      <div class="user-attachment">
        <img class="user-attachment-img" src="${objectUrl}" alt="${name}" loading="lazy" />
        <div class="user-attachment-caption">${name}</div>
      </div>
    `;
  }

  if (isPdf) {
    return `
      <div class="user-attachment user-attachment-pdf">
        <div class="user-attachment-pdf-badge">PDF</div>
        <div class="user-attachment-caption">${name}</div>
        <button type="button" class="user-attachment-open" data-url="${objectUrl}">Ver</button>
      </div>
    `;
  }

  return `
    <div class="user-attachment user-attachment-file">
      <div class="user-attachment-file-badge">FILE</div>
      <div class="user-attachment-caption">${name}</div>
    </div>
  `;
}

function sanitizeUrl(url) {
  const raw = String(url || '').trim();
  if (!raw) return '';
  const lower = raw.toLowerCase();
  if (lower.startsWith('javascript:') || lower.startsWith('data:') || lower.startsWith('vbscript:')) return '';
  if (lower.startsWith('http://') || lower.startsWith('https://') || lower.startsWith('mailto:')) return raw;
  if (raw.startsWith('/') || raw.startsWith('./') || raw.startsWith('../') || raw.startsWith('#')) return raw;
  return '';
}

function renderInlineMarkdown(escapedText) {
  let out = String(escapedText);

  // Inline code
  out = out.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Bold then italic
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/(^|[^*])\*([^*]+)\*(?!\*)/g, '$1<em>$2</em>');

  // Links: [text](url)
  out = out.replace(/\[([^\]]+?)\]\(([^\s)]+?)\)/g, (m, label, url) => {
    const safe = sanitizeUrl(url);
    if (!safe) return label;
    // `safe` ya viene desde texto escapado; evitar doble escape.
    return `<a href="${safe}" target="_blank" rel="noopener noreferrer">${label}</a>`;
  });

  return out;
}

function renderBotRichText(text) {
  // Render seguro: escapamos HTML de entrada y soportamos un subconjunto de Markdown
  const input = String(text ?? '');
  const normalized = input.replace(/\r\n?/g, '\n');

  // Code fences ```...```
  const parts = normalized.split('```');
  const html = [];
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    const isCode = i % 2 === 1;
    if (isCode) {
      const codeEscaped = escapeHtml(part.replace(/^\w+\n/, ''));
      html.push(`<pre><code>${codeEscaped}</code></pre>`);
      continue;
    }

    const lines = part.split('\n');
    let paragraph = [];
    let listType = null; // 'ul' | 'ol'
    const flushParagraph = () => {
      if (!paragraph.length) return;
      const joined = paragraph.join('<br>');
      html.push(`<p>${joined}</p>`);
      paragraph = [];
    };
    const closeList = () => {
      if (!listType) return;
      html.push(`</${listType}>`);
      listType = null;
    };

    for (const rawLine of lines) {
      const line = rawLine.trimEnd();
      const trimmed = line.trim();

      if (!trimmed) {
        flushParagraph();
        closeList();
        continue;
      }

      // Horizontal rule
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        flushParagraph();
        closeList();
        html.push('<hr>');
        continue;
      }

      // Headings (#, ##, ###)
      const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
      if (headingMatch) {
        const level = headingMatch[1].length;
        const content = renderInlineMarkdown(escapeHtml(headingMatch[2]));
        flushParagraph();
        closeList();
        html.push(`<h${level}>${content}</h${level}>`);
        continue;
      }

      // Lists
      const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
      const olMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (ulMatch || olMatch) {
        const nextType = ulMatch ? 'ul' : 'ol';
        const itemText = ulMatch ? ulMatch[1] : olMatch[1];
        const itemHtml = renderInlineMarkdown(escapeHtml(itemText));

        flushParagraph();
        if (listType && listType !== nextType) closeList();
        if (!listType) {
          listType = nextType;
          html.push(`<${listType}>`);
        }
        html.push(`<li>${itemHtml}</li>`);
        continue;
      }

      // Default: paragraph line
      closeList();
      paragraph.push(renderInlineMarkdown(escapeHtml(trimmed)));
    }

    flushParagraph();
    closeList();
  }

  return html.join('');
}

function persistHistory() {
localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(chatHistory));
localStorage.setItem(CHAT_META_KEY, JSON.stringify({
day: todayKey(),
updatedAt: Date.now()
}));
localStorage.setItem(SESSION_DAY_KEY, todayKey());
}

function appendMessage(text, className, opts = {}) {
const div = document.createElement('div');
div.className = `message ${className}`;
if (opts.meta) {
  div.dataset.text = opts.meta.text || '';
  div.dataset.hasFile = opts.meta.hasFile ? '1' : '0';
}
// Bot: mantener formato (subconjunto Markdown) sin depender de librerías externas.
// User: texto plano, salvo cuando agregamos adjuntos (HTML controlado por nosotros).
if (className.includes('bot')) {
  div.innerHTML = renderBotRichText(text);
} else {
  const safeText = escapeHtml(text || '');
  const attachmentHtml = opts.attachment && opts.attachment.file && opts.attachment.objectUrl
    ? buildUserAttachmentHtml(opts.attachment.file, opts.attachment.objectUrl)
    : '';
  if (attachmentHtml) {
    const hasText = !!safeText.trim();
    div.innerHTML = `${hasText ? `<div class="user-text">${safeText}</div>` : ''}${attachmentHtml}`;
  } else {
    div.textContent = text;
  }
}
const id = 'msg-' + Date.now();
div.id = id;
messages.appendChild(div);

// Delegación simple para abrir adjuntos PDF inline
if (!className.includes('bot')) {
  const openBtn = div.querySelector('.user-attachment-open');
  if (openBtn) {
    openBtn.addEventListener('click', () => {
      const url = openBtn.getAttribute('data-url') || '';
      if (!url) return;
      try {
        window.open(url, '_blank', 'noopener');
      } catch (e) {
        // ignore
      }
    });
  }
}
const doScroll = opts.forceScroll || shouldAutoScroll();
if (doScroll) messages.scrollTop = messages.scrollHeight;
updateScrollControls();
const persist = opts.persist !== false && !className.includes('loading');
if (persist) {
chatHistory.push({
role: className.includes('user') ? 'user' : 'bot',
text,
ts: Date.now()
});
persistHistory();
}
return id;
}

function appendQuickActions(actions) {
  if (!Array.isArray(actions) || !actions.length) return;
  const wrapper = document.createElement('div');
  wrapper.className = 'quick-actions';
  actions.forEach((action) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'quick-action-btn';
    btn.textContent = action.label;
    btn.addEventListener('click', () => {
      if (action.type === 'open_camera') {
        closeToolsMenu();
        cameraInput?.click();
        return;
      }
      if (action.type === 'open_attach') {
        closeToolsMenu();
        fileInput?.click();
        return;
      }
      if (action.type === 'posture_start') {
        startPostureFlow();
        return;
      }
      if (action.type === 'posture_cancel') {
        cancelPostureFlow();
        return;
      }
      if (action.type === 'posture_capture' && action.view) {
        postureFlow.active = true;
        postureFlow.captureTarget = action.view;
        postureFlow.step = action.view === 'front' ? 'need_front' : 'need_side';
        savePostureState();
        if (action.source === 'attach') {
          closeToolsMenu();
          fileInput?.click();
        } else {
          closeToolsMenu();
          cameraInput?.click();
        }
        return;
      }
      if (action.type === 'posture_safety') {
        const v = String(action.value || '').toLowerCase();
        postureFlow.userContext.injury_recent = v === 'yes';
        postureFlow.userContext.pain_neck = v === 'yes';
        postureFlow.userContext.pain_low_back = v === 'yes';
        postureFlow.step = 'ready';
        savePostureState();

        sendQuickMessage('Analizar postura', {
          posture_request: {
            poses: {
              front: postureFlow.poses.front,
              side: postureFlow.poses.side,
            },
            user_context: postureFlow.userContext,
            locale: 'es-CO',
          },
        });
        return;
      }
      if (action.type === 'link' && action.href) {
        window.location.href = action.href;
        return;
      }
      if (action.type === 'message' && action.text) {
        sendQuickMessage(action.text, action.payload || null);
        return;
      }
      if (action.type === 'qaf_confirm_portion' && action.payload) {
        // Mantener UX: el tap se refleja como un mensaje del usuario (label del botón)
        sendQuickMessage(action.label, action.payload);
      }
    });
    wrapper.appendChild(btn);
  });
  messages.appendChild(wrapper);

    // Quick actions (botones con estilo existente)
    try {
      if (data && typeof data === 'object' && Array.isArray(data.quick_actions) && data.quick_actions.length) {
        appendQuickActions(data.quick_actions);
      }
    } catch (e) {
      // ignore
    }
  messages.scrollTop = messages.scrollHeight;
  updateScrollControls();
}

let lastQafContext = null;

function appendQafFollowUps(followUps, qafContext) {
  if (!Array.isArray(followUps) || !followUps.length) return;
  const q = followUps[0];
  if (!q || q.type !== 'confirm_portion') return;
  const itemId = q.item_id;
  const options = Array.isArray(q.options) ? q.options : [];
  if (!itemId || !options.length) return;

  const actions = options
    .slice(0, 3)
    .map((opt) => {
      const grams = Number(opt.grams);
      const label = String(opt.label || '').trim();
      if (!label || !Number.isFinite(grams) || grams <= 0) return null;
      return {
        label,
        type: 'qaf_confirm_portion',
        payload: {
          confirmed_portions: [{ item_id: itemId, grams }],
          qaf_context: qafContext || null,
        },
      };
    })
    .filter(Boolean);

  if (actions.length) appendQuickActions(actions);
}

async function sendQuickMessage(text, extraPayload = null) {
  if (!text) return;
  appendMessage(text, 'user');
  await processMessage(text, null, null, extraPayload);
}

// Restore history or show greeting
(async () => {
  messages.innerHTML = '';
  if (chatHistory.length) {
    chatHistory.forEach((msg) => {
      appendMessage(msg.text, msg.role === 'user' ? 'user' : 'bot', { persist: false });
    });
    return;
  }

  const user = getUserProfile();
  const name = (user?.full_name || user?.username || '').trim().split(' ')[0] || 'hoy';
  const context = (await fetchCoachContext()) || getCachedContext();
  const hasDevice = (context?.devices?.connected_providers || []).length > 0;

  appendMessage(getTimeGreeting(name), 'bot');

  if (shouldShowOnboarding(user?.username || 'anon')) {
    const day = getOnboardingDay(user?.username || 'anon');
    const onboardingText = getOnboardingMessage(day, hasDevice);
    if (onboardingText) appendMessage(onboardingText, 'bot');
  } else {
    appendMessage(getStableMessage(), 'bot');
  }

  const contextual = getContextualPrompt(context);
  if (contextual) appendMessage(contextual, 'bot');

  appendQuickActions(buildQuickActions(context || {}));
})();
})();
