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
link.href = "/css/chat.css"; // Ruta absoluta desde raíz del servidor // frontend
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
<form id="chat-input-area" style="display: flex; align-items:
center;">
<div class="chat-input-tools">
<label for="chat-file-input" style="cursor: pointer; padding:
10px; color: #666;">
<svg viewBox="0 0 24 24" width="24" height="24"
fill="currentColor"><path d="M16.5 6v11.5c0 2.21-1.79 4-4
4s-4-1.79-4-4V5a2.5 2.5 0 0 1 5 0v10.5c0 .55-.45 1-1
1s-1-.45-1-1V6H10v9.5a2.5 2.5 0 0 0 5
0V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5
5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>
</label>
<input type="file" id="chat-file-input" style="display: none;" accept="image/*,.pdf">
<button type="button" id="chat-mic-btn" title="Voz">🎤</button>
<button type="button" id="chat-voice-cancel" class="chat-voice-action" hidden>Cancelar</button>
<button type="button" id="chat-voice-retry" class="chat-voice-action" hidden>Reintentar</button>
</div>
<input type="text" id="chat-input" placeholder="Escribe tu
duda..." autocomplete="off" style="flex:1;">
<button type="submit" id="chat-send-btn">
<svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2
10l15 2-15 2z"/></svg>
</button>
</form>
</div>
`;
document.body.appendChild(container);
// 3. Lógica
const toggleBtn = document.getElementById('chat-toggle-btn');
const chatWindow = document.getElementById('chat-window');
const form = document.getElementById('chat-input-area');
const input = document.getElementById('chat-input');
const fileInput = document.getElementById('chat-file-input');
const micBtn = document.getElementById('chat-mic-btn');
const micCancelBtn = document.getElementById('chat-voice-cancel');
const micRetryBtn = document.getElementById('chat-voice-retry');
const messages = document.getElementById('chat-messages');

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
  micBtn.textContent = isRecording ? '■' : '🎤';
  if (micCancelBtn) micCancelBtn.hidden = !isRecording;
  if (isRecording && micRetryBtn) micRetryBtn.hidden = true;
}

function setRetryVisible(visible) {
  if (!micRetryBtn) return;
  micRetryBtn.hidden = !visible;
}

async function startMediaRecorder() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    appendMessage('Tu navegador no soporta grabación de voz.', 'bot');
    setRetryVisible(true);
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
        setRetryVisible(true);
        return;
      }
      const ok = await sendAudioForStt(blob);
      if (!ok) setRetryVisible(true);
    };
    mediaRecorder.start();
    setRecordingState(true);
    activeVoiceMode = 'media';
    recordingTimeout = setTimeout(() => {
      stopMediaRecorder();
    }, 12000);
  } catch (e) {
    appendMessage('No se pudo acceder al micrófono.', 'bot');
    setRetryVisible(true);
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
    setRetryVisible(true);
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
    setRetryVisible(true);
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
      setRetryVisible(true);
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
  actions.push({ label: 'Análisis profundo', type: 'message', text: 'Quiero un análisis profundo QAF.' });
  return actions.slice(0, 3);
}

async function fetchCoachContext() {
  const user = getUserProfile();
  const username = user?.username || localStorage.getItem('username');
  if (!username || !window.API_URL) return null;
  const token = getAuthToken();
  if (!token) return null;

  try {
    const res = await (window.authFetch || fetch)(
      `${API_URL}coach_context/?include_text=0&username=${encodeURIComponent(username)}`,
      {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      }
    );
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
// Toggle Function - Expanded Logic
function toggleChat() {
const containerWidget =
document.getElementById('chat-widget-container');
chatWindow.classList.toggle('open');
containerWidget.classList.toggle('expanded'); // Critical for Mobile // CSS
if (chatWindow.classList.contains('open')) {
if (window.innerWidth > 480) input.focus();
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

if (micCancelBtn) {
  micCancelBtn.addEventListener('click', () => {
    voiceCancelled = true;
    if (activeVoiceMode === 'native') {
      stopNativeSpeech();
    } else if (speechRecognition && activeVoiceMode === 'web') {
      speechRecognition.abort();
    } else {
      stopMediaRecorder();
    }
    setRecordingState(false);
    appendMessage('Grabación cancelada.', 'bot');
  });
}

if (micRetryBtn) {
  micRetryBtn.addEventListener('click', async () => {
    if (micBtn?.classList.contains('recording')) return;
    await startVoiceCapture();
  });
}
// File Selection Feedback
fileInput.addEventListener('change', () => {
if (fileInput.files.length > 0) {
input.placeholder = `📎 ${fileInput.files[0].name} (Adjunto)`;
input.focus();
} else {
input.placeholder = "Escribe tu duda...";
}
});
// Send Message
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  const file = fileInput.files[0];
  if (!text && !file) return;
  if (file) appendMessage(`📎 Subiendo: ${file.name}...`, 'user');
  if (text) appendMessage(text, 'user');
  input.value = '';
  fileInput.value = '';
  input.placeholder = "Escribe tu duda...";
  await processMessage(text, file);
});

async function processMessage(text, file) {
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
    if (file) {
      if (!username || !token) {
        document.getElementById(loadingId)?.remove();
        appendMessage('Para enviar adjuntos necesitas iniciar sesion.', 'bot');
        return;
      }
      const formData = new FormData();
      formData.append('username', username);
      formData.append('file', file);
      const uploadResp = await (window.authFetch || fetch)(API_URL + 'upload_chat_attachment/', {
        method: 'POST',
        headers: token ? { 'Authorization': `Bearer ${token}` } : undefined,
        body: formData
      });
      const uploadData = await uploadResp.json();
      if (uploadData.success) {
        attachmentUrl = uploadData.file_url;
        attachmentText = uploadData.extracted_text;
      } else {
        throw new Error('Error subiendo archivo: ' + uploadData.error);
      }
    }

    const authFetch = window.authFetch || fetch;
    const response = await authFetch(API_URL + 'chat/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text || "Analiza este documento adjunto.",
        sessionId: sessionId,
        attachment: attachmentUrl,
        attachment_text: attachmentText,
        username: username
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
  } catch (err) {
    console.error(err);
    document.getElementById(loadingId)?.remove();
    appendMessage(`❌ Error: ${err.message} `, 'bot');
  }
}
// Inject Marked.js for Markdown parsing
if (!window.marked) {
const script = document.createElement('script');
script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
script.onload = () => { console.log("Marked.js loaded"); };
document.head.appendChild(script);
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
// Use Markdown if available, otherwise fallback to plain text
if (window.marked && className.includes('bot')) {
div.innerHTML = window.marked.parse(text);
} else {
div.textContent = text;
}
const id = 'msg-' + Date.now();
div.id = id;
messages.appendChild(div);
messages.scrollTop = messages.scrollHeight;
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
      if (action.type === 'link' && action.href) {
        window.location.href = action.href;
        return;
      }
      if (action.type === 'message' && action.text) {
        sendQuickMessage(action.text);
      }
    });
    wrapper.appendChild(btn);
  });
  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}

async function sendQuickMessage(text) {
  if (!text) return;
  appendMessage(text, 'user');
  await processMessage(text, null);
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
