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
const messages = document.getElementById('chat-messages');

// --- Chat persistence (24h + reset diario) ---
const CHAT_STORAGE_KEY = 'gtg_chat_history';
const CHAT_META_KEY = 'gtg_chat_meta';
const SESSION_DAY_KEY = 'gtg_session_day';
const CHAT_TTL_MS = 24 * 60 * 60 * 1000;
const GREETING_TEXT = 'Te ayudo a sacar lo mejor de ti. ¿Por dónde empezamos? ✨';

const todayKey = () => {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
};

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
// Visual Feedback for User
if (file) appendMessage(`📎 Subiendo: ${file.name}...`, 'user');
if (text) appendMessage(text, 'user');
input.value = '';
fileInput.value = ''; // Reset file
input.placeholder = "Escribe tu duda...";
// Loading State
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
// 1⃣ Intentar usar sessionId definido por backend (preferido)
let sessionId = localStorage.getItem('gtg_session_id');
// 2⃣ Fallback: usuario logueado pero sin sessionId guardado aún
if (!sessionId && user.username) {
sessionId = `user_${user.username}_${today}`;
localStorage.setItem('gtg_session_id', sessionId);
}
// 3⃣ Fallback final: guest persistente (NO usar "invitado")
if (!sessionId) {
sessionId = localStorage.getItem('gtg_guest_session_id');
if (!sessionId) {
sessionId = `guest_${today}_${crypto.randomUUID()}`;
localStorage.setItem('gtg_guest_session_id', sessionId);
}
}
let attachmentUrl = null;
let attachmentText = null;
// 1. Upload File if exists
if (file) {
const formData = new FormData();
formData.append('username', sessionId);
formData.append('file', file);
const uploadResp = await fetch(API_URL + 'upload_medical/', {
method: 'POST',
body: formData
});
const uploadData = await uploadResp.json();
if (uploadData.success) {
attachmentUrl = uploadData.file_url;
attachmentText = uploadData.extracted_text; // Recibimos el texto
} else {
throw new Error('Error subiendo archivo: ' +
uploadData.error);
}
}
// 2. Logica Chat (enviando URL del adjunto si existe)
const userData = JSON.parse(localStorage.getItem('user') || 'null');
const username = userData?.username || localStorage.getItem('username') || null;

const authFetch = window.authFetch || fetch;
const response = await authFetch(API_URL + 'chat/', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({
message: text || "Analiza este documento adjunto.",
sessionId: sessionId,
attachment: attachmentUrl,
attachment_text: attachmentText, // Enviamos texto al proxy
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
// Remove loading
    document.getElementById(loadingId)?.remove();
if (data.error) {
    appendMessage('❌ Error: ' + data.error, 'bot');
} else {
// n8n suele devolver { output: "texto" } o una lista
let reply = "No entendí eso.";
// Prioridad de campos comunes de n8n
if (data.output && data.output.trim() !== "") reply =
data.output;
else if (data.text) reply = data.text;
else if (Array.isArray(data) && data[0] && data[0].text) reply
= data[0].text;
else if (typeof data === 'string') reply = data;
// FIX: Si n8n devuelve un iframe (común en modo chat), extraemos el texto
    if (typeof reply === "string" && reply.startsWith('<iframe')) {
const match = reply.match(/srcdoc="([^"]*)"/);
if (match && match[1]) {
reply = match[1];
}
}
appendMessage(reply, 'bot');
}
} catch (err) {
console.error(err);
if (document.getElementById(loadingId))
document.getElementById(loadingId).remove();
appendMessage(`❌ Error: ${err.message} `, 'bot');
}
});
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

// Restore history or show greeting
(() => {
messages.innerHTML = '';
if (chatHistory.length) {
chatHistory.forEach((msg) => {
appendMessage(msg.text, msg.role === 'user' ? 'user' : 'bot', { persist: false });
});
} else {
appendMessage(GREETING_TEXT, 'bot');
}
})();
})();
