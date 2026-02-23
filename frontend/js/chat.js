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

  <input type="file" id="chat-file-input" hidden accept="image/*,.pdf" multiple>
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

// --- Exp-010 Medición muscular (pose-estimation en cliente; 1..4 vistas) ---
let muscleFlow = {
  active: false,
  step: 'idle',
  captureTarget: null, // view_id
  focus: null,
  poses: {
    front_relaxed: null,
    side_right_relaxed: null,
    back_relaxed: null,
    front_flex: null,
  },
};

let musclePoseEstimator = null;
let musclePoseEstimatorLoading = null;

const MUSCLE_STATE_KEY = 'gtg_muscle_flow_state_v0';

function saveMuscleState() {
  try {
    localStorage.setItem(MUSCLE_STATE_KEY, JSON.stringify(muscleFlow));
  } catch (e) {
    // ignore
  }
}

function loadMuscleState() {
  try {
    const raw = localStorage.getItem(MUSCLE_STATE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    if (!parsed.active) return;
    muscleFlow = {
      ...muscleFlow,
      ...parsed,
      poses: parsed.poses || muscleFlow.poses,
    };
  } catch (e) {
    // ignore
  }
}

loadMuscleState();

// --- Exp-011 Vitalidad de la Piel (1 foto; análisis en backend) ---
let skinFlow = {
  active: false,
  step: 'idle',
};

const SKIN_STATE_KEY = 'gtg_skin_flow_state_v0';

function saveSkinState() {
  try {
    localStorage.setItem(SKIN_STATE_KEY, JSON.stringify(skinFlow));
  } catch (e) {
    // ignore
  }
}

function loadSkinState() {
  try {
    const raw = localStorage.getItem(SKIN_STATE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    if (!parsed.active) return;
    skinFlow = {
      ...skinFlow,
      ...parsed,
      active: !!parsed.active,
      step: String(parsed.step || skinFlow.step),
    };
  } catch (e) {
    // ignore
  }
}

loadSkinState();

// --- Exp-012 Alta Costura Inteligente (pose-estimation en cliente; 1..2 vistas) ---
let shapeFlow = {
  active: false,
  step: 'idle',
  captureTarget: null, // view_id
  poses: {
    front_relaxed: null,
    side_right_relaxed: null,
  },
};

// --- Exp-013 Arquitectura Corporal (pose-estimation en cliente; 2 obligatorias + 1 opcional) ---
let ppFlow = {
  active: false,
  step: 'idle',
  captureTarget: null,
  poses: {
    front_relaxed: null,
    side_right_relaxed: null,
    back_relaxed: null,
  },
};

const SHAPE_STATE_KEY = 'gtg_shape_flow_state_v0';

const PP_STATE_KEY = 'gtg_posture_proportion_flow_state_v0';

function saveShapeState() {
  try {
    localStorage.setItem(SHAPE_STATE_KEY, JSON.stringify(shapeFlow));
  } catch (e) {
    // ignore
  }
}

function loadShapeState() {
  try {
    const raw = localStorage.getItem(SHAPE_STATE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    if (!parsed.active) return;
    shapeFlow = {
      ...shapeFlow,
      ...parsed,
      poses: parsed.poses || shapeFlow.poses,
    };
  } catch (e) {
    // ignore
  }
}

loadShapeState();

function savePpState() {
  try {
    localStorage.setItem(PP_STATE_KEY, JSON.stringify(ppFlow));
  } catch (e) {
    // ignore
  }
}

function loadPpState() {
  try {
    const raw = localStorage.getItem(PP_STATE_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    if (!parsed.active) return;
    ppFlow = {
      ...ppFlow,
      ...parsed,
      poses: parsed.poses || ppFlow.poses,
    };
  } catch (e) {
    // ignore
  }
}

loadPpState();

async function getMusclePoseEstimator() {
  // Reutilizamos el mismo estimador de postura para no descargar dos veces.
  if (posturePoseEstimator) return posturePoseEstimator;
  if (posturePoseEstimatorLoading) return posturePoseEstimatorLoading;
  // fallback: usa el getter existente.
  return await getPosturePoseEstimator();
}

function startMuscleFlow() {
  // Un solo flujo activo a la vez
  try { cancelPostureFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelShapeFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelPpFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelSkinFlow({ silent: true }); } catch (e) { /* ignore */ }

  muscleFlow = {
    active: true,
    step: 'need_front',
    captureTarget: null,
    focus: null,
    poses: {
      front_relaxed: null,
      side_right_relaxed: null,
      back_relaxed: null,
      front_flex: null,
    },
  };
  saveMuscleState();

  appendMessage(
    'Vamos a hacer una **Medición del progreso muscular** con fotos (comparación relativa, sin prometer cm exactos).\n\n' +
      'Para que el seguimiento sea **real** semana a semana, vamos a usar **fotos de referencia**:\n' +
      '- Recomendado: **frente relajado + perfil derecho**\n' +
      '- Mejor (más precisión): agrega **espalda + frente flex suave**\n\n' +
      'Importante:\n' +
      '• Misma luz, misma distancia, misma altura de cámara\n' +
      '• Si es selfie en espejo: temporizador y aléjate (2–3m)\n' +
      '• Si quieres **centralizar un músculo** (ej. bíceps o glúteos): perfecto, pero no recortes hombros/cadera/codos/rodillas\n\n' +
      'Empecemos con **frente relajado** (cuerpo completo, buena luz, cámara a 2–3m).',
    'bot'
  );

  appendQuickActions([
    { label: 'Tomar foto frente', type: 'muscle_capture', view: 'front_relaxed', source: 'camera' },
    { label: 'Adjuntar foto frente', type: 'muscle_capture', view: 'front_relaxed', source: 'attach' },
    { label: 'Cancelar', type: 'muscle_cancel' },
  ]);
}

function startSkinFlow() {
  // Un solo flujo activo a la vez
  try { cancelPostureFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelShapeFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelPpFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelMuscleFlow({ silent: true }); } catch (e) { /* ignore */ }

  skinFlow = { active: true, step: 'active' };
  saveSkinState();

  // Activa el modo en backend (y devuelve los CTAs de contexto/foto).
  sendQuickMessage('Vitalidad de la Piel', null);
}

function startShapeFlow() {
  // Un solo flujo activo a la vez
  try { cancelPostureFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelMuscleFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelPpFlow({ silent: true }); } catch (e) { /* ignore */ }

  shapeFlow = {
    active: true,
    step: 'need_front',
    captureTarget: null,
    poses: {
      front_relaxed: null,
      side_right_relaxed: null,
    },
  };
  saveShapeState();

  appendMessage(
    'Vamos a hacer **Alta Costura Inteligente** (arquitectura visual + verticalidad + presencia) con fotos (ratios ópticos, sin prometer cm reales).\n\n' +
      '- Mínimo: **1** foto (frente relajado)\n' +
      '- Mejor: agrega **1** foto (perfil derecho)\n\n' +
      'Empecemos con **frente relajado** (cuerpo completo, buena luz, cámara a 2–3m).',
    'bot'
  );
  appendQuickActions([
    { label: 'Tomar foto frente', type: 'shape_capture', view: 'front_relaxed', source: 'camera' },
    { label: 'Adjuntar foto frente', type: 'shape_capture', view: 'front_relaxed', source: 'attach' },
    { label: 'Cancelar', type: 'shape_cancel' },
  ]);
}

function startPpFlow() {
  // Un solo flujo activo a la vez
  try { cancelPostureFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelMuscleFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelShapeFlow({ silent: true }); } catch (e) { /* ignore */ }

  ppFlow = {
    active: true,
    step: 'need_front',
    captureTarget: null,
    poses: {
      front_relaxed: null,
      side_right_relaxed: null,
      back_relaxed: null,
    },
  };
  savePpState();

  appendMessage(
    'Vamos a hacer **Arquitectura Corporal** (experiencia premium) con fotos (proxies por keypoints, sin prometer cm reales).\n\n' +
      'Esto traduce tu alineación en decisiones simples que se sienten en la vida real: estabilidad, eficiencia y presencia.\n\n' +
      'Necesito **2** fotos obligatorias y 1 opcional (recomendado):\n' +
      '- Frente relajado (obligatoria)\n' +
      '- Perfil derecho (obligatoria)\n' +
      '- Espalda (opcional recomendado)\n\n' +
      'Empecemos con **frente relajado** (cuerpo completo, buena luz, cámara a 2–3m).',
    'bot'
  );
  appendQuickActions([
    { label: 'Tomar foto frente', type: 'pp_capture', view: 'front_relaxed', source: 'camera' },
    { label: 'Adjuntar foto frente', type: 'pp_capture', view: 'front_relaxed', source: 'attach' },
    { label: 'Cancelar', type: 'pp_cancel' },
  ]);
}

function cancelPpFlow(opts = {}) {
  const wasActive = !!ppFlow.active;
  ppFlow = { ...ppFlow, active: false, step: 'idle', captureTarget: null };
  try {
    localStorage.removeItem(PP_STATE_KEY);
  } catch (e) {
    // ignore
  }
  if (!opts.silent && wasActive) {
    appendMessage('Listo. Si quieres retomarlo, escribe: "Arquitectura Corporal".', 'bot');
  }
}

function _humanPpViewLabel(view) {
  if (view === 'front_relaxed') return 'Frente relajado';
  if (view === 'side_right_relaxed') return 'Perfil derecho';
  if (view === 'back_relaxed') return 'Espalda';
  return 'Foto';
}

function _nextPpOffer(view) {
  if (view === 'front_relaxed') return 'side_right_relaxed';
  if (view === 'side_right_relaxed') return 'back_relaxed';
  return null;
}

function cancelShapeFlow(opts = {}) {
  const wasActive = !!shapeFlow.active;
  shapeFlow = { ...shapeFlow, active: false, step: 'idle', captureTarget: null };
  try {
    localStorage.removeItem(SHAPE_STATE_KEY);
  } catch (e) {
    // ignore
  }
  if (!opts.silent && wasActive) {
    appendMessage('Listo. Si quieres retomarlo, escribe: "Alta Costura Inteligente".', 'bot');
  }
}

function _humanShapeViewLabel(view) {
  if (view === 'front_relaxed') return 'Frente relajado';
  if (view === 'side_right_relaxed') return 'Perfil derecho';
  return 'Foto';
}

function _nextShapeOffer(view) {
  if (view === 'front_relaxed') return 'side_right_relaxed';
  return null;
}

function cancelMuscleFlow(opts = {}) {
  const wasActive = !!muscleFlow.active;
  muscleFlow = { ...muscleFlow, active: false, step: 'idle', captureTarget: null };
  try {
    localStorage.removeItem(MUSCLE_STATE_KEY);
  } catch (e) {
    // ignore
  }
  if (!opts.silent && wasActive) {
    appendMessage('Listo. Si quieres retomarlo, escribe: "Medición del progreso muscular".', 'bot');
  }
}

function cancelSkinFlow(opts = {}) {
  const wasActive = !!skinFlow.active;
  skinFlow = { ...skinFlow, active: false, step: 'idle' };
  try {
    localStorage.removeItem(SKIN_STATE_KEY);
  } catch (e) {
    // ignore
  }
  if (!opts.silent && wasActive) {
    appendMessage('Listo. Si quieres retomarlo, escribe: "Vitalidad de la Piel".', 'bot');
  }
}

function _humanMuscleViewLabel(view) {
  if (view === 'front_relaxed') return 'Frente relajado';
  if (view === 'side_right_relaxed') return 'Perfil derecho';
  if (view === 'back_relaxed') return 'Espalda';
  if (view === 'front_flex') return 'Frente flex suave';
  return 'Foto';
}

function _nextMuscleOffer(view) {
  if (view === 'front_relaxed') return 'side_right_relaxed';
  if (view === 'side_right_relaxed') return 'back_relaxed';
  if (view === 'back_relaxed') return 'front_flex';
  return null;
}

function _computeTorsoDefinitionBeta(img, keypoints) {
  try {
    const kps = Array.isArray(keypoints) ? keypoints : [];
    const map = {};
    kps.forEach((kp) => {
      if (!kp || !kp.name) return;
      map[String(kp.name)] = kp;
    });

    const ls = map.left_shoulder;
    const rs = map.right_shoulder;
    const lh = map.left_hip;
    const rh = map.right_hip;
    if (!ls || !rs || !lh || !rh) return null;

    const w = Number(img.naturalWidth || img.width || 0);
    const h = Number(img.naturalHeight || img.height || 0);
    if (!w || !h) return null;

    // Bounding box torso
    const x1 = Math.max(0, Math.min(ls.x, rs.x, lh.x, rh.x) * w);
    const x2 = Math.min(w, Math.max(ls.x, rs.x, lh.x, rh.x) * w);
    const y1 = Math.max(0, Math.min(ls.y, rs.y, lh.y, rh.y) * h);
    const y2 = Math.min(h, Math.max(ls.y, rs.y, lh.y, rh.y) * h);

    const bw = Math.max(1, Math.floor(x2 - x1));
    const bh = Math.max(1, Math.floor(y2 - y1));
    if (bw < 60 || bh < 80) return null;

    // ROI abdomen: mitad inferior del torso, centrada.
    const roiX = Math.floor(x1 + bw * 0.15);
    const roiW = Math.floor(bw * 0.7);
    const roiY = Math.floor(y1 + bh * 0.45);
    const roiH = Math.floor(bh * 0.5);
    if (roiW < 40 || roiH < 40) return null;

    const canvas = document.createElement('canvas');
    canvas.width = roiW;
    canvas.height = roiH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(img, roiX, roiY, roiW, roiH, 0, 0, roiW, roiH);
    const imgData = ctx.getImageData(0, 0, roiW, roiH);
    const data = imgData.data;

    // Métrica simple: energía de gradiente + contraste, con guardrail de iluminación.
    let sum = 0;
    let sum2 = 0;
    let edges = 0;
    const n = roiW * roiH;
    if (!n) return null;

    // Precompute luminance
    const lum = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      lum[i] = y;
      sum += y;
      sum2 += y * y;
    }
    const mean = sum / n;
    const variance = Math.max(0, sum2 / n - mean * mean);
    const std = Math.sqrt(variance);

    // Demasiado oscuro o demasiado brillante = poco confiable
    if (mean < 35 || mean > 235) return { score: 0, mean, std, note: 'luz_no_fiable' };

    // Aproximación de gradiente (diferencias vecinales)
    const thresh = Math.max(10, std * 0.6);
    for (let y = 1; y < roiH - 1; y++) {
      for (let x = 1; x < roiW - 1; x++) {
        const idx = y * roiW + x;
        const gx = Math.abs(lum[idx + 1] - lum[idx - 1]);
        const gy = Math.abs(lum[idx + roiW] - lum[idx - roiW]);
        const gmag = gx + gy;
        if (gmag > thresh) edges += 1;
      }
    }
    const edgeDensity = edges / Math.max(1, (roiW - 2) * (roiH - 2));

    // Combinar: edgeDensity (0..~0.25) y std (contraste)
    const edgeScore = Math.max(0, Math.min(1, (edgeDensity - 0.02) / (0.14 - 0.02)));
    const contrastScore = Math.max(0, Math.min(1, (std - 12) / (38 - 12)));
    const score01 = 0.65 * edgeScore + 0.35 * contrastScore;
    const score = Math.round(score01 * 100);

    return { score, mean, std, edgeDensity };
  } catch (e) {
    return null;
  }
}

async function estimateMusclePoseFromImageFile(file, opts = {}) {
  // Igual que posture, pero dejamos la función separada por claridad.
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

    const estimator = await getMusclePoseEstimator();
    const res = await new Promise((resolve) => {
      estimator.onResults((results) => resolve(results || {}));
      estimator.send({ image: img });
    });
    const kps = mediapipeLandmarksToKeypoints(res.poseLandmarks);

    let appearance = null;
    try {
      if (opts && opts.withAppearance) {
        const td = _computeTorsoDefinitionBeta(img, kps);
        if (td && typeof td === 'object') {
          appearance = {
            torso_definition_beta: Number.isFinite(Number(td.score)) ? Number(td.score) : 0,
            mean_luma: td.mean,
            contrast_std: td.std,
            edge_density: td.edgeDensity,
            note: td.note || null,
          };
        }
      }
    } catch (e) {
      // ignore
    }

    return {
      keypoints: kps,
      image: { width: img.naturalWidth || img.width || 0, height: img.naturalHeight || img.height || 0 },
      appearance: appearance,
    };
  } finally {
    try {
      URL.revokeObjectURL(url);
    } catch (e) {
      // ignore
    }
  }
}

async function handleSkinCapture(file) {
  if (!_isImageFile(file)) {
    appendMessage('Para Vitalidad de la Piel necesito una imagen.', 'bot');
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  appendMessage('Foto rostro', 'user', {
    meta: { text: 'Foto rostro', hasFile: true },
    attachment: { file, objectUrl },
  });

  // Nota: processMessage ya muestra "Analizando...".
  await processMessage('Vitalidad de la Piel', file, null, null);
}

async function handleMuscleCapture(file, view) {
  if (!_isImageFile(file)) {
    appendMessage('Para medición muscular necesito una imagen.', 'bot');
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  appendMessage(_humanMuscleViewLabel(view), 'user', {
    meta: { text: _humanMuscleViewLabel(view), hasFile: true },
    attachment: { file, objectUrl },
  });

  appendMessage('Analizando (pose estimation local)...', 'bot');

  let pose;
  try {
    pose = await estimateMusclePoseFromImageFile(file, { withAppearance: true });
  } catch (e) {
    console.warn('Muscle pose estimation failed:', e);
    appendMessage('No pude detectar bien tu cuerpo. Repite con mejor luz y cuerpo completo.', 'bot');
    return;
  }

  muscleFlow.active = true;
  muscleFlow.poses[view] = pose;
  muscleFlow.captureTarget = null;
  saveMuscleState();

  const nextView = _nextMuscleOffer(view);
  if (nextView) {
    const label = _humanMuscleViewLabel(nextView).toLowerCase();
    appendMessage(`¿Quieres agregar **${label}** para mejorar la medición o analizar ya?`, 'bot');
    appendQuickActions([
      { label: `Tomar ${label}`, type: 'muscle_capture', view: nextView, source: 'camera' },
      { label: `Adjuntar ${label}`, type: 'muscle_capture', view: nextView, source: 'attach' },
      { label: 'Analizar ahora', type: 'muscle_analyze' },
      { label: 'Cancelar', type: 'muscle_cancel' },
    ]);
    return;
  }

  appendQuickActions([
    { label: 'Analizar ahora', type: 'muscle_analyze' },
    { label: 'Cancelar', type: 'muscle_cancel' },
  ]);
}

async function handleShapeCapture(file, view) {
  if (!_isImageFile(file)) {
    appendMessage('Para Alta Costura Inteligente necesito una imagen.', 'bot');
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  appendMessage(_humanShapeViewLabel(view), 'user', {
    meta: { text: _humanShapeViewLabel(view), hasFile: true },
    attachment: { file, objectUrl },
  });

  appendMessage('Analizando (calibración visual local)...', 'bot');

  let pose;
  try {
    // Reutilizamos el mismo pipeline que medición muscular.
    pose = await estimateMusclePoseFromImageFile(file, { withAppearance: false });
  } catch (e) {
    console.warn('Shape pose estimation failed:', e);
    appendMessage('No pude detectar bien tu cuerpo. Repite con mejor luz y cuerpo completo.', 'bot');
    return;
  }

  shapeFlow.active = true;
  shapeFlow.poses[view] = pose;
  shapeFlow.captureTarget = null;
  saveShapeState();

  const nextView = _nextShapeOffer(view);
  if (nextView) {
    const label = _humanShapeViewLabel(nextView).toLowerCase();
    appendMessage(`¿Quieres agregar **${label}** para una lectura más fina o analizamos ya?`, 'bot');
    appendQuickActions([
      { label: `Tomar ${label}`, type: 'shape_capture', view: nextView, source: 'camera' },
      { label: `Adjuntar ${label}`, type: 'shape_capture', view: nextView, source: 'attach' },
      { label: 'Analizar ahora', type: 'shape_analyze' },
      { label: 'Cancelar', type: 'shape_cancel' },
    ]);
    return;
  }

  appendQuickActions([
    { label: 'Analizar ahora', type: 'shape_analyze' },
    { label: 'Cancelar', type: 'shape_cancel' },
  ]);
}

async function handlePpCapture(file, view) {
  if (!_isImageFile(file)) {
    appendMessage('Para Arquitectura Corporal necesito una imagen.', 'bot');
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  appendMessage(_humanPpViewLabel(view), 'user', {
    meta: { text: _humanPpViewLabel(view), hasFile: true },
    attachment: { file, objectUrl },
  });

  appendMessage('Analizando (calibración postural local)...', 'bot');

  let pose;
  try {
    pose = await estimateMusclePoseFromImageFile(file);
  } catch (e) {
    console.warn('PP pose estimation failed:', e);
    appendMessage('No pude detectar bien tu cuerpo. Repite con mejor luz y cuerpo completo.', 'bot');
    return;
  }

  ppFlow.active = true;
  ppFlow.poses[view] = pose;
  ppFlow.captureTarget = null;
  savePpState();

  const nextView = _nextPpOffer(view);
  if (nextView) {
    const label = _humanPpViewLabel(nextView).toLowerCase();
    const isOptional = nextView === 'back_relaxed';
    appendMessage(
      isOptional
        ? `¿Quieres agregar **${label}** (opcional) para mejorar el análisis o analizamos ya?`
        : `Ahora necesito **${label}** para completar el análisis.`,
      'bot'
    );
    appendQuickActions([
      { label: `Tomar ${label}`, type: 'pp_capture', view: nextView, source: 'camera' },
      { label: `Adjuntar ${label}`, type: 'pp_capture', view: nextView, source: 'attach' },
      { label: 'Analizar ahora', type: 'pp_analyze' },
      { label: 'Cancelar', type: 'pp_cancel' },
    ]);
    return;
  }

  appendQuickActions([
    { label: 'Analizar ahora', type: 'pp_analyze' },
    { label: 'Cancelar', type: 'pp_cancel' },
  ]);
}

function sendPpAnalyze() {
  const poses = ppFlow.poses || {};
  const hasFront = poses.front_relaxed && Array.isArray(poses.front_relaxed.keypoints) && poses.front_relaxed.keypoints.length;
  const hasSide = poses.side_right_relaxed && Array.isArray(poses.side_right_relaxed.keypoints) && poses.side_right_relaxed.keypoints.length;
  const hasAny = !!(hasFront || hasSide);
  if (!hasAny) {
    appendMessage('Primero necesito al menos 1 foto (frente o perfil).', 'bot');
    return;
  }
  if (!(hasFront && hasSide)) {
    appendMessage('Haré un **cálculo parcial** con 1 foto. Para que sea más fiable, agrega también la otra vista (frente + perfil).', 'bot');
  }
  sendQuickMessage('Arquitectura Corporal', {
    posture_proportion_request: {
      poses: poses,
      locale: 'es-CO',
    },
  });
}

function sendShapeAnalyze() {
  const poses = shapeFlow.poses || {};
  const hasAny = Object.values(poses).some((p) => p && typeof p === 'object' && Array.isArray(p.keypoints) && p.keypoints.length);
  if (!hasAny) {
    appendMessage('Primero necesito al menos 1 foto (frente relajado).', 'bot');
    return;
  }
  sendQuickMessage('Alta Costura Inteligente', {
    shape_presence_request: {
      poses: poses,
      locale: 'es-CO',
    },
  });
}

function sendMuscleAnalyze() {
  const poses = muscleFlow.poses || {};
  const hasAny = Object.values(poses).some((p) => p && typeof p === 'object' && Array.isArray(p.keypoints) && p.keypoints.length);
  if (!hasAny) {
    appendMessage('Primero necesito al menos 1 foto (frente relajado).', 'bot');
    return;
  }

  const focus = (muscleFlow && muscleFlow.focus) ? String(muscleFlow.focus) : '';
  const focusLow = focus.trim().toLowerCase();
  const wantBiceps = focusLow === 'biceps' || focusLow === 'bíceps' || focusLow === 'bicep';
  const hasFlex = poses.front_flex && Array.isArray(poses.front_flex.keypoints) && poses.front_flex.keypoints.length;

  if (wantBiceps && !hasFlex) {
    appendMessage(
      'Para medir **bíceps** de verdad necesito la foto **Frente flex suave** (esa pose activa el indicador del brazo).\n\n' +
        '¿La tomamos y luego analizamos?',
      'bot'
    );
    appendQuickActions([
      { label: 'Tomar frente flex suave', type: 'muscle_capture', view: 'front_flex', source: 'camera' },
      { label: 'Adjuntar frente flex suave', type: 'muscle_capture', view: 'front_flex', source: 'attach' },
      { label: 'Analizar sin flex', type: 'muscle_analyze' },
      { label: 'Cancelar', type: 'muscle_cancel' },
    ]);
    return;
  }

  sendQuickMessage('Analizar ahora', {
    muscle_measure_request: {
      poses: poses,
      locale: 'es-CO',
      focus: focusLow || null,
    },
  });
}

function startPostureFlow() {
  // Un solo flujo activo a la vez
  try { cancelMuscleFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelShapeFlow({ silent: true }); } catch (e) { /* ignore */ }
  try { cancelPpFlow({ silent: true }); } catch (e) { /* ignore */ }

  postureFlow = {
    active: true,
    step: 'need_front',
    captureTarget: null,
    poses: { front: null, side: null },
    userContext: { pain_neck: false, pain_low_back: false, injury_recent: false, level: 'beginner' },
  };
  savePostureState();

  appendMessage(
    '**Corrección de postura**\n\n' +
      'En menos de 60 segundos te muestro qué está afectando tu alineación y te dejo una rutina simple para mejorar tu presencia y prevenir molestias.\n\n' +
      'Necesito 2 fotos:\n' +
      '📸 Frontal\n' +
      '📸 Perfil (lateral)\n\n' +
      'Para que el análisis sea preciso:\n\n' +
      '• Celular a la altura del pecho (2–3 m de distancia)\n' +
      '• Cuerpo completo (de pies a cabeza)\n' +
      '• Buena luz y fondo limpio\n' +
      '• Brazos relajados\n' +
      '• Si estás solo: usa temporizador o espejo sin taparte\n\n' +
      'Empecemos con la foto frontal.',
    'bot'
  );
  appendQuickActions([
    { label: 'Tomar foto frontal', type: 'posture_capture', view: 'front', source: 'camera' },
    { label: 'Adjuntar foto frontal', type: 'posture_capture', view: 'front', source: 'attach' },
    { label: 'Cancelar', type: 'posture_cancel' },
  ]);
}

function cancelPostureFlow(opts = {}) {
  const wasActive = !!postureFlow.active;
  postureFlow = { ...postureFlow, active: false, step: 'idle', captureTarget: null };
  try {
    localStorage.removeItem(POSTURE_STATE_KEY);
  } catch (e) {
    // ignore
  }
  if (!opts.silent && wasActive) {
    appendMessage('Listo.\nCuando quieras retomarlo, toca "Corrección de postura" o escribe: corrección de postura.', 'bot');
  }
}

function _postureQualityFront(pose) {
  const kps = (pose && pose.keypoints) || [];
  const needed = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'];
  const map = {};
  kps.forEach((kp) => {
    if (!kp || !kp.name) return;
    map[String(kp.name)] = kp;
  });
  let present = 0;
  let sum = 0;
  const missing = [];
  needed.forEach((n) => {
    const kp = map[n];
    if (kp && Number.isFinite(Number(kp.score))) {
      present += 1;
      sum += Number(kp.score);
    } else {
      missing.push(n);
    }
  });
  const ratio = present / Math.max(1, needed.length);
  const avg = present ? sum / present : 0;
  return { ratio, avg, ok: ratio >= 0.8 && avg >= 0.5, missing };
}

function _postureQualitySide(pose) {
  const kps = (pose && pose.keypoints) || [];
  const map = {};
  kps.forEach((kp) => {
    if (!kp || !kp.name) return;
    map[String(kp.name)] = kp;
  });
  const groups = [
    ['right_ear', 'left_ear'],
    ['right_shoulder', 'left_shoulder'],
    ['right_hip', 'left_hip'],
    ['right_knee', 'left_knee'],
    ['right_ankle', 'left_ankle'],
  ];
  let present = 0;
  let sum = 0;
  const missing = [];
  groups.forEach((g) => {
    const kp = map[g[0]] || map[g[1]];
    if (kp && Number.isFinite(Number(kp.score))) {
      present += 1;
      sum += Number(kp.score);
    } else {
      missing.push(g.join('|'));
    }
  });
  const ratio = present / Math.max(1, groups.length);
  const avg = present ? sum / present : 0;
  return { ratio, avg, ok: ratio >= 0.8 && avg >= 0.5, missing };
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
    13: 'left_elbow',
    14: 'right_elbow',
    15: 'left_wrist',
    16: 'right_wrist',
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

  appendMessage('Estamos analizando tu alineación corporal…', 'bot');

  let pose;
  try {
    pose = await estimatePoseFromImageFile(file);
  } catch (e) {
    console.warn('Pose estimation failed:', e);
    appendMessage('No pude detectar bien tu cuerpo en la imagen. Repite con mejor luz y el cuerpo completo visible.', 'bot');
    return;
  }

  // Feedback inmediato: si la calidad es baja, pedimos retomar sin avanzar (mejor UX).
  try {
    const q = view === 'front' ? _postureQualityFront(pose) : _postureQualitySide(pose);
    if (!q.ok) {
      const pct = Math.round((q.ratio || 0) * 100);
      appendMessage(
        `No logré verte con suficiente claridad (calidad aprox: ${pct}%).\n\n` +
          'Para que el resultado sea realmente útil:\n\n' +
          '• Aléjate un poco para que salga el cuerpo completo\n' +
          '• Más luz frontal\n' +
          '• Celular estable\n' +
          '• Evita recortes\n\n' +
          '¿La repetimos?',
        'bot'
      );
      const actions = [
        { label: view === 'front' ? 'Tomar foto frontal' : 'Tomar foto lateral', type: 'posture_capture', view, source: 'camera' },
        { label: view === 'front' ? 'Adjuntar foto frontal' : 'Adjuntar foto lateral', type: 'posture_capture', view, source: 'attach' },
      ];
      // Si falló la lateral pero ya tenemos frontal, ofrecer análisis parcial (caso real: no hay más fotos).
      try {
        const hasFront = !!(postureFlow && postureFlow.poses && postureFlow.poses.front);
        if (view === 'side' && hasFront) {
          actions.push({ label: 'Analizar parcial', type: 'posture_analyze_partial' });
        }
      } catch (e) {
        // ignore
      }
      actions.push({ label: 'Cancelar', type: 'posture_cancel' });
      appendQuickActions(actions);
      return;
    }
  } catch (e) {
    // si falla el check, seguimos normal
  }

  postureFlow.poses[view] = pose;
  postureFlow.captureTarget = null;
  savePostureState();

  if (view === 'front') {
    postureFlow.step = 'need_side';
    savePostureState();
    appendMessage(
      '✅ Foto frontal lista.\n\n' +
        'Ahora vamos con la vista lateral (perfil).\n' +
        'Con ambas vistas puedo darte un resultado más preciso y confiable.\n\n' +
        'Si prefieres, puedo hacer un análisis parcial con esta foto, pero será menos exacto.',
      'bot'
    );
    appendQuickActions([
      { label: 'Tomar foto lateral', type: 'posture_capture', view: 'side', source: 'camera' },
      { label: 'Adjuntar foto lateral', type: 'posture_capture', view: 'side', source: 'attach' },
      { label: 'Analizar parcial', type: 'posture_analyze_partial' },
      { label: 'Cancelar', type: 'posture_cancel' },
    ]);
    return;
  }

  postureFlow.step = 'need_safety';
  savePostureState();
  appendMessage(
    '✅ Listo. Ya tengo frontal + lateral.\n\n' +
      'Antes de recomendar ejercicios, cuido tu seguridad.\n\n' +
      '¿Tienes dolor agudo, hormigueo, adormecimiento o una lesión reciente?',
    'bot'
  );
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

function setAttachmentPreview(fileOrFiles, state = 'ready') {
  if (!attachmentPreview) return;
  const files = Array.isArray(fileOrFiles) ? fileOrFiles.filter(Boolean) : (fileOrFiles ? [fileOrFiles] : []);
  if (!files.length) {
    revokeAttachmentPreviewUrl();
    attachmentPreview.hidden = true;
    attachmentPreview.innerHTML = '';
    scheduleFooterMetricsUpdate();
    return;
  }

  const file = files[0];
  const extraCount = Math.max(0, files.length - 1);

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
    <div class="attachment-preview-thumbwrap">
      ${previewHtml}
      ${extraCount > 0 ? `<span class="attachment-multi-badge">+${extraCount}</span>` : ''}
    </div>
    <div class="attachment-meta">
      <span class="attachment-name">${escapeHtml(files.length > 1 ? `${files.length} imágenes seleccionadas` : (file.name || 'adjunto'))}</span>
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
  // Si el último mensaje del usuario pasó de pending -> sent, mostrar botón Editar.
  try { scheduleEditButtonUpdate(); } catch (e) { /* ignore */ }
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
  // Guardrail UX: si un flujo está activo, no mostrar el menú base.
  try {
    if ((postureFlow && postureFlow.active) || (muscleFlow && muscleFlow.active) || (shapeFlow && shapeFlow.active) || (ppFlow && ppFlow.active) || (skinFlow && skinFlow.active)) {
      return [];
    }
  } catch (e) {
    // ignore
  }

  const actions = [];
  const hasPlan = Array.isArray(context?.documents?.types) && context.documents.types.length > 0;
  const hasDevice = (context?.devices?.connected_providers || []).length > 0;

  // Mantener 3 botones máximo, pero sin perder los pilares.
  // 1) Contextual (uno solo)
  // - Si hay plan: "Revisar plan"
  // - Si no hay dispositivo: "Sincronizar"
  // - Si hay dispositivo: "Estado de hoy" (Exp-007)
  if (hasPlan) {
    actions.push({ label: 'Revisar plan', type: 'link', href: '/pages/settings/PlanEntrenamiento.html' });
  } else if (!hasDevice) {
    actions.push({ label: 'Sincronizar', type: 'link', href: '/pages/settings/Dispositivos.html' });
  } else {
    actions.push({
      label: 'Estado de hoy',
      type: 'message',
      text: 'Estado de hoy',
      payload: { lifestyle_request: { days: 14 } },
    });
  }

  // 2) Vitalidad de la Piel (Exp-011)
  actions.push({ label: 'Alta Costura Inteligente', type: 'shape_start' });

  // 3) Vitalidad de la Piel (Exp-011)
  actions.push({ label: 'Vitalidad de la Piel', type: 'skin_start' });

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

function getSelectedFiles() {
  const fromFile = fileInput && fileInput.files ? Array.from(fileInput.files) : [];
  const fromCam = cameraInput && cameraInput.files ? Array.from(cameraInput.files) : [];
  const files = (fromFile.length ? fromFile : fromCam) || [];
  return files.filter(Boolean);
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
    const files = getSelectedFiles();
    const file = files[0] || null;

    // Guardrail: si el flujo muscular está activo, NO mandar imágenes al backend por el canal genérico.
    // En vez de eso, asignar automáticamente cada foto al siguiente ángulo faltante.
    if (files && files.length && muscleFlow && muscleFlow.active && !muscleFlow.captureTarget) {
      const ordered = ['front_relaxed', 'side_right_relaxed', 'back_relaxed', 'front_flex'];
      const poses = (muscleFlow.poses && typeof muscleFlow.poses === 'object') ? muscleFlow.poses : {};
      const missing = ordered.filter((v) => !(poses[v] && Array.isArray(poses[v].keypoints) && poses[v].keypoints.length));

      setAttachmentPreview(null);
      clearSelectedFiles();

      (async () => {
        if (files.length > 1) {
          appendMessage('Recibí varias fotos. Para que la medición sea más clara, vamos **una por una**.', 'bot');
        }
        const f = files[0];
        const view = missing[0] || 'front_relaxed';
        await handleMuscleCapture(f, view);
      })();
      return;
    }

    // Guardrail: si Vitalidad de la Piel está activo, mandar 1 foto al backend y analizar.
    if (file && skinFlow && skinFlow.active) {
      setAttachmentPreview(null);
      clearSelectedFiles();
      handleSkinCapture(file);
      return;
    }

    if (file && ppFlow.active && ppFlow.captureTarget) {
      const view = ppFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handlePpCapture(file, view);
      return;
    }
    if (file && shapeFlow.active && shapeFlow.captureTarget) {
      const view = shapeFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handleShapeCapture(file, view);
      return;
    }
    if (file && muscleFlow.active && muscleFlow.captureTarget) {
      const view = muscleFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handleMuscleCapture(file, view);
      return;
    }
    if (file && postureFlow.active && postureFlow.captureTarget) {
      const view = postureFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handlePostureCapture(file, view);
      return;
    }
    if (files.length > 1) {
      setAttachmentPreview(files.slice(0, 4), 'ready');
      input.focus();
    } else if (file) {
      setAttachmentPreview(file, 'ready');
      input.focus();
    } else {
      setAttachmentPreview(null);
    }
  });
}

if (cameraInput) {
  cameraInput.addEventListener('change', () => {
    const files = getSelectedFiles();
    const file = files[0] || null;

    // Guardrail: igual que attach, si muscleFlow está activo, procesar localmente.
    if (files && files.length && muscleFlow && muscleFlow.active && !muscleFlow.captureTarget) {
      const ordered = ['front_relaxed', 'side_right_relaxed', 'back_relaxed', 'front_flex'];
      const poses = (muscleFlow.poses && typeof muscleFlow.poses === 'object') ? muscleFlow.poses : {};
      const missing = ordered.filter((v) => !(poses[v] && Array.isArray(poses[v].keypoints) && poses[v].keypoints.length));

      setAttachmentPreview(null);
      clearSelectedFiles();

      (async () => {
        if (files.length > 1) {
          appendMessage('Recibí varias fotos. Para que la medición sea más clara, vamos **una por una**.', 'bot');
        }
        const f = files[0];
        const view = missing[0] || 'front_relaxed';
        await handleMuscleCapture(f, view);
      })();
      return;
    }

    // Guardrail: igual que attach, si Vitalidad de la Piel está activo, mandar 1 foto al backend y analizar.
    if (file && skinFlow && skinFlow.active) {
      setAttachmentPreview(null);
      clearSelectedFiles();
      handleSkinCapture(file);
      return;
    }

    if (file && ppFlow.active && ppFlow.captureTarget) {
      const view = ppFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handlePpCapture(file, view);
      return;
    }
    if (file && shapeFlow.active && shapeFlow.captureTarget) {
      const view = shapeFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handleShapeCapture(file, view);
      return;
    }
    if (file && muscleFlow.active && muscleFlow.captureTarget) {
      const view = muscleFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handleMuscleCapture(file, view);
      return;
    }
    if (file && postureFlow.active && postureFlow.captureTarget) {
      const view = postureFlow.captureTarget;
      setAttachmentPreview(null);
      clearSelectedFiles();
      handlePostureCapture(file, view);
      return;
    }
    if (files.length > 1) {
      setAttachmentPreview(files.slice(0, 4), 'ready');
      input.focus();
    } else if (file) {
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
  const filesRaw = getSelectedFiles();
  const files = (filesRaw || []).filter((f) => _isImageFile(f) || String(f?.name || '').toLowerCase().endsWith('.pdf'));
  const limited = files.slice(0, 4);

  if (!text && !limited.length) return;

  // --- Editar último mensaje del usuario (real) ---
  // Si hay target de edición y NO hay adjuntos, actualizamos el mensaje y regeneramos respuesta.
  const editTargetId = (() => {
    try { return String(input.dataset.editTargetId || '').trim(); } catch (e2) { return ''; }
  })();
  if (editTargetId && !limited.length) {
    const target = document.getElementById(editTargetId);
    if (target && _isEditableUserMessage(target)) {
      // 1) Actualizar burbuja
      try {
        target.dataset.text = text;
      } catch (ex) {
        // ignore
      }
      const userTextEl = target.querySelector('.user-text');
      if (userTextEl) userTextEl.textContent = text;

      // 2) Recortar DOM: borrar todo lo que esté después (respuestas/botones previos)
      try {
        let node = target.nextSibling;
        while (node) {
          const next = node.nextSibling;
          try { node.remove(); } catch (ex2) { /* ignore */ }
          node = next;
        }
      } catch (ex) {
        // ignore
      }

      // 3) Recortar historial persistido
      try {
        const targetTs = (target.dataset && target.dataset.ts) ? Number(target.dataset.ts) : NaN;
        const idx = Number.isFinite(targetTs)
          ? chatHistory.findIndex((m) => m && m.role === 'user' && Number(m.ts) === targetTs)
          : -1;
        if (idx >= 0) {
          chatHistory[idx] = { ...chatHistory[idx], text };
          chatHistory = chatHistory.slice(0, idx + 1);
          persistHistory();
        }
      } catch (ex) {
        // ignore
      }

      // 4) Limpiar estado de edición + reenviar
      try {
        delete input.dataset.editTargetId;
        delete input.dataset.editTargetTs;
      } catch (ex) {
        // ignore
      }

      // Marcar como pendiente mientras procesamos
      try {
        target.classList.add('pending');
      } catch (ex) {
        // ignore
      }

      resetInput();
      clearSelectedFiles();
      closeToolsMenu();

      // Regenerar respuesta para el mensaje corregido
      await processMessage(text, null, editTargetId);
      return;
    } else {
      // Si el target ya no es válido, limpiamos estado y seguimos como envío normal
      try {
        delete input.dataset.editTargetId;
        delete input.dataset.editTargetTs;
      } catch (ex) {
        // ignore
      }
    }
  }

  if (files.length > 4) {
    appendMessage('Puedo adjuntar **hasta 4** imágenes a la vez. Enviaré las primeras 4.', 'bot');
  }

  const queue = [];
  if (limited.length) {
    limited.forEach((f, idx) => {
      const objectUrl = URL.createObjectURL(f);
      const pid = appendMessage(idx === 0 ? (text || '') : '', 'user pending', {
        meta: { text: idx === 0 ? text : '', hasFile: true },
        attachment: { file: f, objectUrl },
      });
      queue.push({ file: f, pendingId: pid, text: idx === 0 ? text : '' });
    });
    setAttachmentPreview(limited, 'uploading');
  } else {
    const pid = appendMessage(text, 'user pending', { meta: { text: text, hasFile: false } });
    queue.push({ file: null, pendingId: pid, text });
  }

  resetInput();
  clearSelectedFiles();

  for (let i = 0; i < queue.length; i++) {
    const row = queue[i];
    await processMessage(row.text, row.file, row.pendingId);
  }
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
        // Guardrail UX: si un flujo está activo, no mezclar botones de otros módulos.
        try {
          const actionsIn = data.quick_actions || [];

          if (postureFlow && postureFlow.active) {
            const filtered = actionsIn.filter((a) => {
              if (!a || typeof a !== 'object') return false;
              const t = String(a.type || '').trim();
              return t.startsWith('posture_');
            });
            if (filtered.length) appendQuickActions(filtered);
            return;
          }

          if (muscleFlow && muscleFlow.active) {
            const filtered = actionsIn.filter((a) => {
              if (!a || typeof a !== 'object') return false;
              const t = String(a.type || '').trim();
              if (t.startsWith('muscle_')) return true;
              // Permitir CTAs de enfoque que mandan payload muscle_measure_request
              if (t === 'message' && a.payload && typeof a.payload === 'object' && a.payload.muscle_measure_request) return true;
              return false;
            });
            if (filtered.length) appendQuickActions(filtered);
            return;
          }

          if (skinFlow && skinFlow.active) {
            const filtered = actionsIn.filter((a) => {
              if (!a || typeof a !== 'object') return false;
              const t = String(a.type || '').trim();
              if (t === 'open_camera' || t === 'open_attach') return true;
              if (t === 'skin_cancel') return true;
              if (t === 'message' && a.payload && typeof a.payload === 'object') {
                if (a.payload.skin_context_prompt) return true;
                if (a.payload.skin_context_update) return true;
                if (a.payload.skin_habits_request) return true;
              }
              return false;
            });
            if (filtered.length) appendQuickActions(filtered);
            return;
          }

          appendQuickActions(actionsIn);
        } catch (e) {
          appendQuickActions(data.quick_actions);
        }
      }
    } catch (e) {
      // ignore
    }

    // QAF: si el backend manda follow-ups, mostrarlos como botones (misma UX de quick-actions)
    try {
      if (data && typeof data === 'object') {
        if (data.qaf_context) lastQafContext = data.qaf_context;
        if (!(skinFlow && skinFlow.active)) {
          appendQafFollowUps(data.follow_up_questions, lastQafContext);
        }
      }
    } catch (e) {
      // ignore
    }

    // Cierre limpio: solo finalizar cuando el backend marque el stage final.
    try {
      if (skinFlow && skinFlow.active && data && typeof data === 'object') {
        const stage = String(data.skin_flow_stage || '').trim().toLowerCase();
        if (stage === 'completed') {
          skinFlow.active = false;
          skinFlow.step = 'idle';
          try { localStorage.removeItem(SKIN_STATE_KEY); } catch (e) { /* ignore */ }
          clearQuickActions();
        }
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
const isUser = className.includes('user');
const hasAttachment = !!(opts.attachment && opts.attachment.file);
// Guardar siempre el texto original del usuario para retry/editar (sin contaminar con botones).
if (isUser) {
  const raw = (opts.meta && typeof opts.meta.text === 'string') ? opts.meta.text : (text || '');
  div.dataset.text = raw;
  div.dataset.hasFile = (opts.meta && opts.meta.hasFile) || hasAttachment ? '1' : '0';
} else if (opts.meta) {
  div.dataset.text = opts.meta.text || '';
  div.dataset.hasFile = opts.meta.hasFile ? '1' : '0';
}
if (opts.ts) {
  try { div.dataset.ts = String(opts.ts); } catch (e) { /* ignore */ }
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
    // Render consistente para poder editar sin romper botones/adjuntos.
    div.innerHTML = `<div class="user-text">${safeText}</div>`;
  }
}
const nowTs = Date.now();
const id = 'msg-' + nowTs;
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
  const recordTs = (opts.ts && Number.isFinite(Number(opts.ts))) ? Number(opts.ts) : nowTs;
  try { div.dataset.ts = String(recordTs); } catch (e) { /* ignore */ }
  chatHistory.push({
    role: className.includes('user') ? 'user' : 'bot',
    text,
    ts: recordTs
  });
persistHistory();
}

// UX secundaria: mantener el botón “Editar” en el último mensaje del usuario.
try { scheduleEditButtonUpdate(); } catch (e) { /* ignore */ }
return id;
}

// --- UX: Editar último mensaje del usuario (copiar al input) ---
function _getUserMessagePlainText(el) {
  if (!el) return '';
  const fromData = (el.dataset && typeof el.dataset.text === 'string') ? el.dataset.text : '';
  if (fromData && fromData.trim()) return fromData;
  const userText = el.querySelector && el.querySelector('.user-text');
  if (userText && typeof userText.textContent === 'string') return userText.textContent;
  return '';
}

function _isEditableUserMessage(el) {
  if (!el) return false;
  if (!(el instanceof Element)) return false;
  if (!el.classList.contains('message') || !el.classList.contains('user')) return false;
  if (el.classList.contains('pending') || el.classList.contains('failed')) return false;
  if ((el.dataset && el.dataset.hasFile) === '1') return false;
  // Si tiene adjunto inline, no editamos (reintentar adjunto ya tiene su flujo)
  if (el.querySelector && el.querySelector('.user-attachment')) return false;
  const txt = _getUserMessagePlainText(el);
  return !!(txt && txt.trim());
}

function updateLastUserEditButton() {
  try {
    if (!messages) return;
    // Limpiar cualquier botón previo
    messages.querySelectorAll('.edit-last-btn').forEach((b) => {
      try { b.remove(); } catch (e) { /* ignore */ }
    });

    const userMsgs = messages.querySelectorAll('.message.user');
    if (!userMsgs || !userMsgs.length) return;

    let target = null;
    for (let i = userMsgs.length - 1; i >= 0; i--) {
      const el = userMsgs[i];
      if (_isEditableUserMessage(el)) {
        target = el;
        break;
      }
    }
    if (!target) return;

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'edit-last-btn';
    btn.textContent = 'Editar';
    btn.addEventListener('click', (ev) => {
      try { ev.preventDefault(); } catch (e) { /* ignore */ }
      try { ev.stopPropagation(); } catch (e) { /* ignore */ }

      const txt = _getUserMessagePlainText(target);
      if (!txt || !txt.trim()) return;
      input.value = txt;
      try {
        input.dataset.editTargetId = String(target.id || '');
        input.dataset.editTargetTs = String((target.dataset && target.dataset.ts) ? target.dataset.ts : '');
      } catch (e) {
        // ignore
      }
      try {
        input.focus({ preventScroll: true });
      } catch (e) {
        try { input.focus(); } catch (e2) { /* ignore */ }
      }
      try { autoResizeInput(); } catch (e) { /* ignore */ }
      try {
        const end = input.value.length;
        input.setSelectionRange(end, end);
      } catch (e) {
        // ignore
      }
      scheduleFooterMetricsUpdate();
      scheduleViewportOffsetUpdate();
    });

    target.appendChild(btn);
  } catch (e) {
    // Nunca romper el chat por UX secundaria.
  }
}

let _editBtnUpdateTimer = null;
function scheduleEditButtonUpdate() {
  if (_editBtnUpdateTimer) return;
  _editBtnUpdateTimer = setTimeout(() => {
    _editBtnUpdateTimer = null;
    updateLastUserEditButton();
  }, 0);
}

function clearQuickActions() {
  try {
    document.querySelectorAll('.quick-actions').forEach((el) => {
      try { el.remove(); } catch (e) { /* ignore */ }
    });
  } catch (e) {
    // ignore
  }
}

function appendQuickActions(actions) {
  if (!Array.isArray(actions) || !actions.length) return;

  // UX: no acumular bloques viejos de quick-actions (evita duplicados visuales)
  try {
    document.querySelectorAll('.quick-actions').forEach((el) => {
      try { el.remove(); } catch (e) { /* ignore */ }
    });
  } catch (e) {
    // ignore
  }

  // Deduplicar acciones (backend a veces reinyecta CTAs)
  const seen = new Set();
  const normalized = [];
  for (const a of actions) {
    if (!a || typeof a !== 'object') continue;
    const t = String(a.type || '');
    const v = String(a.view || '');
    const l = String(a.label || '');
    const txt = String(a.text || '');
    let payloadKey = '';
    try {
      payloadKey = a.payload ? JSON.stringify(a.payload) : '';
    } catch (e) {
      payloadKey = '';
    }
    const key = `${t}|${v}|${l}|${txt}|${payloadKey}`;
    if (seen.has(key)) continue;
    seen.add(key);
    normalized.push(a);
  }
  actions = normalized;
  if (!actions.length) return;

  const wrapper = document.createElement('div');
  wrapper.className = 'quick-actions';
  actions.forEach((action) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'quick-action-btn';
    btn.textContent = action.label || action.text || 'Acción';
    btn.addEventListener('click', () => {
      // Evitar doble tap (especialmente en mobile)
      if (btn.disabled) return;

      // Anti-loop: deshabilitar TODO el bloque de quick-actions y removerlo.
      try {
        wrapper.querySelectorAll('button').forEach((b) => {
          try { b.disabled = true; } catch (e) { /* ignore */ }
        });
      } catch (e) {
        // ignore
      }
      setTimeout(() => {
        try { wrapper.remove(); } catch (e) { /* ignore */ }
      }, 50);

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
      if (action.type === 'skin_start') {
        startSkinFlow();
        return;
      }
      if (action.type === 'muscle_start') {
        startMuscleFlow();
        return;
      }
      if (action.type === 'shape_start') {
        startShapeFlow();
        return;
      }
      if (action.type === 'pp_start') {
        startPpFlow();
        return;
      }
      if (action.type === 'posture_cancel') {
        cancelPostureFlow();
        return;
      }
      if (action.type === 'skin_cancel') {
        cancelSkinFlow();
        clearQuickActions();
        // Notificar al backend para cerrar el modo y evitar re-enganche accidental.
        try {
          sendQuickMessage('Cancelar', { skin_cancel: true });
        } catch (e) {
          // ignore
        }
        return;
      }
      if (action.type === 'muscle_cancel') {
        cancelMuscleFlow();
        return;
      }
      if (action.type === 'shape_cancel') {
        cancelShapeFlow();
        return;
      }
      if (action.type === 'pp_cancel') {
        cancelPpFlow();
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
      if (action.type === 'muscle_capture' && action.view) {
        muscleFlow.active = true;
        muscleFlow.captureTarget = action.view;
        muscleFlow.step = String(action.view || 'need_front');
        saveMuscleState();
        if (action.source === 'attach') {
          closeToolsMenu();
          fileInput?.click();
        } else {
          closeToolsMenu();
          cameraInput?.click();
        }
        return;
      }
      if (action.type === 'shape_capture' && action.view) {
        shapeFlow.active = true;
        shapeFlow.captureTarget = action.view;
        shapeFlow.step = String(action.view || 'need_front');
        saveShapeState();
        if (action.source === 'attach') {
          closeToolsMenu();
          fileInput?.click();
        } else {
          closeToolsMenu();
          cameraInput?.click();
        }
        return;
      }
      if (action.type === 'pp_capture' && action.view) {
        ppFlow.active = true;
        ppFlow.captureTarget = action.view;
        ppFlow.step = String(action.view || 'need_front');
        savePpState();
        if (action.source === 'attach') {
          closeToolsMenu();
          fileInput?.click();
        } else {
          closeToolsMenu();
          cameraInput?.click();
        }
        return;
      }
      if (action.type === 'muscle_analyze') {
        sendMuscleAnalyze();
        return;
      }
      if (action.type === 'shape_analyze') {
        sendShapeAnalyze();
        return;
      }
      if (action.type === 'pp_analyze') {
        sendPpAnalyze();
        return;
      }
      if (action.type === 'posture_analyze_partial') {
        const front = postureFlow?.poses?.front || null;
        const side = postureFlow?.poses?.side || null;
        if (!front && !side) {
          appendMessage('Primero necesito al menos 1 foto (frontal o lateral).', 'bot');
          return;
        }
        appendMessage('Haré un **análisis parcial** con 1 foto. Para mayor precisión, agrega también la segunda vista (frontal + lateral).', 'bot');
        sendQuickMessage('Analizar postura (parcial)', {
          posture_request: {
            poses: {
              front: front,
              side: side,
            },
            user_context: postureFlow.userContext,
            locale: 'es-CO',
          },
        });
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
        // Si es un CTA de enfoque muscular, persistirlo en el flujo.
        try {
          const fx = action.payload && action.payload.muscle_measure_request && action.payload.muscle_measure_request.focus;
          if (fx && muscleFlow && muscleFlow.active) {
            muscleFlow.focus = String(fx);
            saveMuscleState();
          }
        } catch (e) {
          // ignore
        }
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

  messages.scrollTop = messages.scrollHeight;
  updateScrollControls();
}

let lastQafContext = null;

function appendQafFollowUps(followUps, qafContext) {
  try {
    if (skinFlow && skinFlow.active) return;
  } catch (e) {
    // ignore
  }
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
  try {
    const low = String(text || '').toLowerCase();
    if ((low.includes('vitalidad') && (low.includes('piel') || low.includes('peil'))) || low.includes('skin health') || low.includes('skincare')) {
      skinFlow = { active: true, step: 'active' };
      saveSkinState();
    }
  } catch (e) {
    // ignore
  }
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
    try { scheduleEditButtonUpdate(); } catch (e) { /* ignore */ }
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

  try { scheduleEditButtonUpdate(); } catch (e) { /* ignore */ }
})();
})();
