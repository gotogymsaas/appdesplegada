
// config.js — fuente única de verdad (compatible con código antiguo y nuevo)
(function () {
  const host = window.location.hostname;
  const DEFAULT_PROD_API = "https://api.gotogym.store/api/";
  const DEFAULT_MEDIA_PUBLIC_BASE = "https://gotogymweb3755.blob.core.windows.net/media/";
  const DEFAULT_AVATAR_URL = "https://gotogymweb3755.blob.core.windows.net/media/Logo%20Fondo%20negro.png";
  const LOCAL_API = "http://127.0.0.1:8000/api/";
  // En LAN queremos pegarle al backend en el mismo host que sirve el frontend.
  // Esto permite abrir el frontend desde el celular (misma Wi-Fi) sin hardcodear una IP.
  const LAN_API = `http://${host}:8000/api/`;

  const isNative = () =>
    window.Capacitor && typeof window.Capacitor.isNativePlatform === "function"
      ? window.Capacitor.isNativePlatform()
      : false;

  const DEBUG =
    window.GTG_DEBUG === true || localStorage.getItem("gtg_debug") === "1";

  const overrideApi =
    window.GTG_API_URL || localStorage.getItem("api_url_override");

  // 1) Define base URL
  let baseApi = DEFAULT_PROD_API;

  if (overrideApi) {
    baseApi = overrideApi;
  } else if (isNative()) {
    baseApi = DEFAULT_PROD_API;
  } else if (host === "127.0.0.1" || host === "localhost") {
    baseApi = LOCAL_API;
  } else if (
    host.startsWith("192.168.") ||
    host.startsWith("10.") ||
    /^172\.(1[6-9]|2\d|3[0-1])\./.test(host)
  ) {
    baseApi = LAN_API;
  } else {
    baseApi = DEFAULT_PROD_API;
  }

  // 2) Compatibilidad total: window.API_URL y variable global API_URL
  window.API_URL = baseApi;
  // "API_URL" global (para código viejo que hace fetch(API_URL + ...))
  window.API_URL_ALIAS = baseApi; // opcional, por claridad
  // Esto crea API_URL en el scope global del browser
  if (typeof window.API_URL !== "undefined") {
    // eslint-disable-next-line no-undef
    API_URL = window.API_URL; // crea global si no existe
  }

  if (DEBUG) {
    console.log("⚙️ API_URL =", window.API_URL);
  }

  const mediaBaseOverride =
    window.GTG_MEDIA_PUBLIC_BASE || localStorage.getItem("media_public_base_override");
  const MEDIA_PUBLIC_BASE = (mediaBaseOverride || DEFAULT_MEDIA_PUBLIC_BASE).endsWith("/")
    ? (mediaBaseOverride || DEFAULT_MEDIA_PUBLIC_BASE)
    : `${mediaBaseOverride || DEFAULT_MEDIA_PUBLIC_BASE}/`;

  function resolveMediaUrl(value) {
    if (!value || typeof value !== "string") return "";
    if (/^https?:\/\//i.test(value)) return value;
    if (value.startsWith("blob:") || value.startsWith("data:")) return value;

    let cleaned = value.trim();
    if (!cleaned) return "";

    cleaned = cleaned.replace(/^\/+/, "");
    cleaned = cleaned.replace(/^media\//i, "");

    return encodeURI(`${MEDIA_PUBLIC_BASE}${cleaned}`);
  }

  // --- AUTH HELPERS ---
  const ACCESS_KEY = "access";
  const REFRESH_KEY = "refresh";
  const TOKEN_KEY = "token"; // compat legado

  function getAccessToken() {
    return (
      localStorage.getItem(ACCESS_KEY) ||
      localStorage.getItem(TOKEN_KEY) ||
      null
    );
  }

  function getRefreshToken() {
    return (
      localStorage.getItem(REFRESH_KEY) ||
      localStorage.getItem("refresh_token") ||
      null
    );
  }

  function setAccessToken(token) {
    if (!token) return;
    localStorage.setItem(ACCESS_KEY, token);
    localStorage.setItem(TOKEN_KEY, token); // compat legado
  }

  function setRefreshToken(token) {
    if (!token) return;
    localStorage.setItem(REFRESH_KEY, token);
  }

  async function refreshAccessToken() {
    const refresh = getRefreshToken();
    if (!refresh) return null;

    try {
      const res = await fetch(`${window.API_URL}token/refresh/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh }),
      });

      if (!res.ok) return null;
      const data = await res.json();
      if (data && data.access) {
        setAccessToken(data.access);
        return data.access;
      }
      return null;
    } catch (err) {
      console.warn("refreshAccessToken failed:", err);
      return null;
    }
  }

  async function authFetch(input, init = {}) {
    const url = typeof input === "string" ? input : input.url;
    const opts = { ...init };
    opts.headers = { ...(init.headers || {}) };

    // Agregar Authorization si existe token y no viene definido
    const token = getAccessToken();
    if (token && !opts.headers.Authorization && !opts.headers.authorization) {
      opts.headers.Authorization = `Bearer ${token}`;
    }

    let res = await fetch(input, opts);

    // Evita loop en refresh/login
    const isAuthEndpoint =
      (url && url.includes("/token/refresh/")) ||
      (url && url.includes("/login/"));

    if (res.status !== 401 || isAuthEndpoint) return res;

    // Intentar refresh una sola vez
    const newToken = await refreshAccessToken();
    if (!newToken) {
      try {
        localStorage.removeItem(ACCESS_KEY);
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(REFRESH_KEY);
        localStorage.removeItem("refresh_token");
        localStorage.removeItem("isLoggedIn");
      } catch (_) {}

      try {
        const path = (window.location && window.location.pathname) ? window.location.pathname : "";
        if (path.includes('/pages/admin/')) {
          window.location.href = '/pages/auth/indexInicioDeSesion.html?reason=session_expired';
        }
      } catch (_) {}

      return res;
    }

    const retryOpts = {
      ...opts,
      headers: { ...opts.headers, Authorization: `Bearer ${newToken}` },
    };

    return fetch(input, retryOpts);
  }

  // Exponer helpers
  window.getAccessToken = getAccessToken;
  window.getRefreshToken = getRefreshToken;
  window.setAccessToken = setAccessToken;
  window.setRefreshToken = setRefreshToken;
  window.refreshAccessToken = refreshAccessToken;
  window.authFetch = authFetch;
  window.resolveMediaUrl = resolveMediaUrl;
  window.MEDIA_PUBLIC_BASE = MEDIA_PUBLIC_BASE;
  window.DEFAULT_AVATAR_URL = DEFAULT_AVATAR_URL;
  window.GTG_DEBUG = DEBUG;

  // --- THEME INIT ---
  if (typeof document !== 'undefined') {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.body.dataset.theme = savedTheme;
  }

  // --- ICONS (PWA / iOS) ---
  if (typeof document !== 'undefined') {
    const ensureLink = (rel, href, sizes) => {
      let link = document.querySelector(`link[rel="${rel}"]`);
      if (!link) {
        link = document.createElement('link');
        link.rel = rel;
        document.head.appendChild(link);
      }
      if (sizes) link.sizes = sizes;
      link.href = href;
    };

    ensureLink('icon', '/assets/images/logo-gotogym-192.png');
    ensureLink('apple-touch-icon', '/assets/images/apple-touch-icon.png', '180x180');
  }

// --- AUTO-INJECT CHAT WIDGET ---
// Detecta si estamos en el navegador y carga el script del chat automáticamente
  if (typeof document !== 'undefined') {
    const CHAT_WIDGET_VERSION = '2026-02-21-8';
    if (!document.querySelector('script[data-gtg-chat-widget="1"]')) {
      const script = document.createElement('script');
      // Asumimos que la estructura es /js/chat.js relativa a la raíz del servidor web (port 5500)
      // Como config.js suele estar en /js/config.js, podemos intentar ruta relativa si la absoluta falla
      // Pero http://192.168.1.9:5500/js/chat.js es lo más seguro.
      script.src = `${window.location.origin}/js/chat.js?v=${CHAT_WIDGET_VERSION}`;
      script.async = true;
      script.dataset.gtgChatWidget = '1';
      document.body.appendChild(script);
    }

    // Notifications: no cargar en páginas de autenticación.
    const path = (window.location && window.location.pathname) ? window.location.pathname : '';
    const isAuthPage = path.includes('/pages/auth/') || path.includes('indexInicioDeSesion') || path.includes('indexRegistrar');
    if (!isAuthPage) {
      const notifScript = document.createElement('script');
      notifScript.src = window.location.origin + '/js/notifications.js';
      notifScript.async = true;
      document.body.appendChild(notifScript);
    }
}
})();
