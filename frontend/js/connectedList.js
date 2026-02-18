// connectedList.js (GoToGym) — Dispositivos con Switch ON/OFF
(() => {
  // === Fuente única de URL ===
  const API_URL = window.API_URL; // viene de config.js o fallback inline del HTML
  const authFetch = window.authFetch || fetch;

  // === Providers TOP 4 (alineado al backend Django) ===
  const PROVIDERS = [
    { key: "apple_health", name: "Apple Health", icon: "🍏", tags: ["Recomendado", "iOS"], oauth: false },
    { key: "google_fit", name: "Google Fit", icon: "🤖", tags: ["Android", "Wear OS"], oauth: true },
    { key: "garmin",       name: "Garmin Connect",icon:"🟦", tags: ["iOS / Android", "OAuth"], oauth: true },
    { key: "fitbit",       name: "Fitbit",       icon: "🟪", tags: ["iOS / Android", "OAuth"], oauth: true },
  ];

  // ========= Helpers =========
    function getAuthHeaders() {
      // Estrategia REAL del proyecto: /api/login
      const token =
        (typeof window.getAccessToken === "function" && window.getAccessToken()) ||
        localStorage.getItem("access") ||   // login original
        localStorage.getItem("token") ||  // fallback si existe
        null;

      const isLoggedIn = localStorage.getItem("isLoggedIn");
      const user = localStorage.getItem("user");

      console.log("AUTH STATE:", {
        hasToken: !!token,
        isLoggedIn,
        hasUser: !!user
      });

      // Si no hay token, la sesión es inválida
      if (!token || !isLoggedIn) {
        showToast("Sesión expirada. Inicia sesión nuevamente.", "error");

        // limpieza suave (no destructiva)
        localStorage.removeItem("isLoggedIn");

        setTimeout(() => {
          window.location.href =
            "../../pages/auth/indexInicioDeSesion.html";
        }, 300);

        return { "Content-Type": "application/json" };
      }

      return {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${token}`,
      };
    }

  function showToast(message, type = "success") {
    if (typeof window.showToast === "function") return window.showToast(message, type);
    const toast = document.getElementById("toast");
    if (!toast) return;
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = "block";
    clearTimeout(window.__toastTimer);
    window.__toastTimer = setTimeout(() => (toast.style.display = "none"), 2600);
  }

  function setStatus(text) {
    const el = document.getElementById("statusChip");
    if (el) el.textContent = text;
  }

  function setGlobalLoading(isLoading) {
    const gl = document.getElementById("globalLoading");
    const btn = document.getElementById("refreshBtn");
    const loadingState = document.getElementById("loadingState");
    const content = document.getElementById("devicesContent");
    if (gl) gl.style.display = isLoading ? "inline" : "none";
    if (btn) btn.disabled = !!isLoading;
    if (loadingState) loadingState.style.display = isLoading ? "block" : "none";
    if (content) content.style.display = isLoading ? "none" : "block";
  }

  function handleUnauthorized() {
    showToast("Sesión expirada. Inicia sesión nuevamente.", "error");
    localStorage.removeItem("isLoggedIn");
    localStorage.removeItem("access");
    localStorage.removeItem("refresh");
    localStorage.removeItem("token");
    setTimeout(() => {
      window.location.href = "../../pages/auth/indexInicioDeSesion.html";
    }, 300);
  }

  // Mapa rápido provider->meta (icon/name)
  const META = Object.fromEntries(PROVIDERS.map(p => [p.key, p]));

  function pillTag(t) {
    const low = String(t).toLowerCase();
    if (low === "oauth") return "pill warn";
    if (low === "recomendado") return "pill ok";
    return "pill";
  }

  function safeLabelFromStatus(status) {
    if (status === "connected") return { cls: "ok", text: "Conectado" };
    if (status === "error") return { cls: "err", text: "Error" };
    if (status === "pending") return { cls: "warn", text: "Pendiente" };
    return { cls: "warn", text: "Desconectado" };
  }

  // ========= Render =========
    function renderProviders(list, connectedSet) {
      connectedSet = connectedSet instanceof Set ? connectedSet : new Set();
      const el = document.getElementById("providersList");
      if (!el) return;

      el.innerHTML = "";

      list.forEach((p) => {
        const isOn = connectedSet.has(p.key);
        const enabled = p.enabled !== false; // default true si no viene
        const disabledReason = String(p.disabled_reason || p.disabledReason || '').trim();

        const card = document.createElement("div");
        card.className = `device-card ${isOn ? "connected" : ""}`;

        const tagsHtml = (p.tags || [])
          .map((t) => `<span class="${pillTag(t)}">${t}</span>`)
          .join("");

        const buttonClass = isOn ? "btn btn-outline-gold" : "btn btn-secondary";
        const buttonLabel = !enabled ? "Próximamente" : (isOn ? "Desconectar" : "Conectar");

        card.innerHTML = `
          <div class="device-info">
            <div class="device-icon">${p.icon || "⌚"}</div>
            <div>
              <div class="device-name">${p.name}</div>
              <div class="device-status">${tagsHtml}</div>
            </div>
          </div>

          <div style="display:flex; gap:10px; align-items:center;">
            <button class="${buttonClass}" data-action="toggle" ${enabled ? "" : "disabled"} title="${!enabled && disabledReason ? disabledReason.replace(/\"/g,'&quot;') : ''}">
              ${buttonLabel}
            </button>
          </div>
        `;

        card.querySelector(`[data-action="toggle"]`)?.addEventListener("click", async (e) => {
          const btn = e.currentTarget;
          btn.disabled = true;
          try {
            if (!enabled) {
              showToast(disabledReason || "Próximamente.", "error");
              return;
            }
            if (!isOn) await connect(p.key);
            else await disconnect(p.key);
          } finally {
            btn.disabled = !enabled ? true : false;
          }
        });

        el.appendChild(card);
      });
    }

    function renderConnected(devices) {
      const el = document.getElementById("connectedList");
      const badge = document.getElementById("connectedCount");

      const count = Array.isArray(devices) ? devices.length : 0;
      if (badge) badge.textContent = String(count);
      if (!el) return;

      if (!count) {
        el.innerHTML = `<div class="empty">No tienes dispositivos conectados todavía.</div>`;
        return;
      }

      el.innerHTML = "";

      devices.forEach((d) => {
        const meta = META[d.provider] || {};
        const statusInfo = safeLabelFromStatus(d.status);
        const lastSync = d.last_sync_label || "—";

        const card = document.createElement("div");
        card.className = `device-card ${d.status === "connected" ? "connected" : ""}`;

        card.innerHTML = `
          <div class="device-info">
            <div class="device-icon">${meta.icon || "⌚"}</div>
            <div>
              <div class="device-name">${meta.name || d.provider}</div>
              <div class="device-status">
                <span class="pill ${statusInfo.cls}">${statusInfo.text}</span>
                <span class="pill">Última sync: ${lastSync}</span>
              </div>
            </div>
          </div>

          <div style="display:flex; gap:10px; align-items:center;">
            <button class="btn btn-secondary" data-sync="${d.provider}">Sync</button>
            <button class="btn btn-outline-gold" data-disconnect="${d.provider}">Desconectar</button>
          </div>
        `;

        card.querySelector(`[data-sync]`)?.addEventListener("click", () => syncNow(d.provider));
        card.querySelector(`[data-disconnect]`)?.addEventListener("click", () => disconnect(d.provider));

        el.appendChild(card);
      });
    }

  // ========= API =========
  async function loadDevices() {
    if (!API_URL) {
      console.error("API_URL no está definida");
      showToast("Falta API_URL. Revisa config.js.", "error");
      return;
    }

    try {
      setStatus("Cargando…");
      setGlobalLoading(true);

        const res = await authFetch(API_URL + "devices/", {
        method: "GET",
        headers: getAuthHeaders(),
      });

      if (res.status === 401) {
        handleUnauthorized();
        return;
      }
      if (!res.ok) throw new Error("No se pudo cargar devices: " + res.status);

      const data = await res.json();

      const connected = Array.isArray(data.connected) ? data.connected : [];
      const connectedSet = new Set(
        connected
          .filter(x => x.status !== "disconnected") // por si acaso
          .map(x => x.provider)
      );

      // Providers: usa backend si viene, sino usa fallback
      const providersRaw = Array.isArray(data.providers) ? data.providers : null;

      // Normaliza providers del backend si vienen como {provider,label}
      const providers =
        providersRaw
          ? providersRaw.map(p => ({
              key: p.provider,
              name: p.label || (META[p.provider]?.name ?? p.provider),
              icon: META[p.provider]?.icon ?? "⌚",
              tags: META[p.provider]?.tags ?? [],
              oauth: META[p.provider]?.oauth ?? false,
            }))
          : PROVIDERS;

        const availableProviders = providers.filter(p => !connectedSet.has(p.key));
        
        renderProviders(availableProviders, connectedSet);
        renderConnected(connected);

      setStatus("Listo");
    } catch (e) {
      console.error(e);
      setStatus("Error");
      renderProviders(PROVIDERS, new Set());
      renderConnected([]);
      showToast("No se pudo cargar la lista. Revisa sesión / backend.", "error");
    } finally {
      setGlobalLoading(false);
    }
  }

  async function connect(provider) {
    setStatus("Conectando…");
    showToast(`Conectando ${provider}…`, "success");

    const res = await authFetch(API_URL + `devices/${provider}/connect/`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({}),
    });

    if (res.status === 401) {
      handleUnauthorized();
      return;
    }

    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Connect failed");

      if (data.redirect_url) {
        const token = localStorage.getItem("access") || localStorage.getItem("token");

        if (!token) {
          showToast("No hay token en sesión. Inicia sesión de nuevo.", "error");
          return;
        }

        const u = data.redirect_url;
        const BACKEND_BASE = API_URL.replace(/\/api\/?$/, "");
        const sep = u.includes("?") ? "&" : "?";
        window.location.href = `${BACKEND_BASE}${u}${sep}token=${encodeURIComponent(token)}`;
        return;
      }

    showToast("Conectado. Actualizando…", "success");
    await loadDevices();
    setStatus("Listo");
  }

    async function disconnect(provider) {
    try
        {setStatus("Desconectando…");
    const res = await authFetch(API_URL + `devices/${provider}/disconnect/`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify({}),
    });
    if (res.status === 401) {
    handleUnauthorized();
    return;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || "Disconnect failed");
    showToast("Dispositivo desconectado.", "success");
    await loadDevices();
    } catch (e) {
    console.error(e);
    setStatus("Error");
    showToast("No se pudo desconectar.", "error");
    } finally {
    setStatus("Listo");
    }
    }
    async function syncNow(provider) {
    try {
    setStatus("Sincronizando…");
    const res = await authFetch(API_URL + `devices/${provider}/sync/`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: JSON.stringify({}),
    });
    if (res.status === 401) {
    handleUnauthorized();
    return;
    }
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const msg = data.error || data.detail || `No se pudo sincronizar (HTTP ${res.status}).`;
      throw new Error(msg);
    }
    if (data && data.ok === false && data.error) {
      showToast(String(data.error), "error");
    } else {
      showToast("Sincronización completada.", "success");
    }
    await loadDevices();
    } catch (e) {
    console.error(e);
    setStatus("Error");
    showToast("No se pudo sincronizar.", "error");
    } finally {
    setStatus("Listo");
    }
    }
    // Exponer solo si tu HTML llama funciones inline (si ya no, no importa)
    window.loadDevices = loadDevices;
    window.connect = connect;
    window.disconnect = disconnect;
    window.syncNow = syncNow;
    // Init
    document.addEventListener("DOMContentLoaded", () => {
    // feedback al volver desde OAuth
    const params = new URLSearchParams(window.location.search);
    const oauth = params.get("oauth");
    const provider = params.get("provider");
    if (oauth === "success") {
      showToast(`Autorización exitosa${provider ? `: ${provider}` : ""}`, "success");
      params.delete("oauth");
      params.delete("provider");
      const next = params.toString();
      const clean = next ? `${window.location.pathname}?${next}` : window.location.pathname;
      window.history.replaceState({}, "", clean);
    }

    // botón actualizar
    document.getElementById("refreshBtn")?.addEventListener("click", () =>
    loadDevices());
    loadDevices();
    });
    })();
