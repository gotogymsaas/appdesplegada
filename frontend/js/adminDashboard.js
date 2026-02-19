// adminDashboard.js — Panel Admin (GoToGym)
// Objetivo: mantener UX existente, agregar diagramas de tiempo y facilitar edición.
// Depende de: theme.css (variables), config.js (API_URL, authFetch).

(() => {
  const DEFAULT_DAYS_WINDOW = 90;
  const RANGE_STORAGE_KEY = 'adminDashboardRangeDays';

  const els = {
    kpiTotal: () => document.getElementById('kpi-total'),
    kpiPremium: () => document.getElementById('kpi-premium'),
    kpiToday: () => document.getElementById('kpi-today'),
    kpi7d: () => document.getElementById('kpi-7d'),
    kpi30d: () => document.getElementById('kpi-30d'),
    kpiHappy7d: () => document.getElementById('kpi-happy-7d'),
    usersBody: () => document.getElementById('usersBody'),
    searchInput: () => document.getElementById('searchInput'),

    globalChart: () => document.getElementById('globalChart'),
    signupsChart: () => document.getElementById('signupsChart'),
    signupsTrendChart: () => document.getElementById('signupsTrendChart'),
    usersCumulativeChart: () => document.getElementById('usersCumulativeChart'),
    premiumShareChart: () => document.getElementById('premiumShareChart'),
    planChart: () => document.getElementById('planChart'),

    rangeSelect: () => document.getElementById('rangeSelect'),
    rangeLabel: () => document.getElementById('rangeLabel'),

    bulkCreateInput: () => document.getElementById('bulkCreateInput'),
    bulkPlanInput: () => document.getElementById('bulkPlanInput'),

    createModal: () => document.getElementById('createModal'),
    editModal: () => document.getElementById('editModal'),

    editUserId: () => document.getElementById('editUserId'),
    editUsername: () => document.getElementById('editUsername'),
    editEmail: () => document.getElementById('editEmail'),
    editPlan: () => document.getElementById('editPlan'),
  };

  const authFetch = window.authFetch;
  const API_URL = window.API_URL;

  let allUsers = [];

  let daysWindow = DEFAULT_DAYS_WINDOW;
  let globalHistoryCache = null;

  let globalChartInstance = null;
  let signupsChartInstance = null;
  let signupsTrendChartInstance = null;
  let planChartInstance = null;
  let usersCumulativeChartInstance = null;
  let premiumShareChartInstance = null;

  function getThemeVars() {
    const styles = getComputedStyle(document.documentElement);
    const primary = (styles.getPropertyValue('--primary') || '').trim() || '#0FBFB0';
    const secondary = (styles.getPropertyValue('--secondary') || '').trim() || '#D4B46A';
    const textMuted = (styles.getPropertyValue('--text-muted') || '').trim() || '#AAAAAA';
    return { primary, secondary, textMuted };
  }

  function clampDaysWindow(value) {
    const v = Number(value);
    if (v === 90 || v === 180 || v === 270 || v === 365 || v === 730) return v;
    return DEFAULT_DAYS_WINDOW;
  }

  function rangeLabelText(days) {
    if (days === 730) return 'últimos 2 años';
    return `últimos ${days} días`;
  }

  function syncRangeUI() {
    const labelEl = els.rangeLabel();
    if (labelEl) labelEl.textContent = rangeLabelText(daysWindow);

    document.querySelectorAll('[data-range-chip]').forEach((el) => {
      el.textContent = `Últimos ${daysWindow} días`;
    });

    const selectEl = els.rangeSelect();
    if (selectEl) selectEl.value = String(daysWindow);
  }

  function showToast(message, type = 'success') {
    const t = document.getElementById('toast');
    if (!t) return;
    t.textContent = message;
    t.className = `toast ${type}`;
    t.style.display = 'block';
    setTimeout(() => (t.style.display = 'none'), 3000);
  }

  function requireAdminSessionOrRedirect() {
    const userStr = localStorage.getItem('user');
    const hasToken = !!(localStorage.getItem('access') || localStorage.getItem('token'));
    const isLoggedIn = localStorage.getItem('isLoggedIn') === 'true';

    if (!userStr || !hasToken || !isLoggedIn) {
      window.location.href = '/pages/auth/indexInicioDeSesion.html';
      return false;
    }

    try {
      const user = JSON.parse(userStr);
      if (!user.is_superuser) {
        alert('Acceso denegado. Área exclusiva para administradores.');
        window.location.href = '/pages/profile/Perfil.html';
        return false;
      }
      return true;
    } catch (e) {
      localStorage.removeItem('user');
      window.location.href = '/pages/auth/indexInicioDeSesion.html';
      return false;
    }
  }

  function calculateKPIs(users) {
    if (els.kpiTotal()) els.kpiTotal().innerText = String(users.length);

    const premiumCount = users.filter((u) => u.plan === 'Premium').length;
    const percentage = users.length > 0 ? Math.round((premiumCount / users.length) * 100) : 0;
    if (els.kpiPremium()) els.kpiPremium().innerText = `${premiumCount} (${percentage}%)`;

    const today = new Date().toLocaleDateString();
    const newToday = users.filter(
      (u) => u.date_joined && new Date(u.date_joined).toLocaleDateString() === today,
    ).length;
    if (els.kpiToday()) els.kpiToday().innerText = String(newToday);

    const countSinceDays = (days) => {
      const now = new Date();
      const start = new Date(now);
      start.setHours(0, 0, 0, 0);
      start.setDate(now.getDate() - (days - 1));

      let count = 0;
      users.forEach((u) => {
        if (!u || !u.date_joined) return;
        try {
          const dt = new Date(u.date_joined);
          if (!Number.isNaN(dt.getTime()) && dt >= start) count += 1;
        } catch (e) {
          // ignore
        }
      });
      return count;
    };

    if (els.kpi7d()) els.kpi7d().innerText = String(countSinceDays(7));
    if (els.kpi30d()) els.kpi30d().innerText = String(countSinceDays(30));
  }

  function renderTable(users) {
    const tbody = els.usersBody();
    if (!tbody) return;

    tbody.innerHTML = '';

    users.forEach((user) => {
      const row = document.createElement('tr');
      const date = user.date_joined ? new Date(user.date_joined).toLocaleDateString() : '-';
      const badgeClass = user.plan === 'Premium' ? 'badge-premium' : 'badge-free';

      const nextPlan = user.plan === 'Premium' ? 'Gratis' : 'Premium';
      const planBtnLabel = user.plan === 'Premium' ? 'Quitar Premium' : 'Dar Premium';
      row.innerHTML = `
        <td>#${user.id}</td>
        <td style="font-weight: bold; color: var(--text-main);">${user.username}</td>
        <td style="color: var(--text-muted);">${user.email}</td>
        <td><span class="badge ${badgeClass}">${user.plan}</span></td>
        <td>${date}</td>
        <td>
          <button class="btn-delete" style="border-color: var(--secondary); color: var(--secondary); margin-right:5px;" onclick='window.openEditModal(${JSON.stringify(
            user,
          )})'>✏️</button>
          <button class="btn-delete" style="border-color: var(--primary); color: var(--primary); margin-right:5px;" onclick="window.quickSetPlan(${user.id}, '${nextPlan}')">${planBtnLabel}</button>
          <button class="btn-delete" onclick="window.deleteUser(${user.id})">🗑️</button>
        </td>
      `;

      tbody.appendChild(row);
    });
  }

  function lastNDaysLabels(days) {
    const labels = [];
    const now = new Date();
    for (let i = days - 1; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(now.getDate() - i);
      labels.push(d.toISOString().slice(0, 10));
    }
    return labels;
  }

  function safeISODate(dateStr) {
    if (!dateStr) return null;
    try {
      const d = new Date(dateStr);
      if (Number.isNaN(d.getTime())) return null;
      return d.toISOString().slice(0, 10);
    } catch (e) {
      return null;
    }
  }

  function countSignupsByDay(users, labels) {
    const counts = Object.fromEntries(labels.map((l) => [l, 0]));
    users.forEach((u) => {
      if (!u || !u.date_joined) return;
      try {
        const day = safeISODate(u.date_joined);
        if (!day) return;
        if (counts[day] !== undefined) counts[day] += 1;
      } catch (e) {
        // ignore
      }
    });
    return labels.map((l) => counts[l] || 0);
  }

  function countSignupsByDayAndPlan(users, labels) {
    const premiumCounts = Object.fromEntries(labels.map((l) => [l, 0]));
    const freeCounts = Object.fromEntries(labels.map((l) => [l, 0]));

    users.forEach((u) => {
      if (!u || !u.date_joined) return;
      const day = safeISODate(u.date_joined);
      if (!day) return;
      if (premiumCounts[day] === undefined) return;
      if (u.plan === 'Premium') premiumCounts[day] += 1;
      else freeCounts[day] += 1;
    });

    return {
      premium: labels.map((l) => premiumCounts[l] || 0),
      free: labels.map((l) => freeCounts[l] || 0),
    };
  }

  function rollingAverage(values, windowSize) {
    const out = [];
    const w = Math.max(1, Number(windowSize) || 1);
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - w + 1);
      const slice = values.slice(start, i + 1);
      const sum = slice.reduce((a, b) => a + (Number(b) || 0), 0);
      out.push(slice.length ? sum / slice.length : 0);
    }
    return out;
  }

  function cumulativeSeries(users, labels) {
    const labelSet = new Set(labels);
    const counts = Object.fromEntries(labels.map((l) => [l, { total: 0, premium: 0 }]));

    users.forEach((u) => {
      const day = safeISODate(u?.date_joined);
      if (!day) return;
      if (!labelSet.has(day)) return;
      counts[day].total += 1;
      if (u.plan === 'Premium') counts[day].premium += 1;
    });

    let runningTotal = 0;
    let runningPremium = 0;
    const cumulativeTotal = [];
    const cumulativePremium = [];

    labels.forEach((l) => {
      runningTotal += counts[l]?.total || 0;
      runningPremium += counts[l]?.premium || 0;
      cumulativeTotal.push(runningTotal);
      cumulativePremium.push(runningPremium);
    });

    const premiumShare = cumulativeTotal.map((t, i) => (t > 0 ? Math.round((cumulativePremium[i] / t) * 1000) / 10 : 0));

    return { cumulativeTotal, cumulativePremium, premiumShare };
  }

  function loadUserCharts(users) {
    const { primary, secondary, textMuted } = getThemeVars();

    const labels = lastNDaysLabels(daysWindow);
    const signups = countSignupsByDay(users, labels);
    const signupsByPlan = countSignupsByDayAndPlan(users, labels);
    const trend = rollingAverage(signups, 7);

    // 1) Altas (Premium vs Gratis) - stacked bar
    const signupsEl = els.signupsChart();
    if (signupsEl && window.Chart) {
      const ctx = signupsEl.getContext('2d');
      if (signupsChartInstance) signupsChartInstance.destroy();
      signupsChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'Premium',
              data: signupsByPlan.premium,
              backgroundColor: 'rgba(212, 180, 106, 0.35)',
              borderWidth: 0,
              stack: 'signups',
            },
            {
              label: 'Gratis',
              data: signupsByPlan.free,
              backgroundColor: 'rgba(255, 255, 255, 0.10)',
              borderWidth: 0,
              stack: 'signups',
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: textMuted, boxWidth: 10 },
            },
          },
          scales: {
            y: {
              beginAtZero: true,
              stacked: true,
              grid: { color: 'rgba(255,255,255,0.1)' },
              ticks: { color: textMuted },
            },
            x: {
              stacked: true,
              grid: { display: false },
              ticks: { maxTicksLimit: 7, color: textMuted },
            },
          },
        },
      });
    }

    // 2) Tendencia (promedio móvil 7d) - line
    const trendEl = els.signupsTrendChart();
    if (trendEl && window.Chart) {
      const ctx = trendEl.getContext('2d');
      if (signupsTrendChartInstance) signupsTrendChartInstance.destroy();
      signupsTrendChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Promedio móvil 7d',
              data: trend,
              borderColor: primary,
              backgroundColor: 'rgba(15, 191, 176, 0.12)',
              borderWidth: 2,
              tension: 0.35,
              fill: true,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: textMuted } },
            x: { grid: { display: false }, ticks: { maxTicksLimit: 7, color: textMuted } },
          },
        },
      });
    }

    // 3) Acumulados y % premium
    const { cumulativeTotal, premiumShare } = cumulativeSeries(users, labels);

    const cumEl = els.usersCumulativeChart();
    if (cumEl && window.Chart) {
      const ctx = cumEl.getContext('2d');
      if (usersCumulativeChartInstance) usersCumulativeChartInstance.destroy();
      usersCumulativeChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Usuarios',
              data: cumulativeTotal,
              borderColor: secondary,
              backgroundColor: 'rgba(212, 180, 106, 0.10)',
              borderWidth: 2,
              tension: 0.25,
              fill: true,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: textMuted } },
            x: { grid: { display: false }, ticks: { maxTicksLimit: 7, color: textMuted } },
          },
        },
      });
    }

    const shareEl = els.premiumShareChart();
    if (shareEl && window.Chart) {
      const ctx = shareEl.getContext('2d');
      if (premiumShareChartInstance) premiumShareChartInstance.destroy();
      premiumShareChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: '% Premium',
              data: premiumShare,
              borderColor: secondary,
              backgroundColor: 'rgba(212, 180, 106, 0.06)',
              borderWidth: 2,
              tension: 0.35,
              fill: true,
              pointRadius: 0,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: 'rgba(255,255,255,0.1)' },
              ticks: { color: textMuted, callback: (v) => `${v}%` },
            },
            x: { grid: { display: false }, ticks: { maxTicksLimit: 7, color: textMuted } },
          },
        },
      });
    }

    const premium = users.filter((u) => u && u.plan === 'Premium').length;
    const free = Math.max(0, users.length - premium);

    const planEl = els.planChart();
    if (planEl && window.Chart) {
      const ctx = planEl.getContext('2d');
      if (planChartInstance) planChartInstance.destroy();
      planChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: ['Premium', 'Gratis'],
          datasets: [
            {
              data: [premium, free],
              borderWidth: 0,
              backgroundColor: ['rgba(212, 180, 106, 0.35)', 'rgba(255, 255, 255, 0.12)'],
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: textMuted, boxWidth: 10 },
            },
          },
        },
      });
    }
  }

  async function loadGlobalChart() {
    const { secondary } = getThemeVars();

    try {
      if (!globalHistoryCache) {
        const res = await authFetch(API_URL + 'stats/global_history/');
        const data = await res.json();
        globalHistoryCache = Array.isArray(data) ? data : [];
      }

      const data = globalHistoryCache;
      if (!data || data.length === 0) return;

      const sliced = data.slice(-daysWindow);
      const labels = sliced.map((d) => d.date);
      const values = sliced.map((d) => d.value);

      // KPI adicional: felicidad promedio últimos 7 datos (si vienen en orden cronológico)
      try {
        const nums = (values || []).map((v) => Number(v)).filter((n) => Number.isFinite(n));
        if (els.kpiHappy7d()) {
          if (nums.length === 0) {
            els.kpiHappy7d().innerText = '--';
          } else {
            const last = nums.slice(-7);
            const avg = last.reduce((a, b) => a + b, 0) / last.length;
            els.kpiHappy7d().innerText = avg.toFixed(1);
          }
        }
      } catch (e) {
        if (els.kpiHappy7d()) els.kpiHappy7d().innerText = '--';
      }

      const globalEl = els.globalChart();
      if (!globalEl || !window.Chart) return;

      const ctx = globalEl.getContext('2d');
      if (globalChartInstance) globalChartInstance.destroy();
      globalChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels,
          datasets: [
            {
              label: 'Felicidad Promedio',
              data: values,
              borderColor: secondary,
              backgroundColor: 'rgba(212, 180, 106, 0.1)',
              borderWidth: 2,
              tension: 0.4,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, max: 10, grid: { color: 'rgba(255,255,255,0.1)' } },
            x: { grid: { display: false }, ticks: { maxTicksLimit: 7 } },
          },
        },
      });
    } catch (e) {
      console.error('Chart error', e);
    }
  }

  function setDaysWindow(newDays) {
    daysWindow = clampDaysWindow(newDays);
    localStorage.setItem(RANGE_STORAGE_KEY, String(daysWindow));
    syncRangeUI();
    renderDashboard();
  }

  function loadStoredDaysWindow() {
    const v = localStorage.getItem(RANGE_STORAGE_KEY);
    daysWindow = clampDaysWindow(v || DEFAULT_DAYS_WINDOW);
  }

  function bindRangeSelect() {
    const selectEl = els.rangeSelect();
    if (!selectEl) return;
    selectEl.addEventListener('change', (e) => {
      setDaysWindow(e.target.value);
    });
  }

  function renderDashboard() {
    renderTable(allUsers);
    calculateKPIs(allUsers);
    loadUserCharts(allUsers);
    loadGlobalChart();
  }

  async function fetchUsers() {
    try {
      const res = await authFetch(API_URL + 'users/');
      const users = await res.json();
      if (users.error) {
        showToast('Error cargando usuarios', 'error');
        return;
      }
      allUsers = Array.isArray(users) ? users : [];
      renderDashboard();
    } catch (e) {
      console.error(e);
      showToast('Error de conexión', 'error');
    }
  }

  function filterUsers() {
    const term = (els.searchInput()?.value || '').toLowerCase();
    const filtered = allUsers.filter(
      (u) => u.username.toLowerCase().includes(term) || u.email.toLowerCase().includes(term),
    );
    renderTable(filtered);
  }

  async function deleteUser(id) {
    if (!confirm('¿Eliminar usuario?')) return;
    try {
      const res = await authFetch(API_URL + `users/delete/${id}/`, { method: 'DELETE' });
      if (res.ok) {
        showToast('Eliminado', 'success');
        fetchUsers();
      } else {
        showToast('Error', 'error');
      }
    } catch (e) {
      showToast('Error', 'error');
    }
  }

  async function quickSetPlan(userId, plan) {
    const user = allUsers.find((u) => u && u.id === userId);
    if (!user) {
      showToast('Usuario no encontrado', 'error');
      return;
    }
    const pretty = plan === 'Premium' ? 'Premium' : 'Gratis';
    if (!confirm(`¿Cambiar plan a ${pretty} para ${user.email}?`)) return;

    try {
      const res = await authFetch(API_URL + `users/update_admin/${userId}/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user.username, email: user.email, plan: pretty }),
      });
      if (res.ok) {
        showToast('Plan actualizado', 'success');
        fetchUsers();
      } else {
        showToast('Error actualizando plan', 'error');
      }
    } catch (e) {
      showToast('Error actualizando plan', 'error');
    }
  }

  function parseCSV(text) {
    const rows = [];
    let row = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      const next = text[i + 1];
      if (ch === '"') {
        if (inQuotes && next === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (!inQuotes && ch === ',') {
        row.push(cur.trim());
        cur = '';
      } else if (!inQuotes && (ch === '\n' || ch === '\r')) {
        if (ch === '\r' && next === '\n') i++;
        row.push(cur.trim());
        cur = '';
        if (row.some((c) => c !== '')) rows.push(row);
        row = [];
      } else {
        cur += ch;
      }
    }
    row.push(cur.trim());
    if (row.some((c) => c !== '')) rows.push(row);
    return rows;
  }

  function normalizeHeader(header) {
    return String(header || '')
      .trim()
      .toLowerCase()
      .replace(/\s+/g, '_');
  }

  function rowsToObjects(csvRows) {
    if (!csvRows || csvRows.length < 2) return [];
    const headers = csvRows[0].map(normalizeHeader);
    const out = [];
    for (let i = 1; i < csvRows.length; i++) {
      const r = csvRows[i];
      const obj = {};
      headers.forEach((h, idx) => {
        obj[h] = (r[idx] || '').trim();
      });
      out.push(obj);
    }
    return out;
  }

  async function readFileAsText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('file_read_error'));
      reader.readAsText(file);
    });
  }

  async function bulkCreateUsersFromCsv(file) {
    const text = await readFileAsText(file);
    const objs = rowsToObjects(parseCSV(text));
    if (objs.length === 0) {
      showToast('CSV vacío', 'error');
      return;
    }

    const hasEmail = 'email' in objs[0];
    if (!hasEmail) {
      showToast('Falta columna: email', 'error');
      return;
    }

    const hasUsername = 'username' in objs[0];
    const hasPassword = 'password' in objs[0];
    // plan es opcional

    const modeText = hasUsername && hasPassword ? 'manual' : 'automático';
    if (!confirm(`Se procesarán ${objs.length} usuarios (modo ${modeText}). ¿Continuar?`)) return;

    const existingUsernames = new Set(allUsers.map((u) => String(u.username || '').toLowerCase()));
    const stagedUsernames = new Set();

    const sanitizeUsername = (value) => {
      const v = String(value || '')
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9_\-\.]/g, '_')
        .replace(/_+/g, '_')
        .replace(/^_+|_+$/g, '');
      return v || 'user';
    };

    const uniqueUsername = (base) => {
      let candidate = sanitizeUsername(base);
      if (candidate.length > 28) candidate = candidate.slice(0, 28);

      const isTaken = (name) => existingUsernames.has(name) || stagedUsernames.has(name);
      if (!isTaken(candidate)) {
        stagedUsernames.add(candidate);
        return candidate;
      }

      for (let i = 1; i <= 9999; i++) {
        const suffix = String(i);
        const trimmed = candidate.slice(0, Math.max(1, 28 - suffix.length));
        const attempt = `${trimmed}${suffix}`;
        if (!isTaken(attempt)) {
          stagedUsernames.add(attempt);
          return attempt;
        }
      }

      // fallback
      const rand = String(Math.floor(Math.random() * 100000));
      const fallback = `${candidate.slice(0, Math.max(1, 28 - rand.length))}${rand}`;
      stagedUsernames.add(fallback);
      return fallback;
    };

    const generatePassword = () => {
      const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789!@#$%';
      const length = 12;
      let out = '';
      try {
        const buf = new Uint32Array(length);
        (window.crypto || window.msCrypto).getRandomValues(buf);
        for (let i = 0; i < length; i++) out += alphabet[buf[i] % alphabet.length];
        return out;
      } catch (e) {
        for (let i = 0; i < length; i++) out += alphabet[Math.floor(Math.random() * alphabet.length)];
        return out;
      }
    };

    const normalizePlan = (planValue) => {
      const p = String(planValue || '').trim().toLowerCase();
      if (p === 'premium') return 'Premium';
      return 'Gratis';
    };

    const createdCreds = [];
    let ok = 0;
    let fail = 0;

    for (const u of objs) {
      const email = String(u.email || '').trim();
      if (!email) {
        fail += 1;
        continue;
      }

      const plan = normalizePlan(u.plan);
      const username = hasUsername ? String(u.username || '').trim() : uniqueUsername(email.split('@')[0]);
      const password = hasPassword ? String(u.password || '').trim() : generatePassword();

      if (!username || !password) {
        fail += 1;
        continue;
      }

      try {
        const res = await authFetch(API_URL + 'users/create/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, email, password, plan }),
        });
        if (res.ok) {
          ok += 1;
          createdCreds.push({ username, email, password, plan });
        } else {
          fail += 1;
        }
      } catch (e) {
        fail += 1;
      }
    }

    if (createdCreds.length) {
      try {
        const header = 'username,email,password,plan\n';
        const lines = createdCreds
          .map((c) => {
            const esc = (s) => {
              const v = String(s ?? '');
              if (/[",\n\r]/.test(v)) return `"${v.replace(/"/g, '""')}"`;
              return v;
            };
            return `${esc(c.username)},${esc(c.email)},${esc(c.password)},${esc(c.plan)}`;
          })
          .join('\n');
        const csv = header + lines + '\n';

        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        const stamp = new Date().toISOString().slice(0, 10);
        a.download = `credenciales_creadas_${stamp}.csv`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } catch (e) {
        // ignore
      }
    }

    showToast(`Carga masiva: ${ok} ok, ${fail} fallidos`, fail ? 'error' : 'success');
    fetchUsers();
  }

  async function bulkUpdatePlansFromCsv(file) {
    const text = await readFileAsText(file);
    const objs = rowsToObjects(parseCSV(text));
    if (objs.length === 0) {
      showToast('CSV vacío', 'error');
      return;
    }

    const required = ['email', 'plan'];
    const missing = required.filter((k) => !(k in objs[0]));
    if (missing.length) {
      showToast(`Faltan columnas: ${missing.join(', ')}`, 'error');
      return;
    }

    if (!confirm(`Se actualizarán planes para ${objs.length} registros. ¿Continuar?`)) return;

    const byEmail = new Map(allUsers.map((u) => [String(u.email || '').toLowerCase(), u]));
    let ok = 0;
    let fail = 0;

    for (const row of objs) {
      const email = String(row.email || '').trim().toLowerCase();
      const plan = String(row.plan || '').trim();
      const target = byEmail.get(email);
      if (!email || !target) {
        fail += 1;
        continue;
      }
      const normalizedPlan = plan.toLowerCase() === 'premium' ? 'Premium' : 'Gratis';
      try {
        const res = await authFetch(API_URL + `users/update_admin/${target.id}/`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: target.username, email: target.email, plan: normalizedPlan }),
        });
        if (res.ok) ok += 1;
        else fail += 1;
      } catch (e) {
        fail += 1;
      }
    }

    showToast(`Planes: ${ok} ok, ${fail} fallidos`, fail ? 'error' : 'success');
    fetchUsers();
  }

  function triggerBulkCreate() {
    const input = els.bulkCreateInput();
    if (!input) return;
    input.value = '';
    input.click();
  }

  function triggerBulkPlanUpdate() {
    const input = els.bulkPlanInput();
    if (!input) return;
    input.value = '';
    input.click();
  }

  function bindBulkInputs() {
    const createInput = els.bulkCreateInput();
    if (createInput) {
      createInput.addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        await bulkCreateUsersFromCsv(file);
      });
    }

    const planInput = els.bulkPlanInput();
    if (planInput) {
      planInput.addEventListener('change', async (e) => {
        const file = e.target.files && e.target.files[0];
        if (!file) return;
        await bulkUpdatePlansFromCsv(file);
      });
    }
  }

  function openModal() {
    const m = els.createModal();
    if (m) m.style.display = 'flex';
  }

  function closeModal() {
    const m = els.createModal();
    if (m) m.style.display = 'none';
  }

  function openEditModal(user) {
    if (!user) return;
    if (els.editUserId()) els.editUserId().value = user.id;
    if (els.editUsername()) els.editUsername().value = user.username;
    if (els.editEmail()) els.editEmail().value = user.email;
    if (els.editPlan()) els.editPlan().value = user.plan;
    const m = els.editModal();
    if (m) m.style.display = 'flex';
  }

  function closeEditModal() {
    const m = els.editModal();
    if (m) m.style.display = 'none';
  }

  async function handleCreate(e) {
    e.preventDefault();
    const username = document.getElementById('newUsername').value;
    const email = document.getElementById('newEmail').value;
    const password = document.getElementById('newPassword').value;
    const plan = document.getElementById('newPlan').value;

    try {
      const res = await authFetch(API_URL + 'users/create/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password, plan }),
      });
      if (res.ok) {
        showToast('Usuario creado', 'success');
        closeModal();
        fetchUsers();
        e.target.reset();
      } else {
        showToast('Error al crear', 'error');
      }
    } catch (err) {
      showToast('Error', 'error');
    }
  }

  async function handleEdit(e) {
    e.preventDefault();
    const id = els.editUserId().value;
    const username = els.editUsername().value;
    const email = els.editEmail().value;
    const plan = els.editPlan().value;

    try {
      const res = await authFetch(API_URL + `users/update_admin/${id}/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, plan }),
      });

      if (res.ok) {
        showToast('Usuario actualizado', 'success');
        closeEditModal();
        fetchUsers();
      } else {
        showToast('Error al actualizar', 'error');
      }
    } catch (err) {
      showToast('Error', 'error');
    }
  }

  async function requestPermission() {
    if (!('Notification' in window)) {
      showToast('Este navegador no soporta notificaciones', 'error');
      return false;
    }

    const permission = await Notification.requestPermission();
    if (permission !== 'granted') {
      showToast('Permiso denegado ❌', 'error');
      return false;
    }

    // Registrar SW a nivel raíz y suscribir Web Push (para que el admin también pueda recibir pruebas)
    try {
      if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
        showToast('Web Push no soportado en este navegador', 'error');
        return true;
      }

      const reg = await navigator.serviceWorker.register('/sw.js');
      const existing = await reg.pushManager.getSubscription();
      if (existing && existing.endpoint) {
        await authFetch(API_URL + 'push/web/subscribe/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            endpoint: existing.endpoint,
            keys: existing.toJSON().keys,
            device_id: localStorage.getItem('gtg_device_id') || '',
          }),
        });
        showToast('Permisos activados ✅');
        return true;
      }

      const keyRes = await authFetch(API_URL + 'push/web/key/');
      if (!keyRes.ok) {
        showToast('VAPID no configurado en el servidor', 'error');
        return true;
      }
      const keyData = await keyRes.json();
      const publicKey = keyData && keyData.public_key ? keyData.public_key : '';
      if (!publicKey) {
        showToast('VAPID no configurado', 'error');
        return true;
      }

      const urlBase64ToUint8Array = (base64String) => {
        const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
        const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
        const raw = window.atob(base64);
        const output = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i += 1) output[i] = raw.charCodeAt(i);
        return output;
      };

      const sub = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(publicKey),
      });

      await authFetch(API_URL + 'push/web/subscribe/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          endpoint: sub.endpoint,
          keys: sub.toJSON().keys,
          device_id: localStorage.getItem('gtg_device_id') || '',
        }),
      });
    } catch (e) {
      console.error('web push subscribe error', e);
    }

    showToast('Permisos activados ✅');
    return true;
  }

  async function sendDashboardNotification() {
    try {
      const variable = document.getElementById('notifVariable').value;
      const questions = {
        s_sleep: '¿Cómo dormiste?',
        s_stress_inv: '¿Cómo está tu estrés?',
        s_energy: '¿Cómo está tu energía?',
      };
      const questionBody = questions[variable] || '¿Cómo te sientes?';

      const res = await authFetch(API_URL + 'push/admin/broadcast/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: 'GoToGym Alerta 🔔',
          body: questionBody,
          data: {
            source: 'admin_dashboard',
            variable,
          },
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        showToast(err.error || 'Error enviando broadcast', 'error');
        return;
      }

      const out = await res.json().catch(() => ({}));
      const webSent = out?.result?.web?.sent || 0;
      const fcmSent = out?.result?.fcm?.sent || 0;
      showToast(`Broadcast enviado ✅ (Web: ${webSent} | Móvil: ${fcmSent})`, 'success');
    } catch (e) {
      console.error(e);
      showToast('Error al enviar', 'error');
    }
  }

  function logout() {
    if (confirm('¿Cerrar sesión de administrador?')) {
      localStorage.removeItem('user');
      localStorage.removeItem('isLoggedIn');
      window.location.href = '/pages/auth/indexInicioDeSesion.html';
    }
  }

  // Expose minimal functions used by inline HTML handlers
  window.filterUsers = filterUsers;
  window.deleteUser = deleteUser;
  window.openModal = openModal;
  window.closeModal = closeModal;
  window.openEditModal = openEditModal;
  window.closeEditModal = closeEditModal;
  window.handleCreate = handleCreate;
  window.handleEdit = handleEdit;
  window.quickSetPlan = quickSetPlan;
  window.triggerBulkCreate = triggerBulkCreate;
  window.triggerBulkPlanUpdate = triggerBulkPlanUpdate;
  window.requestPermission = requestPermission;
  window.sendDashboardNotification = sendDashboardNotification;
  window.logout = logout;

  document.addEventListener('DOMContentLoaded', () => {
    if (!requireAdminSessionOrRedirect()) return;

    loadStoredDaysWindow();
    bindRangeSelect();
    syncRangeUI();
    bindBulkInputs();

    fetchUsers();

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch((err) => console.error('SW Fail', err));
    }
  });
})();
