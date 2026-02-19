// adminDashboard.js — Panel Admin (GoToGym)
// Objetivo: mantener UX existente, agregar diagramas de tiempo y facilitar edición.
// Depende de: theme.css (variables), config.js (API_URL, authFetch).

(() => {
  const DEFAULT_DAYS_WINDOW = 30;
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
    if (v === 7 || v === 30 || v === 90) return v;
    return DEFAULT_DAYS_WINDOW;
  }

  function rangeLabelText(days) {
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
    const permission = await Notification.requestPermission();
    if (permission === 'granted') {
      showToast('Permisos activados ✅');
      return true;
    }
    showToast('Permiso denegado ❌', 'error');
    return false;
  }

  async function sendDashboardNotification() {
    if (!('serviceWorker' in navigator)) {
      showToast('❌ Las alertas solo funcionan en localhost o HTTPS (no en IP numérica)', 'error');
      return;
    }

    if (Notification.permission !== 'granted') {
      const ok = await requestPermission();
      if (!ok) {
        showToast('🚫 Permiso denegado', 'error');
        return;
      }
    }

    try {
      const variable = document.getElementById('notifVariable').value;
      const reg = await navigator.serviceWorker.ready;

      const questions = {
        s_sleep: '¿Cómo dormiste?',
        s_stress_inv: '¿Cómo está tu estrés?',
        s_energy: '¿Cómo está tu energía?',
      };
      const questionBody = questions[variable] || '¿Cómo te sientes?';

      const options = {
        body: questionBody,
        icon: '../../assets/images/recurso-14.png',
        actions: [
          { action: `${variable}-2`, title: 'Mal (2)' },
          { action: `${variable}-10`, title: 'Bien (10)' },
        ],
      };

      reg.showNotification('GoToGym Alerta 🔔', options);
      showToast('Notificación enviada');
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
  window.requestPermission = requestPermission;
  window.sendDashboardNotification = sendDashboardNotification;
  window.logout = logout;

  document.addEventListener('DOMContentLoaded', () => {
    if (!requireAdminSessionOrRedirect()) return;

    loadStoredDaysWindow();
    bindRangeSelect();
    syncRangeUI();

    fetchUsers();

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('../../sw.js').catch((err) => console.error('SW Fail', err));
    }
  });
})();
