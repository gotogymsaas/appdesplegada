// adminDashboard.js — Panel Admin (GoToGym)
// Objetivo: mantener UX existente, agregar diagramas de tiempo y facilitar edición.
// Depende de: theme.css (variables), config.js (API_URL, authFetch).

(() => {
  const DAYS_WINDOW = 30;

  const els = {
    kpiTotal: () => document.getElementById('kpi-total'),
    kpiPremium: () => document.getElementById('kpi-premium'),
    kpiToday: () => document.getElementById('kpi-today'),
    usersBody: () => document.getElementById('usersBody'),
    searchInput: () => document.getElementById('searchInput'),

    globalChart: () => document.getElementById('globalChart'),
    signupsChart: () => document.getElementById('signupsChart'),
    planChart: () => document.getElementById('planChart'),

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

  let globalChartInstance = null;
  let signupsChartInstance = null;
  let planChartInstance = null;

  function getThemeVars() {
    const styles = getComputedStyle(document.documentElement);
    const primary = (styles.getPropertyValue('--primary') || '').trim() || '#0FBFB0';
    const secondary = (styles.getPropertyValue('--secondary') || '').trim() || '#D4B46A';
    const textMuted = (styles.getPropertyValue('--text-muted') || '').trim() || '#AAAAAA';
    return { primary, secondary, textMuted };
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

  function countSignupsByDay(users, labels) {
    const counts = Object.fromEntries(labels.map((l) => [l, 0]));
    users.forEach((u) => {
      if (!u || !u.date_joined) return;
      try {
        const day = new Date(u.date_joined).toISOString().slice(0, 10);
        if (counts[day] !== undefined) counts[day] += 1;
      } catch (e) {
        // ignore
      }
    });
    return labels.map((l) => counts[l] || 0);
  }

  function loadUserCharts(users) {
    const { primary, secondary, textMuted } = getThemeVars();

    const labels = lastNDaysLabels(DAYS_WINDOW);
    const signups = countSignupsByDay(users, labels);

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
              label: 'Altas',
              data: signups,
              borderColor: primary,
              backgroundColor: 'rgba(15, 191, 176, 0.15)',
              borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: {
            y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' } },
            x: { grid: { display: false }, ticks: { maxTicksLimit: 6, color: textMuted } },
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
      const res = await authFetch(API_URL + 'stats/global_history/');
      const data = await res.json();
      if (!data || data.length === 0) return;

      const labels = data.map((d) => d.date);
      const values = data.map((d) => d.value);

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
            x: { grid: { display: false } },
          },
        },
      });
    } catch (e) {
      console.error('Chart error', e);
    }
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
      renderTable(allUsers);
      calculateKPIs(allUsers);
      loadUserCharts(allUsers);
      await loadGlobalChart();
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

    fetchUsers();

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('../../sw.js').catch((err) => console.error('SW Fail', err));
    }
  });
})();
