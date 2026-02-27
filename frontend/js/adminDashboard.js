// adminDashboard.js — Panel Admin (GoToGym)
// Objetivo: mantener UX existente, agregar diagramas de tiempo y facilitar edición.
// Depende de: theme.css (variables), config.js (API_URL, authFetch).

(() => {
  const DEFAULT_DAYS_WINDOW = 90;
  const RANGE_STORAGE_KEY = 'adminDashboardRangeDays';
  const COMPARE_STORAGE_KEY = 'adminDashboardCompareEnabled';
  const pageMode = (document.body?.dataset?.adminPage || 'metrics').trim().toLowerCase();

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

    compareToggle: () => document.getElementById('compareToggle'),
    lastUpdated: () => document.getElementById('lastUpdated') || document.getElementById('adminUpdatedAt'),

    kpiActiveUsers7d: () => document.getElementById('kpi-active-users-7d'),
    kpiRangeSignups: () => document.getElementById('kpi-range-signups'),
    kpiRangePremiumSignups: () => document.getElementById('kpi-range-premium-signups'),
    kpiRangeConversion: () => document.getElementById('kpi-range-conversion'),

    deltaActiveUsers7d: () => document.getElementById('delta-active-users-7d'),
    deltaRangeSignups: () => document.getElementById('delta-range-signups'),
    deltaRangePremiumSignups: () => document.getElementById('delta-range-premium-signups'),
    deltaRangeConversion: () => document.getElementById('delta-range-conversion'),

    kpiMrrCurrent: () => document.getElementById('kpi-mrr-current'),
    kpiMrrCurrentSub: () => document.getElementById('kpi-mrr-current-sub'),
    kpiMrrProjected30d: () => document.getElementById('kpi-mrr-projected-30d'),
    kpiMrrProjected30dSub: () => document.getElementById('kpi-mrr-projected-30d-sub'),

    mrrHistoryChart: () => document.getElementById('mrrHistoryChart'),
    mrrProjectionChart: () => document.getElementById('mrrProjectionChart'),

    opsExperienceChart: () => document.getElementById('opsExperienceChart'),
    opsReqTotal: () => document.getElementById('ops-req-total'),
    opsSuccessRate: () => document.getElementById('ops-success-rate'),
    opsAvgLatency: () => document.getElementById('ops-avg-latency'),
    opsTokensTotal: () => document.getElementById('ops-tokens-total'),
    opsCostCop: () => document.getElementById('ops-cost-cop'),
    opsCostPerUser: () => document.getElementById('ops-cost-per-user'),
    opsUsersRange: () => document.getElementById('ops-users-range'),
    experienceRefBody: () => document.getElementById('experienceRefBody'),

    bulkCreateInput: () => document.getElementById('bulkCreateInput'),
    bulkPlanInput: () => document.getElementById('bulkPlanInput'),

    auditBody: () => document.getElementById('auditBody'),

    createModal: () => document.getElementById('createModal'),
    editModal: () => document.getElementById('editModal'),

    editUserId: () => document.getElementById('editUserId'),
    editUsername: () => document.getElementById('editUsername'),
    editEmail: () => document.getElementById('editEmail'),
    editPlan: () => document.getElementById('editPlan'),

    adminMenuBtn: () => document.getElementById('adminMenuBtn'),
    adminMenuPanel: () => document.getElementById('adminMenuPanel'),
  };

  const authFetch = window.authFetch;
  const API_URL = window.API_URL;

  let allUsers = [];
  const selectedUserIds = new Set();

  let overviewCache = null;
  let signupsSeriesCache = null;
  let opsMetricsCache = null;

  let auditCache = null;
  let activeSegment = 'all';

  let daysWindow = DEFAULT_DAYS_WINDOW;
  let compareEnabled = true;
  let globalHistoryCache = null;

  let globalChartInstance = null;
  let signupsChartInstance = null;
  let signupsTrendChartInstance = null;
  let planChartInstance = null;
  let usersCumulativeChartInstance = null;
  let premiumShareChartInstance = null;

  let mrrHistoryChartInstance = null;
  let mrrProjectionChartInstance = null;
  let opsExperienceChartInstance = null;

  const ACTIVE_MAX_DAYS = 3;
  const LOW_ACTIVITY_MAX_DAYS = 14;

  const EXPERIENCE_ENDPOINT_CATALOG = [
    { key: 'exp-001_calories', code: 'Exp-001', label: 'Calorías Inteligentes', endpoint: '/api/chat/', support: '/api/upload_chat_attachment/' },
    { key: 'exp-002_meal_coherence', code: 'Exp-002', label: 'Coherencia Nutricional', endpoint: '/api/qaf/meal_coherence/', support: '/api/chat/' },
    { key: 'exp-003_metabolic_profile', code: 'Exp-003', label: 'Perfil Metabólico', endpoint: '/api/qaf/metabolic_profile/', support: '/api/chat/' },
    { key: 'exp-004_meal_plan', code: 'Exp-004', label: 'Menú Semanal', endpoint: '/api/qaf/meal_plan/', support: '/api/qaf/meal_plan/mutate/' },
    { key: 'exp-005_body_trend', code: 'Exp-005', label: 'Tendencia 6 Semanas', endpoint: '/api/qaf/body_trend/', support: '/api/chat/' },
    { key: 'exp-006_posture', code: 'Exp-006', label: 'Postura Correctiva', endpoint: '/api/qaf/posture/', support: '/api/upload_chat_attachment/' },
    { key: 'exp-007_lifestyle', code: 'Exp-007', label: 'Lifestyle', endpoint: '/api/qaf/lifestyle/', support: '/api/chat/' },
    { key: 'exp-008_motivation', code: 'Exp-008', label: 'Motivación', endpoint: '/api/qaf/motivation/', support: '/api/chat/' },
    { key: 'exp-009_progression', code: 'Exp-009', label: 'Progresión', endpoint: '/api/qaf/progression/', support: '/api/chat/' },
    { key: 'exp-010_muscle_measure', code: 'Exp-010', label: 'Medición Muscular', endpoint: '/api/qaf/muscle_measure/', support: '/api/chat/' },
    { key: 'exp-011_skin_health', code: 'Exp-011', label: 'Salud de Piel', endpoint: '/api/qaf/skin_health/', support: '/api/chat/' },
    { key: 'exp-012_shape_presence', code: 'Exp-012', label: 'Shape Presence', endpoint: '/api/qaf/shape_presence/', support: '/api/chat/' },
    { key: 'exp-013_body_architecture', code: 'Exp-013', label: 'Posture Proportion', endpoint: '/api/qaf/posture_proportion/', support: '/api/chat/' },
  ];

  function isPage(mode) {
    return pageMode === mode;
  }

  function getPremiumPriceCOP() {
    const override =
      window.GTG_PREMIUM_PRICE_COP ||
      localStorage.getItem('gtg_premium_price_cop') ||
      localStorage.getItem('premium_price_cop');
    const v = Number(String(override || '').replace(/[^0-9.]/g, ''));
    if (Number.isFinite(v) && v > 0) return v;
    // Fuente: frontend/pages/plans/Planes.html (Mensual: $20.900 COP / mes)
    return 20900;
  }

  function formatCOP(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '--';
    try {
      return n.toLocaleString('es-CO', { style: 'currency', currency: 'COP', maximumFractionDigits: 0 });
    } catch (e) {
      return `$${Math.round(n).toLocaleString('es-CO')} COP`;
    }
  }

  function computePremiumSignupsSeries() {
    const rows = signupsSeriesCache && signupsSeriesCache.data;
    if (!Array.isArray(rows) || rows.length === 0) return null;
    const labels = rows.map((r) => r.date);
    const premium = rows.map((r) => Number(r.premium || 0));
    return { labels, premium };
  }

  function movingAverage(values, windowSize) {
    const w = Math.max(1, Number(windowSize) || 1);
    const out = [];
    for (let i = 0; i < values.length; i++) {
      const start = Math.max(0, i - w + 1);
      const slice = values.slice(start, i + 1);
      const sum = slice.reduce((a, b) => a + (Number(b) || 0), 0);
      out.push(slice.length ? sum / slice.length : 0);
    }
    return out;
  }

  function addDaysISO(iso, days) {
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      d.setDate(d.getDate() + days);
      return d.toISOString().slice(0, 10);
    } catch (e) {
      return iso;
    }
  }

  function buildEmptyOpsMetrics(days) {
    const safeDays = Math.max(1, Number(days) || DEFAULT_DAYS_WINDOW);
    const start = new Date();
    start.setHours(0, 0, 0, 0);
    start.setDate(start.getDate() - (safeDays - 1));

    const experiences = EXPERIENCE_ENDPOINT_CATALOG.map((item) => ({
      key: item.key,
      label: item.code,
    }));

    const series = [];
    for (let i = 0; i < safeDays; i += 1) {
      const day = new Date(start);
      day.setDate(start.getDate() + i);
      const row = {
        date: day.toISOString().slice(0, 10),
        total: 0,
      };
      experiences.forEach((exp) => {
        row[exp.key] = 0;
      });
      series.push(row);
    }

    const activeUsersRef = Number(overviewCache?.data?.active_users_7d || 0);
    return {
      data: {
        experiences,
        series,
        benchmark: {
          requests_total: 0,
          requests_success: 0,
          requests_error: 0,
          success_rate: 0,
          error_rate: 0,
          avg_latency_ms_total: 0,
          avg_latency_ms_n8n: 0,
          avg_latency_ms_qaf: 0,
          tokens_in_total: 0,
          tokens_out_total: 0,
        },
        costs: {
          estimated_total_usd: 0,
          estimated_total_cop: 0,
          active_users_range: activeUsersRef,
          cost_per_active_user_cop: 0,
          model: {
            input_cost_per_1k_usd: 0,
            output_cost_per_1k_usd: 0,
            usd_to_cop: 4000,
          },
        },
      },
      meta: {
        dateFrom: series[0]?.date || '',
        dateTo: series[series.length - 1]?.date || '',
        timezone: 'America/Bogota',
        days: safeDays,
        source: 'fallback',
      },
    };
  }

  function renderRevenueAndProjections() {
    if (!overviewCache) return;
    const d = overviewCache.data || {};
    const premiumActiveNow = Number(d.premium_active || 0);
    const price = getPremiumPriceCOP();

    const mrrBackend = Number(d.mrr_current_cop);
    const hasBackendMrr = Number.isFinite(mrrBackend) && mrrBackend >= 0;
    const mrrSource = String(d.mrr_source || '').trim().toLowerCase();
    const mrrActiveSubscriptions = Number(d.mrr_active_subscriptions || 0);

    const mrrNow = hasBackendMrr ? mrrBackend : 0;
    if (els.kpiMrrCurrent()) els.kpiMrrCurrent().innerText = formatCOP(mrrNow);
    if (els.kpiMrrCurrentSub()) {
      if (hasBackendMrr && mrrSource === 'mercadopago') {
        const n = Number.isFinite(mrrActiveSubscriptions) ? mrrActiveSubscriptions : 0;
        els.kpiMrrCurrentSub().innerText = `${n} suscripciones activas en Mercado Pago (valor real)`;
      } else {
        els.kpiMrrCurrentSub().innerText = 'Sin datos de cobro en Mercado Pago';
      }
    }

    const series = computePremiumSignupsSeries();
    if (!series) return;

    const sumPremiumSignups = series.premium.reduce((a, b) => a + (Number(b) || 0), 0);
    const premiumAtStart = Math.max(0, premiumActiveNow - sumPremiumSignups);

    // Histórico MRR estimado dentro del rango
    let runningPremium = premiumAtStart;
    const premiumActiveEst = [];
    for (const p of series.premium) {
      runningPremium += Number(p) || 0;
      premiumActiveEst.push(runningPremium);
    }
    const mrrHistory = premiumActiveEst.map((n) => n * price);

    // Proyección MRR 30 días: usa promedio móvil (últimos 14 días o menos)
    const tailWindow = Math.min(14, series.premium.length);
    const tail = series.premium.slice(-tailWindow);
    const avgPremiumPerDay = tailWindow ? tail.reduce((a, b) => a + (Number(b) || 0), 0) / tailWindow : 0;

    const daysForecast = 30;
    const lastLabel = series.labels[series.labels.length - 1] || new Date().toISOString().slice(0, 10);
    const projLabels = [];
    const projMrr = [];

    for (let i = 1; i <= daysForecast; i++) {
      projLabels.push(addDaysISO(lastLabel, i));
      const premiumProjected = premiumActiveNow + avgPremiumPerDay * i;
      projMrr.push(premiumProjected * price);
    }

    const mrrProjected30d = projMrr[projMrr.length - 1] || mrrNow;
    if (els.kpiMrrProjected30d()) els.kpiMrrProjected30d().innerText = formatCOP(mrrProjected30d);
    if (els.kpiMrrProjected30dSub()) {
      els.kpiMrrProjected30dSub().innerText = `Altas Premium/día (prom. ${tailWindow}d): ${avgPremiumPerDay.toFixed(2)} | Precio: ${formatCOP(price)}/mes`;
    }

    // Charts
    const { secondary, primary, textMuted } = getThemeVars();

    const histEl = els.mrrHistoryChart();
    if (histEl && window.Chart) {
      const ctx = histEl.getContext('2d');
      if (mrrHistoryChartInstance) mrrHistoryChartInstance.destroy();
      mrrHistoryChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: series.labels,
          datasets: [
            {
              label: 'MRR estimado',
              data: mrrHistory,
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
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx2) => ` ${formatCOP(ctx2.parsed.y)}`,
              },
            },
          },
          scales: {
            y: {
              beginAtZero: true,
              grid: { color: 'rgba(255,255,255,0.1)' },
              ticks: { color: textMuted, callback: (v) => formatCOP(v) },
            },
            x: { grid: { display: false }, ticks: { maxTicksLimit: 7, color: textMuted } },
          },
        },
      });
    }

    const projEl = els.mrrProjectionChart();
    if (projEl && window.Chart) {
      const ctx = projEl.getContext('2d');
      if (mrrProjectionChartInstance) mrrProjectionChartInstance.destroy();
      mrrProjectionChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: projLabels,
          datasets: [
            {
              label: 'MRR proyectado',
              data: projMrr,
              borderColor: primary,
              backgroundColor: 'rgba(15, 191, 176, 0.10)',
              borderWidth: 2,
              tension: 0.25,
              fill: true,
              pointRadius: 0,
              borderDash: [6, 6],
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label: (ctx2) => ` ${formatCOP(ctx2.parsed.y)}`,
              },
            },
          },
          scales: {
            y: {
              beginAtZero: true,
              grid: { color: 'rgba(255,255,255,0.1)' },
              ticks: { color: textMuted, callback: (v) => formatCOP(v) },
            },
            x: { grid: { display: false }, ticks: { maxTicksLimit: 7, color: textMuted } },
          },
        },
      });
    }
  }

  function getThemeVars() {
    const styles = getComputedStyle(document.documentElement);
    const primary = (styles.getPropertyValue('--primary') || '').trim() || '#0FBFB0';
    const secondary = (styles.getPropertyValue('--secondary') || '').trim() || '#D4B46A';
    const textMuted = (styles.getPropertyValue('--text-muted') || '').trim() || '#AAAAAA';
    return { primary, secondary, textMuted };
  }

  function bindAdminMenu() {
    const btn = els.adminMenuBtn();
    const panel = els.adminMenuPanel();
    if (!btn || !panel) return;

    const close = () => panel.classList.remove('open');
    const toggle = () => panel.classList.toggle('open');

    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      toggle();
    });

    document.addEventListener('click', (e) => {
      if (!panel.contains(e.target) && !btn.contains(e.target)) close();
    });

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') close();
    });
  }

  function renderExperienceReference() {
    const tbody = els.experienceRefBody();
    if (!tbody) return;

    tbody.innerHTML = '';
    EXPERIENCE_ENDPOINT_CATALOG.forEach((item) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${item.code}</td>
        <td>${item.label}</td>
        <td><code>${item.endpoint}</code></td>
        <td><code>${item.support}</code></td>
      `;
      tbody.appendChild(tr);
    });
  }

  function renderOpsMetrics() {
    if (!opsMetricsCache) {
      opsMetricsCache = buildEmptyOpsMetrics(daysWindow);
    }
    const d = opsMetricsCache.data || {};
    const benchmark = d.benchmark || {};
    const costs = d.costs || {};
    const series = Array.isArray(d.series) ? d.series : [];
    const experiences = Array.isArray(d.experiences) ? d.experiences : [];
    const byKey = Object.fromEntries(EXPERIENCE_ENDPOINT_CATALOG.map((item) => [item.key, item]));

    if (els.opsReqTotal()) els.opsReqTotal().innerText = String(benchmark.requests_total ?? '--');
    if (els.opsSuccessRate()) {
      const v = Number(benchmark.success_rate);
      els.opsSuccessRate().innerText = Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '--';
    }
    if (els.opsAvgLatency()) {
      const v = Number(benchmark.avg_latency_ms_total);
      els.opsAvgLatency().innerText = Number.isFinite(v) ? `${Math.round(v)} ms` : '0 ms';
    }
    if (els.opsTokensTotal()) {
      const tIn = Number(benchmark.tokens_in_total || 0);
      const tOut = Number(benchmark.tokens_out_total || 0);
      els.opsTokensTotal().innerText = `${(tIn + tOut).toLocaleString('es-CO')}`;
    }
    if (els.opsCostCop()) {
      const v = Number(costs.estimated_total_cop);
      els.opsCostCop().innerText = Number.isFinite(v) ? formatCOP(v) : formatCOP(0);
    }
    if (els.opsCostPerUser()) {
      const v = Number(costs.cost_per_active_user_cop);
      els.opsCostPerUser().innerText = Number.isFinite(v) ? formatCOP(v) : formatCOP(0);
    }
    if (els.opsUsersRange()) {
      els.opsUsersRange().innerText = String(costs.active_users_range ?? '--');
    }

    const chartEl = els.opsExperienceChart();
    if (!chartEl || !window.Chart || !series.length) return;

    const labels = series.map((r) => r.date);
    const { primary, textMuted } = getThemeVars();
    const basePalette = [
      '#0FBFB0', '#D4B46A', '#8E8EFF', '#F28B82', '#81C995', '#FDD663', '#A7FFEB',
      '#FFAB91', '#B39DDB', '#80DEEA', '#FFCC80', '#90CAF9', '#CE93D8',
    ];

    const datasets = [];
    datasets.push({
      label: 'Total actividad',
      data: series.map((r) => Number(r.total || 0)),
      borderColor: primary,
      backgroundColor: 'rgba(15, 191, 176, 0.10)',
      borderWidth: 2,
      tension: 0.25,
      fill: true,
      pointRadius: 0,
    });

    experiences.forEach((exp, idx) => {
      const key = exp && exp.key ? exp.key : null;
      if (!key) return;
      const catalog = byKey[key] || null;
      const datasetLabel = isPage('experiences')
        ? `${exp.label || key} · ${(catalog && catalog.endpoint) ? catalog.endpoint : '/api/chat/'}`
        : (exp.label || key);
      datasets.push({
        label: datasetLabel,
        data: series.map((r) => Number(r[key] || 0)),
        borderColor: basePalette[idx % basePalette.length],
        borderWidth: 1.5,
        tension: 0.25,
        fill: false,
        pointRadius: 0,
      });
    });

    const ctx = chartEl.getContext('2d');
    if (opsExperienceChartInstance) opsExperienceChartInstance.destroy();
    opsExperienceChartInstance = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        plugins: {
          legend: { position: 'bottom', labels: { color: textMuted, boxWidth: 10 } },
        },
        scales: {
          y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: textMuted } },
          x: { grid: { display: false }, ticks: { maxTicksLimit: 8, color: textMuted } },
        },
      },
    });
  }

  function clampDaysWindow(value) {
    const v = Number(value);
    if (v === 1 || v === 7 || v === 30 || v === 90 || v === 180 || v === 270 || v === 365 || v === 730) return v;
    return DEFAULT_DAYS_WINDOW;
  }

  function rangeLabelText(days) {
    if (days === 1) return 'último día';
    if (days === 7) return 'última semana';
    if (days === 30) return 'último mes';
    if (days === 730) return 'últimos 2 años';
    return `últimos ${days} días`;
  }

  function syncRangeUI() {
    const labelEl = els.rangeLabel();
    if (labelEl) labelEl.textContent = rangeLabelText(daysWindow);

    document.querySelectorAll('[data-range-chip]').forEach((el) => {
      if (daysWindow === 1) el.textContent = 'Último día';
      else if (daysWindow === 7) el.textContent = 'Última semana';
      else if (daysWindow === 30) el.textContent = 'Último mes';
      else if (daysWindow === 730) el.textContent = 'Últimos 2 años';
      else el.textContent = `Últimos ${daysWindow} días`;
    });

    const selectEl = els.rangeSelect();
    if (selectEl) selectEl.value = String(daysWindow);
  }

  function loadStoredCompareEnabled() {
    const raw = localStorage.getItem(COMPARE_STORAGE_KEY);
    if (raw === null || raw === undefined || raw === '') {
      compareEnabled = true;
      return;
    }
    compareEnabled = raw === 'true';
  }

  function setCompareEnabled(next) {
    compareEnabled = !!next;
    localStorage.setItem(COMPARE_STORAGE_KEY, String(compareEnabled));

    const toggleEl = els.compareToggle();
    if (toggleEl) toggleEl.checked = compareEnabled;

    overviewCache = null;
    fetchUsers();
  }

  function bindCompareToggle() {
    const toggleEl = els.compareToggle();
    if (!toggleEl) return;
    toggleEl.addEventListener('change', (e) => {
      setCompareEnabled(!!e.target.checked);
    });
  }

  function setLastUpdatedNow() {
    const el = els.lastUpdated();
    if (!el) return;
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    el.textContent = `Actualizado: ${hh}:${mm}`;
  }

  function formatDeltaText(deltaObj, { isPercentPoints = false } = {}) {
    if (!deltaObj || typeof deltaObj !== 'object') return '—';

    const abs = deltaObj.abs;
    const pct = deltaObj.pct;
    if (abs === null || abs === undefined) return '—';

    const absNum = Number(abs);
    if (Number.isNaN(absNum)) return '—';

    const sign = absNum > 0 ? '+' : '';

    let absText;
    if (isPercentPoints) {
      absText = `${sign}${(absNum * 100).toFixed(1)} pp`;
    } else {
      absText = `${sign}${Math.round(absNum)}`;
    }

    let pctText = '';
    if (pct !== null && pct !== undefined) {
      const pctNum = Number(pct);
      if (!Number.isNaN(pctNum)) {
        const pctSign = pctNum > 0 ? '+' : '';
        pctText = ` (${pctSign}${pctNum.toFixed(1)}%)`;
      }
    }

    return `${absText}${pctText} vs período anterior`;
  }

  function applyDeltaClass(el, deltaObj) {
    if (!el) return;
    el.classList.remove('delta-positive');
    el.classList.remove('delta-negative');
    const abs = deltaObj && typeof deltaObj === 'object' ? Number(deltaObj.abs) : NaN;
    if (Number.isNaN(abs) || abs === 0) return;
    el.classList.add(abs > 0 ? 'delta-positive' : 'delta-negative');
  }

  function renderPeriodComparatives() {
    if (!overviewCache) return;
    const d = overviewCache.data || {};
    const m = overviewCache.meta || {};

    if (els.kpiActiveUsers7d()) els.kpiActiveUsers7d().innerText = String(d.active_users_7d ?? '--');
    if (els.kpiRangeSignups()) els.kpiRangeSignups().innerText = String(d.signups_total ?? '--');
    if (els.kpiRangePremiumSignups()) els.kpiRangePremiumSignups().innerText = String(d.signups_premium ?? '--');
    if (els.kpiRangeConversion()) {
      const conv = Number(d.conversion_premium);
      els.kpiRangeConversion().innerText = Number.isFinite(conv) ? `${(conv * 100).toFixed(1)}%` : '--';
    }

    if (els.deltaActiveUsers7d()) {
      els.deltaActiveUsers7d().innerText = 'Fuente: last_login';
    }

    if (!compareEnabled || !m.compare || !m.deltas) {
      if (els.deltaRangeSignups()) els.deltaRangeSignups().innerText = 'Comparación desactivada';
      if (els.deltaRangePremiumSignups()) els.deltaRangePremiumSignups().innerText = 'Comparación desactivada';
      if (els.deltaRangeConversion()) els.deltaRangeConversion().innerText = 'Comparación desactivada';
      return;
    }

    const deltas = m.deltas || {};

    if (els.deltaRangeSignups()) {
      const el = els.deltaRangeSignups();
      el.innerText = formatDeltaText(deltas.signups_total);
      applyDeltaClass(el, deltas.signups_total);
    }
    if (els.deltaRangePremiumSignups()) {
      const el = els.deltaRangePremiumSignups();
      el.innerText = formatDeltaText(deltas.signups_premium);
      applyDeltaClass(el, deltas.signups_premium);
    }
    if (els.deltaRangeConversion()) {
      const el = els.deltaRangeConversion();
      el.innerText = formatDeltaText(deltas.conversion_premium, { isPercentPoints: true });
      applyDeltaClass(el, deltas.conversion_premium);
    }
  }

  function showToast(message, type = 'success') {
    const t = document.getElementById('toast');
    if (!t) return;
    t.textContent = message;
    t.className = `toast ${type}`;
    t.style.display = 'block';
    setTimeout(() => (t.style.display = 'none'), 3000);
  }

  async function readApiError(res, fallbackMessage) {
    try {
      const payload = await res.json();
      if (payload && typeof payload === 'object') {
        const msg = payload.error || payload.detail || payload.message;
        if (msg) return String(msg);
      }
    } catch (e) {
      // ignore
    }
    return fallbackMessage;
  }

  function daysSinceDate(dateStr) {
    if (!dateStr) return null;
    try {
      const dateValue = new Date(dateStr);
      if (Number.isNaN(dateValue.getTime())) return null;
      const now = new Date();
      return Math.floor((now.getTime() - dateValue.getTime()) / (1000 * 60 * 60 * 24));
    } catch (e) {
      return null;
    }
  }

  function classifyUserActivity(user) {
    if (!user) {
      return {
        key: 'unknown',
        label: '—',
        riskLevel: 'med',
        score: 50,
        lastDays: null,
      };
    }

    if (user.is_active === false) {
      return {
        key: 'inactive',
        label: 'Inactivo',
        riskLevel: 'high',
        score: 100,
        lastDays: daysSinceDate(user.last_login),
      };
    }

    const lastDays = daysSinceDate(user.last_login);
    if (lastDays === null) {
      return {
        key: 'inactive',
        label: 'Inactivo',
        riskLevel: 'high',
        score: 90,
        lastDays: null,
      };
    }

    if (lastDays <= ACTIVE_MAX_DAYS) {
      return {
        key: 'active',
        label: 'Activo',
        riskLevel: 'low',
        score: 10,
        lastDays,
      };
    }

    if (lastDays <= LOW_ACTIVITY_MAX_DAYS) {
      return {
        key: 'low_activity',
        label: 'Poca actividad',
        riskLevel: 'med',
        score: 55,
        lastDays,
      };
    }

    return {
      key: 'inactive',
      label: 'Inactivo',
      riskLevel: 'high',
      score: 85,
      lastDays,
    };
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

    const formatLastActivity = (user) => {
      if (!user?.last_login) return '—';
      try {
        const d = new Date(user.last_login);
        if (Number.isNaN(d.getTime())) return '—';
        return d.toLocaleDateString();
      } catch (e) {
        return '—';
      }
    };

    users.forEach((user) => {
      const row = document.createElement('tr');
      const date = user.date_joined ? new Date(user.date_joined).toLocaleDateString() : '-';
      const badgeClass = user.plan === 'Premium' ? 'badge-premium' : 'badge-free';

      const state = classifyUserActivity(user);
      const lastAct = formatLastActivity(user);
      const risk = classifyUserActivity(user);
      const riskClass = risk.riskLevel === 'high' ? 'risk-badge risk-high' : (risk.riskLevel === 'med' ? 'risk-badge risk-med' : 'risk-badge risk-low');
      const riskTooltip = risk.lastDays === null
        ? `Estado: ${risk.label}`
        : `Estado: ${risk.label} | Última actividad: ${risk.lastDays} día(s)`;

      const checked = selectedUserIds.has(user.id) ? 'checked' : '';

      const nextPlan = user.plan === 'Premium' ? 'Gratis' : 'Premium';
      const planChecked = user.plan === 'Premium' ? 'checked' : '';
      row.innerHTML = `
        <td><input class="row-select" type="checkbox" data-user-id="${user.id}" ${checked} /></td>
        <td>#${user.id}</td>
        <td style="font-weight: bold; color: var(--text-main);">${user.username}</td>
        <td style="color: var(--text-muted);">${user.email}</td>
        <td>
          <div class="plan-toggle-wrap">
            <label class="switch" title="Cambiar plan con confirmación">
              <input class="plan-toggle" type="checkbox" data-user-id="${user.id}" ${planChecked} />
              <span class="slider"></span>
            </label>
            <span class="plan-label">${user.plan}</span>
          </div>
        </td>
        <td>${state.label}</td>
        <td>${lastAct}</td>
        <td><span class="${riskClass}" title="${riskTooltip}">${risk.label}</span></td>
        <td>${date}</td>
        <td class="actions-cell">
          <button class="row-menu-btn" type="button" data-menu-btn="${user.id}" aria-label="Acciones">⋮</button>
          <div class="row-menu" data-row-menu="${user.id}">
            <button type="button" data-action="profile" data-user-id="${user.id}">Ver perfil</button>
            <button type="button" data-action="plan" data-user-id="${user.id}">Cambiar plan</button>
            <button type="button" data-action="toggle_active" data-user-id="${user.id}">${user.is_active === false ? 'Reactivar' : 'Suspender'}</button>
            <button type="button" class="danger" data-action="delete" data-user-id="${user.id}">Eliminar</button>
          </div>
        </td>
      `;

      tbody.appendChild(row);
    });

    bindRowSelectionHandlers();
    bindPlanToggles();
    bindRowMenus();
    syncSelectionUI(users);
  }

  function closeAllRowMenus() {
    document.querySelectorAll('.row-menu.open').forEach((m) => m.classList.remove('open'));
  }

  function bindRowMenus() {
    document.querySelectorAll('button[data-menu-btn]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-menu-btn');
        const menu = document.querySelector(`.row-menu[data-row-menu="${id}"]`);
        if (!menu) return;
        const isOpen = menu.classList.contains('open');
        closeAllRowMenus();
        if (!isOpen) menu.classList.add('open');
      });
    });

    document.querySelectorAll('.row-menu button[data-action]').forEach((b) => {
      b.addEventListener('click', async (e) => {
        e.stopPropagation();
        const action = b.getAttribute('data-action');
        const userId = Number(b.getAttribute('data-user-id'));
        closeAllRowMenus();

        if (action === 'profile') {
          const u = allUsers.find((x) => x && x.id === userId);
          if (u) window.openEditModal(u);
          return;
        }

        if (action === 'plan') {
          const u = allUsers.find((x) => x && x.id === userId);
          if (!u) return;
          const nextPlan = u.plan === 'Premium' ? 'Gratis' : 'Premium';
          await setUserPlan(userId, nextPlan);
          return;
        }

        if (action === 'toggle_active') {
          const u = allUsers.find((x) => x && x.id === userId);
          if (!u) return;
          const nextActive = u.is_active === false;
          await setUserActive(userId, nextActive);
          return;
        }

        if (action === 'delete') {
          await window.deleteUser(userId);
        }
      });
    });
  }

  async function setUserPlan(userId, plan) {
    const user = allUsers.find((u) => u && u.id === userId);
    if (!user) {
      showToast('Usuario no encontrado', 'error');
      return false;
    }
    const pretty = plan === 'Premium' ? 'Premium' : 'Gratis';
    if (!confirm(`¿Cambiar plan a ${pretty} para ${user.email}?`)) return false;
    const reason = (prompt('Motivo (requerido) para cambiar el plan:') || '').trim();
    if (!reason) {
      showToast('Motivo requerido', 'error');
      return false;
    }

    try {
      const res = await authFetch(API_URL + `users/update_admin/${userId}/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user.username, email: user.email, plan: pretty, reason }),
      });
      if (res.ok) {
        showToast('Plan actualizado', 'success');
        await fetchUsers();
        refreshAudit();
        return true;
      }
      showToast('Error actualizando plan', 'error');
      return false;
    } catch (e) {
      showToast('Error actualizando plan', 'error');
      return false;
    }
  }

  async function setUserActive(userId, isActive) {
    const user = allUsers.find((u) => u && u.id === userId);
    if (!user) {
      showToast('Usuario no encontrado', 'error');
      return false;
    }
    const verb = isActive ? 'Reactivar' : 'Suspender';
    if (!confirm(`¿${verb} a ${user.email}?`)) return false;
    const reason = (prompt(`Motivo (requerido) para ${verb.toLowerCase()}:`) || '').trim();
    if (!reason) {
      showToast('Motivo requerido', 'error');
      return false;
    }

    try {
      const res = await authFetch(API_URL + `users/set_active/${userId}/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_active: !!isActive, reason }),
      });
      if (res.ok) {
        showToast('Estado actualizado', 'success');
        await fetchUsers();
        refreshAudit();
        return true;
      }
      const err = await res.json().catch(() => ({}));
      showToast(err.error || 'Error actualizando estado', 'error');
      return false;
    } catch (e) {
      showToast('Error actualizando estado', 'error');
      return false;
    }
  }

  function bindPlanToggles() {
    document.querySelectorAll('input.plan-toggle[data-user-id]').forEach((cb) => {
      cb.addEventListener('change', async (e) => {
        const id = Number(cb.getAttribute('data-user-id'));
        const user = allUsers.find((u) => u && u.id === id);
        if (!user) return;

        // Mostrar contexto antes de permitir acción
        const last = user.last_login ? new Date(user.last_login).toLocaleDateString() : '—';
        const riskText = (() => {
          try {
            const lastDays = user.last_login ? Math.floor((Date.now() - new Date(user.last_login).getTime()) / (1000 * 60 * 60 * 24)) : null;
            if (user.is_active === false) return 'alto (inactivo)';
            if (lastDays === null) return (user.plan === 'Premium' ? 'alto' : 'medio');
            if (user.plan === 'Premium' && lastDays >= 14) return 'alto';
            if (lastDays >= 7) return 'medio';
            return 'bajo';
          } catch {
            return '—';
          }
        })();

        const nextPlan = cb.checked ? 'Premium' : 'Gratis';
        const ok = await setUserPlan(id, nextPlan);
        if (!ok) {
          // revertir UI
          cb.checked = user.plan === 'Premium';
          showToast(`Acción cancelada (últ. actividad: ${last} | riesgo: ${riskText})`, 'error');
        }
      });
    });

    // cerrar menú al click fuera
    if (!document.body.__gtgRowMenuBound) {
      document.body.__gtgRowMenuBound = true;
      document.addEventListener('click', () => closeAllRowMenus());
      document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeAllRowMenus();
      });
    }
  }

  function setSelectedCount(count) {
    const el = document.getElementById('selectedCount');
    if (!el) return;
    el.textContent = `${count} seleccionados`;
  }

  function getRenderedUserIds(users) {
    return (users || []).map((u) => u && u.id).filter((id) => typeof id === 'number' || typeof id === 'string');
  }

  function syncSelectionUI(renderedUsers) {
    setSelectedCount(selectedUserIds.size);

    const selectAllEl = document.getElementById('selectAllUsers');
    if (!selectAllEl) return;

    const renderedIds = getRenderedUserIds(renderedUsers);
    if (renderedIds.length === 0) {
      selectAllEl.checked = false;
      selectAllEl.indeterminate = false;
      return;
    }

    const selectedInRendered = renderedIds.filter((id) => selectedUserIds.has(Number(id) || id)).length;
    selectAllEl.checked = selectedInRendered === renderedIds.length;
    selectAllEl.indeterminate = selectedInRendered > 0 && selectedInRendered < renderedIds.length;
  }

  function bindRowSelectionHandlers() {
    document.querySelectorAll('input.row-select[data-user-id]').forEach((cb) => {
      cb.addEventListener('change', (e) => {
        const idRaw = e.target.getAttribute('data-user-id');
        const id = Number(idRaw);
        if (e.target.checked) selectedUserIds.add(id);
        else selectedUserIds.delete(id);
        setSelectedCount(selectedUserIds.size);
        syncSelectionUI(getCurrentlyRenderedUsers());
      });
    });
  }

  function getCurrentlyRenderedUsers() {
    const term = (els.searchInput()?.value || '').toLowerCase();
    if (!term) return allUsers;
    return allUsers.filter((u) => u.username.toLowerCase().includes(term) || u.email.toLowerCase().includes(term));
  }

  function bindSelectAll() {
    const selectAllEl = document.getElementById('selectAllUsers');
    if (!selectAllEl) return;
    selectAllEl.addEventListener('change', (e) => {
      const renderedUsers = getCurrentlyRenderedUsers();
      const renderedIds = getRenderedUserIds(renderedUsers).map((x) => Number(x));
      if (e.target.checked) {
        renderedIds.forEach((id) => selectedUserIds.add(id));
      } else {
        renderedIds.forEach((id) => selectedUserIds.delete(id));
      }
      renderTable(renderedUsers);
    });
  }

  async function bulkSetPlan(plan) {
    const ids = Array.from(selectedUserIds.values());
    if (!ids.length) {
      showToast('Selecciona usuarios primero', 'error');
      return;
    }

    const pretty = plan === 'Premium' ? 'Premium' : 'Gratis';
    if (!confirm(`¿Cambiar plan a ${pretty} para ${ids.length} usuario(s)?`)) return;
    const reason = (prompt('Motivo (requerido) para el cambio masivo de plan:') || '').trim();
    if (!reason) {
      showToast('Motivo requerido', 'error');
      return;
    }

    let ok = 0;
    let fail = 0;
    for (const id of ids) {
      const user = allUsers.find((u) => u && u.id === id);
      if (!user) {
        fail += 1;
        continue;
      }
      try {
        const res = await authFetch(API_URL + `users/update_admin/${id}/`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: user.username, email: user.email, plan: pretty, reason }),
        });
        if (res.ok) ok += 1;
        else fail += 1;
      } catch (e) {
        fail += 1;
      }
    }

    showToast(`Planes: ${ok} ok, ${fail} fallidos`, fail ? 'error' : 'success');
    await fetchUsers();
  }

  async function bulkDeleteUsers() {
    const ids = Array.from(selectedUserIds.values());
    if (!ids.length) {
      showToast('Selecciona usuarios primero', 'error');
      return;
    }
    if (!confirm(`¿Eliminar (borrado lógico) ${ids.length} usuario(s)?`)) return;
    const reason = (prompt('Motivo (requerido) para eliminar estos usuarios:') || '').trim();
    if (!reason) {
      showToast('Motivo requerido', 'error');
      return;
    }

    let ok = 0;
    let fail = 0;
    for (const id of ids) {
      try {
        const res = await authFetch(API_URL + `users/delete/${id}/`, {
          method: 'DELETE',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reason }),
        });
        if (res.ok) {
          ok += 1;
        } else {
          fail += 1;
          const errorMessage = await readApiError(res, 'No fue posible eliminar algunos usuarios');
          console.warn('bulkDeleteUsers error', id, errorMessage);
        }
      } catch (e) {
        fail += 1;
      }
    }

    showToast(`Eliminados: ${ok} ok, ${fail} fallidos`, fail ? 'error' : 'success');
    selectedUserIds.clear();
    await fetchUsers();
    await refreshAudit();
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

    let labels = lastNDaysLabels(daysWindow);
    let signups = countSignupsByDay(users, labels);
    let signupsByPlan = countSignupsByDayAndPlan(users, labels);

    if (signupsSeriesCache && Array.isArray(signupsSeriesCache.data) && signupsSeriesCache.data.length) {
      try {
        const rows = signupsSeriesCache.data;
        labels = rows.map((r) => r.date);
        signups = rows.map((r) => Number(r.total || 0));
        signupsByPlan = {
          premium: rows.map((r) => Number(r.premium || 0)),
          free: rows.map((r) => Number(r.free || 0)),
        };
      } catch (e) {
        // fallback
      }
    }

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
    overviewCache = null;
    signupsSeriesCache = null;
    loadPageData();
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
    if (!isPage('notifications')) {
      renderTable(allUsers);
    }

    if (isPage('experiences')) {
      renderExperienceReference();
      renderOpsMetrics();
      setLastUpdatedNow();
      return;
    }

    if (!isPage('metrics')) {
      setLastUpdatedNow();
      return;
    }

    // KPIs y series: preferir backend agregado; si falla, fallback local.
    if (overviewCache) {
      try {
        const d = overviewCache.data || {};
        if (els.kpiTotal()) els.kpiTotal().innerText = String(d.total_users ?? '--');
        if (els.kpiPremium()) {
          const premium = Number(d.premium_active || 0);
          const total = Number(d.total_users || 0);
          const pct = total > 0 ? Math.round((premium / total) * 100) : 0;
          els.kpiPremium().innerText = `${premium} (${pct}%)`;
        }
        if (els.kpiToday()) {
          // No existe KPI hoy en overview agregado; fallback local.
          calculateKPIs(allUsers);
        }
      } catch (e) {
        calculateKPIs(allUsers);
      }
    } else {
      calculateKPIs(allUsers);
    }

    loadUserCharts(allUsers);
    loadGlobalChart();

    renderPeriodComparatives();
    renderRevenueAndProjections();
    renderOpsMetrics();
    setLastUpdatedNow();
  }

  async function fetchOverview() {
    try {
      const res = await authFetch(`${API_URL}admin/dashboard/overview/?days=${daysWindow}&timezone=America/Bogota&compare=${compareEnabled ? 'true' : 'false'}`);
      if (!res.ok) return false;
      overviewCache = await res.json();
      return true;
    } catch (e) {
      return false;
    }
  }

  async function fetchSignupsSeries() {
    try {
      const res = await authFetch(`${API_URL}admin/dashboard/signups_series/?days=${daysWindow}&timezone=America/Bogota`);
      if (!res.ok) return false;
      signupsSeriesCache = await res.json();
      return true;
    } catch (e) {
      return false;
    }
  }

  async function fetchOpsMetrics() {
    try {
      const res = await authFetch(`${API_URL}admin/dashboard/ops_metrics/?days=${daysWindow}&timezone=America/Bogota`);
      if (!res.ok) {
        opsMetricsCache = buildEmptyOpsMetrics(daysWindow);
        return false;
      }
      const payload = await res.json();
      if (!payload || !payload.data || !Array.isArray(payload.data.series)) {
        opsMetricsCache = buildEmptyOpsMetrics(daysWindow);
        return false;
      }
      opsMetricsCache = payload;
      return true;
    } catch (e) {
      opsMetricsCache = buildEmptyOpsMetrics(daysWindow);
      return false;
    }
  }

  async function fetchUsers() {
    try {
      // Paginación (MVP): primer page. Mantiene compatibilidad si backend responde array.
      const res = await authFetch(API_URL + 'users/?page=1&pageSize=200');
      const users = await res.json();
      if (users.error) {
        showToast('Error cargando usuarios', 'error');
        return;
      }
      allUsers = Array.isArray(users) ? users : (users && Array.isArray(users.data) ? users.data : []);

      if (isPage('metrics')) {
        await fetchOverview();
        await fetchSignupsSeries();
        await fetchOpsMetrics();
      }
      renderDashboard();
    } catch (e) {
      console.error(e);
      showToast('Error de conexión', 'error');
    }
  }

  async function loadPageData() {
    if (isPage('metrics')) {
      await fetchUsers();
      return;
    }

    if (isPage('users')) {
      await fetchUsers();
      return;
    }

    if (isPage('experiences')) {
      await fetchOpsMetrics();
      renderDashboard();
      return;
    }

    setLastUpdatedNow();
  }

  function applySegment(segment, { silent = false } = {}) {
    const next = (segment || 'all').trim();
    activeSegment = next;

    const now = new Date();
    const filtered = allUsers.filter((u) => {
      if (!u) return false;
      const state = classifyUserActivity(u);
      if (next === 'all') return true;

      if (next === 'new_7d') {
        const dj = u.date_joined ? new Date(u.date_joined) : null;
        if (!dj || Number.isNaN(dj.getTime())) return false;
        return (now.getTime() - dj.getTime()) <= 7 * 24 * 60 * 60 * 1000;
      }

      if (next === 'premium_inactive_14d') {
        if (u.plan !== 'Premium') return false;
        const lastDays = daysSinceDate(u.last_login);
        return u.is_active === false || lastDays === null || lastDays >= 14;
      }

      if (next === 'risk_high') {
        return state.key === 'inactive';
      }

      return true;
    });

    if (!silent) renderTable(filtered);
    return filtered;
  }

  async function fetchAudit() {
    try {
      const res = await authFetch(`${API_URL}admin/audit/?page=1&pageSize=25`);
      if (!res.ok) return false;
      auditCache = await res.json();
      return true;
    } catch (e) {
      return false;
    }
  }

  function renderAudit() {
    const tbody = els.auditBody();
    if (!tbody) return;
    tbody.innerHTML = '';

    const rows = auditCache && (auditCache.data || auditCache);
    const items = Array.isArray(rows) ? rows : [];

    items.forEach((r) => {
      const tr = document.createElement('tr');
      const whenRaw = r.occurred_at || r.created_at;
      const when = whenRaw ? new Date(whenRaw).toLocaleString() : '-';
      const action = r.action || '-';
      const entity = `${r.entity_type || '-'}:${r.entity_id || '-'}`;
      const actorObj = r.actor && typeof r.actor === 'object' ? r.actor : null;
      const actor = (actorObj && (actorObj.username || actorObj.email || actorObj.id)) || r.actor_username || r.actor_id || '-';
      const reason = r.reason || '-';
      tr.innerHTML = `
        <td>${when}</td>
        <td>${action}</td>
        <td>${entity}</td>
        <td>${actor}</td>
        <td style="color: var(--text-muted);">${reason}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  async function refreshAudit() {
    const ok = await fetchAudit();
    if (ok) renderAudit();
  }

  function filterUsers() {
    const term = (els.searchInput()?.value || '').toLowerCase();
    const filtered = allUsers.filter(
      (u) => String(u.username || '').toLowerCase().includes(term) || String(u.email || '').toLowerCase().includes(term),
    );
    renderTable(filtered);
  }

  async function deleteUser(id) {
    const user = allUsers.find((u) => u && u.id === id);
    const who = user ? `${user.email} (#${user.id})` : `#${id}`;
    if (!confirm(`¿Eliminar usuario ${who}? (borrado lógico)`)) return;
    const confirmText = (prompt('Escribe ELIMINAR para confirmar esta acción:') || '').trim();
    if (confirmText !== 'ELIMINAR') {
      showToast('Confirmación cancelada', 'error');
      return;
    }
    const reason = (prompt('Motivo (requerido) para eliminar este usuario:') || '').trim();
    if (!reason) {
      showToast('Motivo requerido', 'error');
      return;
    }
    try {
      const res = await authFetch(API_URL + `users/delete/${id}/`, {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ reason }),
      });
      if (res.ok) {
        showToast('Eliminado', 'success');
        selectedUserIds.delete(Number(id));
        await fetchUsers();
        await refreshAudit();
      } else {
        showToast(await readApiError(res, 'No fue posible eliminar el usuario'), 'error');
      }
    } catch (e) {
      showToast('Error de conexión al eliminar', 'error');
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
    const reason = (prompt('Motivo (requerido) para cambiar el plan:') || '').trim();
    if (!reason) {
      showToast('Motivo requerido', 'error');
      return;
    }

    try {
      const res = await authFetch(API_URL + `users/update_admin/${userId}/`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: user.username, email: user.email, plan: pretty, reason }),
      });
      if (res.ok) {
        showToast('Plan actualizado', 'success');
        fetchUsers();
        refreshAudit();
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
    const username = String(document.getElementById('newUsername').value || '').trim();
    const email = String(document.getElementById('newEmail').value || '').trim();
    const password = String(document.getElementById('newPassword').value || '').trim();
    const plan = String(document.getElementById('newPlan').value || 'Gratis').trim() === 'Premium' ? 'Premium' : 'Gratis';

    if (!username || !email || !password) {
      showToast('Completa usuario, email y contraseña', 'error');
      return;
    }

    try {
      const res = await authFetch(API_URL + 'users/create/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password, plan }),
      });
      if (res.ok) {
        showToast('Usuario creado', 'success');
        closeModal();
        await fetchUsers();
        await refreshAudit();
        e.target.reset();
      } else {
        showToast(await readApiError(res, 'Error al crear usuario'), 'error');
      }
    } catch (err) {
      showToast('Error de conexión al crear', 'error');
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

      const reason = (prompt('Motivo (requerido) para enviar esta notificación masiva:') || '').trim();
      if (!reason) {
        showToast('Motivo requerido', 'error');
        return;
      }

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
          reason,
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
      localStorage.removeItem('access');
      localStorage.removeItem('refresh');
      localStorage.removeItem('token');
      window.location.href = '/pages/auth/indexInicioDeSesion.html';
    }
  }

  function refreshDashboard() {
    loadPageData();
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
  window.refreshDashboard = refreshDashboard;
  window.bulkSetPlan = bulkSetPlan;
  window.bulkDeleteUsers = bulkDeleteUsers;
  window.applySegment = applySegment;
  window.refreshAudit = refreshAudit;

  document.addEventListener('DOMContentLoaded', () => {
    if (!requireAdminSessionOrRedirect()) return;

    bindAdminMenu();

    loadStoredDaysWindow();
    loadStoredCompareEnabled();
    bindRangeSelect();
    bindCompareToggle();
    syncRangeUI();
    bindBulkInputs();
    bindSelectAll();

    const toggleEl = els.compareToggle();
    if (toggleEl) toggleEl.checked = compareEnabled;

    loadPageData();

    if (isPage('users')) {
      refreshAudit();
    }

    if (isPage('notifications')) {
      setLastUpdatedNow();
    }

    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js').catch((err) => console.error('SW Fail', err));
    }
  });
})();
