(function () {
  const NOTIF_ENABLED_KEY = 'notifications_enabled';
  const SCHEDULE_KEY = 'if_notifications_schedule';
  const ACTION_TYPE = 'IF_QUICK';
  const ACTIONS = [
    { id: 'IF_3', title: '3' },
    { id: 'IF_5', title: '5' },
    { id: 'IF_7', title: '7' },
    { id: 'IF_9', title: '9' },
  ];
  const SLOTS = [
    { slot: 'morning', hour: 7, minute: 45 },
    { slot: 'afternoon', hour: 13, minute: 0 },
    { slot: 'night', hour: 21, minute: 0 },
  ];

  const isNative = () => {
    return window.Capacitor && typeof window.Capacitor.isNativePlatform === 'function'
      ? window.Capacitor.isNativePlatform()
      : false;
  };

  const getLocalNotifications = () => {
    const plugins = window.Capacitor && window.Capacitor.Plugins;
    return plugins && plugins.LocalNotifications ? plugins.LocalNotifications : null;
  };

  const getPushNotifications = () => {
    const plugins = window.Capacitor && window.Capacitor.Plugins;
    return plugins && plugins.PushNotifications ? plugins.PushNotifications : null;
  };

  const getPlatform = () => {
    if (window.Capacitor && typeof window.Capacitor.getPlatform === 'function') {
      return window.Capacitor.getPlatform();
    }
    return 'web';
  };

  const getUser = () => {
    const raw = localStorage.getItem('user');
    try {
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  };

  const showToast = (message, type = 'success') => {
    const existing = document.getElementById('toast');
    const toast = existing || document.createElement('div');
    if (!existing) {
      toast.id = 'toast';
      toast.className = 'toast';
      document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.style.display = 'block';
    setTimeout(() => (toast.style.display = 'none'), 3000);
  };

  const PUSH_TOKEN_KEY = 'push_token';
  const savePushToken = (token) => {
    if (!token) return;
    localStorage.setItem(PUSH_TOKEN_KEY, token);
  };
  const getPushToken = () => localStorage.getItem(PUSH_TOKEN_KEY);

  const sendPushToken = async (token) => {
    if (!token) return false;
    const authFetch = window.authFetch || fetch;
    const res = await authFetch(`${API_URL}push/register/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        token,
        platform: getPlatform(),
      }),
    });
    return res.ok;
  };

  const unregisterPushToken = async () => {
    const token = getPushToken();
    if (!token) return false;
    const authFetch = window.authFetch || fetch;
    const res = await authFetch(`${API_URL}push/unregister/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });
    return res.ok;
  };

  const ensurePushReady = async () => {
    if (!isNative()) return false;
    const PushNotifications = getPushNotifications();
    if (!PushNotifications) return false;

    const perm = await PushNotifications.requestPermissions();
    const granted = perm && (perm.receive === 'granted' || perm.display === 'granted');
    if (!granted) {
      showToast('Necesitamos permiso para notificaciones', 'error');
      return false;
    }

    if (!window.__gtgPushListeners) {
      PushNotifications.addListener('registration', async (token) => {
        const value = token && token.value ? token.value : null;
        if (!value) return;
        savePushToken(value);
        await sendPushToken(value);
      });

      PushNotifications.addListener('registrationError', (err) => {
        console.error('Push registration error', err);
      });

      PushNotifications.addListener('pushNotificationReceived', (notification) => {
        const body = notification && notification.body ? notification.body : 'Tienes una notificación';
        showToast(body, 'success');
      });

      window.__gtgPushListeners = true;
    }

    await PushNotifications.register();
    return true;
  };

  const getScheduleState = () => {
    try {
      return JSON.parse(localStorage.getItem(SCHEDULE_KEY)) || {};
    } catch (e) {
      return {};
    }
  };

  const setScheduleState = (state) => {
    localStorage.setItem(SCHEDULE_KEY, JSON.stringify(state));
  };

  const todayKey = () => {
    const now = new Date();
    return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(
      now.getDate()
    ).padStart(2, '0')}`;
  };

  const buildNotificationId = (slot) => {
    const key = todayKey();
    const map = { morning: 1, afternoon: 2, night: 3 };
    return Number(`${key.replace(/-/g, '')}${map[slot] || 0}`);
  };

  const fetchQuestion = async (username, slot, exclude) => {
    const params = new URLSearchParams({ username, slot });
    if (exclude && exclude.length) {
      params.set('exclude', exclude.join(','));
    }
    const authFetch = window.authFetch || fetch;
    const res = await authFetch(`${API_URL}if/question/?${params.toString()}`);
    if (!res.ok) {
      return null;
    }
    return await res.json();
  };

  const postAnswer = async ({ value, slot, source, answered_at }) => {
    const authFetch = window.authFetch || fetch;
    const res = await authFetch(`${API_URL}if/answer/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        score: value,
        slot,
        source,
        answered_at,
      }),
    });
    return res.ok ? await res.json() : null;
  };

  const ensureNativeReady = async () => {
    if (!isNative()) return false;
    const LocalNotifications = getLocalNotifications();
    if (!LocalNotifications) return false;

    const perm = await LocalNotifications.requestPermissions();
    if (perm && perm.display !== 'granted') {
      return false;
    }

    await LocalNotifications.registerActionTypes({
      types: [
        {
          id: ACTION_TYPE,
          actions: ACTIONS,
        },
      ],
    });

    if (!window.__gtgNotifListener) {
      LocalNotifications.addListener('localNotificationActionPerformed', async (event) => {
        const actionId = event.actionId;
        const valueMap = { IF_3: 3, IF_5: 5, IF_7: 7, IF_9: 9 };
        const value = valueMap[actionId];
        if (!value) return;

        const extra = event.notification && event.notification.extra ? event.notification.extra : {};
        const result = await postAnswer({
          value,
          slot: extra.slot,
          source: 'notification',
          answered_at: new Date().toISOString(),
        });

        const user = getUser();
        if (result && result.happiness_index !== undefined && user) {
          user.happiness_index = result.happiness_index;
          localStorage.setItem('user', JSON.stringify(user));
        }

        showToast('Gracias. Te estoy acompañando hoy.', 'success');
        scheduleNext24Hours();
      });
      window.__gtgNotifListener = true;
    }

    return true;
  };

  const scheduleNext24Hours = async () => {
    const user = getUser();
    if (!user) return false;

    const ready = await ensureNativeReady();
    if (!ready) return false;

    const LocalNotifications = getLocalNotifications();

    await cancelAll();
    const scheduleState = {};
    const key = todayKey();

    const exclude = [];
    const notifications = [];

    const now = new Date();
    const windowEnd = new Date(now.getTime() + 24 * 60 * 60 * 1000);

    const slotsToSchedule = [];
    for (const slotDef of SLOTS) {
      const candidate = new Date();
      candidate.setHours(slotDef.hour, slotDef.minute, 0, 0);
      if (candidate > now && candidate <= windowEnd) {
        slotsToSchedule.push({ ...slotDef, when: candidate });
      }
      const nextDay = new Date(candidate);
      nextDay.setDate(nextDay.getDate() + 1);
      if (nextDay > now && nextDay <= windowEnd) {
        slotsToSchedule.push({ ...slotDef, when: nextDay });
      }
    }

    slotsToSchedule.sort((a, b) => a.when - b.when);

    for (const slotDef of slotsToSchedule) {
      if (slotDef.when <= now) {
        continue;
      }

      const questionData = await fetchQuestion(user.username, slotDef.slot, exclude);
      if (!questionData || questionData.completed) {
        continue;
      }

      const question = questionData.question;
      exclude.push(question.id);

      notifications.push({
        id: Number(`${slotDef.when.getTime()}`.slice(-9)),
        title: 'GoToGym • Check-in de bienestar',
        body: question.label,
        schedule: { at: slotDef.when },
        actionTypeId: ACTION_TYPE,
        extra: {
          slot: slotDef.slot,
          week_id: questionData.week_id,
        },
      });
    }

    if (notifications.length) {
      await LocalNotifications.schedule({ notifications });
      scheduleState[key] = { scheduledAt: new Date().toISOString(), slots: notifications.length };
      setScheduleState(scheduleState);
    }

    return true;
  };

  const cancelAll = async () => {
    const LocalNotifications = getLocalNotifications();
    if (!LocalNotifications) return;
    const pending = await LocalNotifications.getPending();
    if (pending && pending.notifications && pending.notifications.length) {
      await LocalNotifications.cancel({ notifications: pending.notifications });
    }
    localStorage.removeItem(SCHEDULE_KEY);
  };

  const enableNotifications = async () => {
    localStorage.setItem(NOTIF_ENABLED_KEY, 'true');
    await scheduleNext24Hours();
    await ensurePushReady();
  };

  const disableNotifications = async () => {
    localStorage.setItem(NOTIF_ENABLED_KEY, 'false');
    await cancelAll();
    await unregisterPushToken();
  };

  window.GoToGymNotifications = {
    enable: enableNotifications,
    disable: disableNotifications,
    scheduleNext24Hours,
    cancelAll,
    isEnabled: () => localStorage.getItem(NOTIF_ENABLED_KEY) !== 'false',
  };

  window.GoToGymPush = {
    enable: ensurePushReady,
    disable: unregisterPushToken,
  };

  if (
    typeof document !== 'undefined' &&
    localStorage.getItem(NOTIF_ENABLED_KEY) !== 'false' &&
    isNative()
  ) {
    scheduleNext24Hours();
    ensurePushReady();
  }
})();
