const CACHE_NAME = 'gotogym-v6';
const ASSETS_TO_CACHE = [
    '/',
    '/manifest.json',
    '/css/theme.css',
    '/js/config.js',
    '/assets/images/recurso-14.png',
    '/assets/images/logo-gotogym-192.png',
    '/assets/images/apple-touch-icon.png'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {
                console.log('[Service Worker] Caching core assets');
                return cache.addAll(ASSETS_TO_CACHE);
            })
    );
    self.skipWaiting();
});

self.addEventListener('activate', event => {
    event.waitUntil(
        caches.keys().then(keyList => {
            return Promise.all(keyList.map(key => {
                if (key !== CACHE_NAME) {
                    return caches.delete(key);
                }
            }));
        })
    );
    self.clients.claim();
});

self.addEventListener('fetch', event => {
    // Only cache GET requests
    if (event.request.method !== 'GET') return;

    const url = new URL(event.request.url);
    const isSameOrigin = url.origin === self.location.origin;

    // Navigations: prefer fresh HTML so UX updates propagate.
    if (event.request.mode === 'navigate') {
        event.respondWith(
            fetch(event.request)
                .then(res => {
                    if (isSameOrigin) {
                        const copy = res.clone();
                        caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
                    }
                    return res;
                })
                .catch(() => caches.match(event.request))
        );
        return;
    }

    // Static assets: stale-while-revalidate for better update behavior.
    if (isSameOrigin && (url.pathname.endsWith('.js') || url.pathname.endsWith('.css') || url.pathname.endsWith('.png') || url.pathname.endsWith('.jpg') || url.pathname.endsWith('.jpeg') || url.pathname.endsWith('.svg') || url.pathname.endsWith('.webp') || url.pathname.endsWith('.ico') || url.pathname.endsWith('.json')))
    {
        event.respondWith(
            caches.match(event.request).then(cached => {
                const networkFetch = fetch(event.request)
                    .then(res => {
                        const copy = res.clone();
                        caches.open(CACHE_NAME).then(cache => cache.put(event.request, copy));
                        return res;
                    })
                    .catch(() => cached);
                return cached || networkFetch;
            })
        );
        return;
    }

    event.respondWith(
        caches.match(event.request)
            .then(response => {
                return response || fetch(event.request);
            })
    );
});

self.addEventListener('push', event => {
    let payload = {};
    try {
        payload = event.data ? event.data.json() : {};
    } catch (e) {
        payload = {};
    }

    const title = payload.title || 'GoToGym';
    const body = payload.body || 'Estoy aqui para acompanarte.';
    const data = payload.data || {};

    const actions = (data.actionType === 'IF_QUICK')
        ? [
            { action: 'IF_3', title: '3' },
            { action: 'IF_5', title: '5' },
            { action: 'IF_7', title: '7' },
            { action: 'IF_9', title: '9' },
        ]
        : [];

    event.waitUntil(
        self.registration.showNotification(title, {
            body,
            data,
            icon: '/assets/images/logo-gotogym-192.png',
            badge: '/assets/images/logo-gotogym-192.png',
            actions,
        })
    );
});

// Interactive Notification Handler
self.addEventListener('notificationclick', event => {
    event.notification.close();

    const data = event.notification.data || {};

    if (event.action && event.action.startsWith('IF_')) {
        const value = Number(event.action.replace('IF_', ''));
        if (value && data.slot) {
            event.waitUntil(
                self.clients.matchAll().then((clientsArr) => {
                    clientsArr.forEach((client) => {
                        client.postMessage({
                            type: 'PUSH_IF_ANSWER',
                            value,
                            slot: data.slot,
                        });
                    });
                })
            );
            return;
        }
    }

    // If normal click (no action), open app
    if (!event.action) {
        clients.openWindow('/pages/profile/Perfil.html');
        return;
    }

    // Handle Actions (e.g. action="s_sleep-8")
    // Format: "variable-value"
    const [variable, value] = event.action.split('-');

    if (variable && value) {
        // We need the username. In a real app we'd store it in IndexedDB or similar.
        // For this demo, let's assume we can get it from an open window or it was passed in notification data.
        // Simpler: Broadcast to client or just try to grab from notification data if sent.

        // Let's assume the notification data container the username
        const username = event.notification.data ? event.notification.data.username : null;

        if (username) {
            event.waitUntil(
                submitScoreUpdate(username, variable, value)
            );
        }
    }
});

async function submitScoreUpdate(username, variable, value) {
    try {
        // Need absolute URL for SW
        // Try localhost and local IP fallback is hard in SW without config.
        // We will assume the origin that registered the SW.
        // But SW 'self.location.origin' works.
        const apiUrl = self.location.origin.replace(':5500', ':8000') + '/api/update_score/';

        const res = await fetch(apiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, variable, value })
        });

        const data = await res.json();

        if (data.success) {
            self.registration.showNotification("¡Actualizado!", {
                body: `Tu Índice de Felicidad ahora es ${data.happiness_percentage}%`,
                icon: '/assets/images/logo-gotogym-192.png'
            });

            // Refresh open windows
            const clientsArr = await self.clients.matchAll();
            clientsArr.forEach(client => client.postMessage({ type: 'REFRESH_PROFILE' }));
        }
    } catch (e) {
        console.error("Update failed", e);
    }
}
