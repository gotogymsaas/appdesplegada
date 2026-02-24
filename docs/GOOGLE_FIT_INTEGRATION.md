# Integración Google Fit — Estado y pendientes

Fecha: 2026-02-24

## Qué ya está implementado (código)

- **UX (Dispositivos.html)**: lista el proveedor, usa `/api/devices/` y permite conectar/desconectar/sync.
- **OAuth (web)**:
  - `GET /oauth/google_fit/authorize/?token=<JWT>`
  - `GET /oauth/google_fit/callback/`
- **Sync**: `POST /api/devices/google_fit/sync/` (usa Google Fitness aggregate API).

## Requisitos de configuración (obligatorio para habilitar “Conectar”)

El backend solo habilita Google Fit si existen estas variables:

- `GF_WEB_CLIENT_ID`
- `GF_WEB_CLIENT_SECRET`
- `GF_WEB_REDIRECT_URI` (debe coincidir exactamente con el configurado en Google)

Opcional (pero recomendado):
- `GOOGLE_FIT_FRONTEND_REDIRECT` (por defecto vuelve a la pantalla local de Dispositivos)

## Redirect URI recomendado para producción

- `GF_WEB_REDIRECT_URI=https://api.gotogym.store/oauth/google_fit/callback/`

Y en Google Cloud Console (OAuth client) registrar ese Redirect URI exacto.

## Cómo validar end-to-end (web)

1) Inicia sesión en la web y abre:
   - `https://www.gotogym.store/pages/settings/Dispositivos.html`
2) En Google Fit, el botón debe aparecer habilitado.
3) Click **Conectar** → debe redirigir a Google OAuth.
4) Al aceptar, debe volver a Dispositivos con:
   - `...?oauth=success&provider=google_fit`
5) Click **Sync** y verificar:
   - Respuesta 200
   - Se crea registro en `FitnessSync` con `provider=google_fit`

## Notas

- El flujo actual pasa el JWT en querystring (`?token=`) para iniciar el authorize; funciona, pero conviene endurecerlo si se va a escalar/producción estricta.
