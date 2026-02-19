# Integración WHOOP — Estado y pendientes

Fecha: 2026-02-19

## Estado (porcentual)

- **UX (Dispositivos.html)**: **100%**
  - WHOOP aparece en la lista de proveedores usando el mismo render, estilos y botones existentes.
  - El botón se habilita/deshabilita según configuración del backend (sin cambios de diseño).

- **Backend (OAuth + Sync + Raw ingest)**: **90%**
  - Endpoints OAuth WHOOP implementados:
    - `GET /oauth/whoop/authorize/?token=<JWT>`
    - `GET /oauth/whoop/callback/`
  - Sync implementado para ingestar **todos los datos** iniciales en capa **raw**.
  - Falta: configuración en App Service + registro OAuth en WHOOP + validación end-to-end con un usuario real.

- **Operación/producción (config + hardening)**: **40%**
  - Falta configurar secretos/variables, verificar tasas (429), y validar que la ingesta no afecte estabilidad.

## Pendientes para habilitar “Conectar” real

### 1) WHOOP Developer (OAuth App)

- Crear OAuth app en WHOOP.
- Configurar Redirect URI exacto:
  - `https://api.gotogym.store/oauth/whoop/callback/`
- Obtener y guardar:
  - `WHOOP_CLIENT_ID`
  - `WHOOP_CLIENT_SECRET`

### 2) Azure App Service (gotogym-api-ca-7581)

Configurar Application Settings (preferible usando secretos / Key Vault references):

- `WHOOP_CLIENT_ID`
- `WHOOP_CLIENT_SECRET`
- `WHOOP_REDIRECT_URI=https://api.gotogym.store/oauth/whoop/callback/`
- (opcional) `WHOOP_SCOPE=read:sleep read:recovery read:cycles read:workout read:profile read:body_measurement`

Luego reiniciar el App Service.

### 3) Validación funcional (end-to-end)

- En web: `https://www.gotogym.store/pages/settings/Dispositivos.html`
- Click en **Conectar** WHOOP → debe redirigir a WHOOP → volver con:
  - `...?oauth=success&provider=whoop`
- Probar `Sync` WHOOP (manual) y verificar que se crean registros raw (capa 1).

## Hardening recomendado (sin cambiar UX)

- **No pasar JWT en querystring** (hoy se usa `?token=` para authorize): migrar a código de un solo uso o cookie de sesión.
- Manejar **429 rate limit** con backoff + reintentos y registrar métricas de fallos.
- Limitar el primer backfill (por ejemplo 30 días) y luego incremental por `last_sync_at`.
