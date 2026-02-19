# Contexto del proyecto GoToGym

## Resumen
Este repositorio contiene:
- **Backend**: API REST en Django (DRF) con JWT.
- **Frontend**: web estática (HTML/CSS/JS) consumida también por apps móviles vía Capacitor.
- **Móvil**: proyectos nativos `android/` y `ios/` generados por Capacitor.
- **Infra/operación**: scripts y workflows de despliegue a Azure.

## Estructura principal del repo
- `backend/`: Django (API, auth, storage, OCR/ML, integraciones).
- `frontend/`: web estática (incluye `js/config.js` como configuración cliente).
- `android/`, `ios/`: contenedores nativos (Capacitor) que embeben `frontend/`.
- `.github/workflows/`: despliegues (App Service + Static Web Apps).
- `infra/`: notas y scripts de soporte.
- `startup.sh`: entrypoint usado en App Service para dependencias runtime + `gunicorn`.

## Backend (Django)
- Punto de entrada: `backend/manage.py`.
- Settings: `backend/backend_gotogym/settings.py`.
- Dependencias runtime: `backend/requirements.runtime.txt`.
- DB: SQLite en dev por defecto; PostgreSQL en prod via `DATABASE_URL` o variables `POSTGRES_*`.

### Variables de entorno (resumen)
Ver `env.example` para la lista completa. Claves típicas:
- `DEBUG`, `DJANGO_SECRET_KEY`, `ALLOWED_HOSTS`
- `CORS_ALLOWED_ORIGINS`, `CSRF_TRUSTED_ORIGINS`
- `DATABASE_URL` o `POSTGRES_*`
- Storage Azure Blob: `AZURE_STORAGE_*` + contenedores por dominio (nutrición/entrenamiento/historia clínica)
- Email: Azure Communication Services (ACS): `ACS_EMAIL_CONNECTION_STRING`
- Push: `FCM_SERVER_KEY`
- Pagos: `MERCADOPAGO_ACCESS_TOKEN`

### Runtime / arranque en Azure
El script `startup.sh`:
- Instala dependencias del sistema para OCR si están disponibles (`tesseract`, `poppler`).
- Instala dependencias Python si faltan.
- Ejecuta `migrate`, `collectstatic` y arranca `gunicorn`.

## Frontend
- Configuración de API: `frontend/js/config.js` resuelve el `API_URL` según entorno (prod/local/LAN) y permite override.
- Se sirve como sitio estático (local) y como build empaquetado en SWA/hosting.

## Móvil (Capacitor)
- Dependencias en `package.json` (Capacitor + push + health integrations).
- Los proyectos nativos viven en `android/` y `ios/`.

## Servicios externos / integraciones
Según `STACK.md` y `ARCHITECTURE.md`:
- Email: Azure Communication Services (Email).
- Push: Firebase Cloud Messaging.
- Pagos: MercadoPago (webhooks).
- Automatización: n8n (túnel/webhook en dev).
- Salud: Google Fit / HealthKit.
- OCR/ML: procesamiento de PDFs e imágenes.

### Integración WHOOP
- Estado/pendientes: ver `docs/WHOOP_INTEGRATION.md`.

## Azure (recursos detectados en la suscripción actual)
- App Service (API): `gotogym-api-ca-7581` (resource group `gotogym-prod-rg`).
- App Service adicional: `gotogym-api-26237` (resource group `gotogym-prod-rg`).
- Web App: `gotogymweb` (resource group `gotogymweb`).
- Static Web App: `GitHub` (resource group `Github`).
- Web App adicional: `apicoachdesalud` (resource group `n8n-gotogym`).

## Flujos de despliegue (GitHub Actions)
- `deploy-appservice.yml`: despliegue manual del API a App Service usando `AZURE_WEBAPP_PUBLISH_PROFILE`.
- `main_gotogym-api-ca-7581.yml`: despliegue automático (push a `main`) del API usando `AZURE_WEBAPP_PUBLISH_PROFILE`.
- `deploy-staticwebapp.yml`: despliegue manual del frontend con `AZURE_STATIC_WEB_APPS_API_TOKEN`.

## Ejecución local (alto nivel)
- Backend: crear venv, instalar requirements y correr `manage.py runserver`.
- Frontend: servir `frontend/` (por ejemplo con `npx serve`).
- Túnel n8n (dev): `localtunnel` sobre el puerto del backend.

Referencias: `STACK.md`, `ARCHITECTURE.md`, `INSTRUCCIONES_INSTALACION.txt`.
