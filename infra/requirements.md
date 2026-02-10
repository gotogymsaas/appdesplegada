# Infra Requirements (Azure)

## Estimación inicial de recursos

- Backend API (Django).
- 1-2 vCPU, 2-4 GB RAM para inicio.
- Escalar vertical u horizontal según crecimiento.

- Base de datos PostgreSQL.
- B1ms o B2ms para inicio, 20-50 GB almacenamiento.
- Backups diarios habilitados.

- Storage de archivos.
- Azure Blob Storage (Hot tier).
- 50-200 GB inicial, escalable.

- Frontend estático.
- Azure Static Web Apps o Azure Storage Static Website + CDN.

## Servicios Azure recomendados

- API.
- Azure App Service (Linux) o Azure Container Apps.
- Runtime: Python 3.12+ (revisar compatibilidad con dependencias ML).

- DB.
- Azure Database for PostgreSQL Flexible Server.

- Archivos.
- Azure Blob Storage + CDN (opcional).

- Email.
- Azure Communication Services (Email).

- Observabilidad.
- Application Insights + Log Analytics.

## Configuraciones de producción críticas

- HTTPS y dominio configurado.
- `DEBUG=false` y `DJANGO_SECRET_KEY` real.
- `ALLOWED_HOSTS` y `CORS_ALLOWED_ORIGINS` restringidos.
- `SECURE_SSL_REDIRECT=true`, `SESSION_COOKIE_SECURE=true`, `CSRF_COOKIE_SECURE=true`.
- `SECURE_PROXY_SSL_HEADER=true` si hay reverse proxy.
- Base de datos PostgreSQL (no SQLite).
- Staticfiles (`collectstatic`) y media servidos desde storage.

## Checklist de despliegue

1. Configurar App Service y subir backend.
2. Crear PostgreSQL y exportar `DATABASE_URL`.
3. Configurar variables de entorno (secrets y dominios).
4. Ejecutar migraciones.
5. Configurar staticfiles y media.
6. Configurar dominio + HTTPS.
7. Verificar endpoints críticos (login, register, contact, webhook).

## Comandos recomendados (local)

- Verificación de dependencias.
`backend/venv_gotogym/bin/python -V`
`backend/venv_gotogym/bin/pip install -r backend/requirements.txt`

- Preparar Django.
`backend/venv_gotogym/bin/python backend/manage.py check`
`backend/venv_gotogym/bin/python backend/manage.py migrate`
`backend/venv_gotogym/bin/python backend/manage.py collectstatic`
