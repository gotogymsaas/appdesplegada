# Dashboard Admin (5 hitos) — implementación en este repo

Este repositorio **no** es un monorepo TypeScript (Next/Nest/Functions). El MVP se implementa sobre el stack real:
- Frontend: HTML/CSS/JS (vanilla) + PWA
- Backend: Django + DRF + SimpleJWT

## Hito 1 — Seguridad + auditoría
- Modelo `AuditLog` y registro en admin.
- Acciones sensibles requieren `reason` (motivo) y generan auditoría:
  - Borrado lógico usuario: `DELETE /api/users/delete/{id}/`
  - Cambio plan (si cambia): `PUT /api/users/update_admin/{id}/`
  - Broadcast push: `POST /api/push/admin/broadcast/`
- Borrado de usuario es **lógico**: se desactiva (`is_active=false`) y se anonimiza email/username.

## Hito 2 — KPIs agregados (rango + compare)
Nuevos endpoints agregados (superuser):
- `GET /api/admin/dashboard/overview/?days=90&timezone=America/Bogota&compare=true`
- `GET /api/admin/dashboard/signups_series/?days=90&timezone=America/Bogota`

## Hito 3 — Users API con filtros/paginación (compat)
- `GET /api/users/` mantiene compatibilidad: si no se pasan params retorna la lista completa.
- Si se pasan `page`/`pageSize`, devuelve `{data, meta}`.
- Filtros: `q`, `plan`, `status`.

## Hito 4 — AdminDashboard conectado
- Frontend solicita `reason` por prompt en acciones sensibles.
- Prefiere endpoints agregados para KPIs/series cuando están disponibles.

## Hito 5 — Tests mínimos + operación
- Tests activos en `backend/api/test_admin_security.py`.
- `backend/api/tests.py` es legacy y está en `SkipTest`.

## Variables de entorno relacionadas
- Django: `DJANGO_SECRET_KEY`
- WebPush: `VAPID_PUBLIC_KEY`, `VAPID_PRIVATE_KEY`, `VAPID_EMAIL`
- (Móvil/FCM): según configuración usada en `_send_fcm` en `backend/api/views.py`
