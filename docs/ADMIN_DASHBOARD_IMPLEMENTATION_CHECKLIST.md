# Checklist de implementación — Refactor Dashboard Admin

Fecha: 2026-02-27

## 1) Arquitectura y separación de responsabilidades
- [x] Separar dashboard de métricas vs operación (usuarios/notificaciones).
- [x] Definir navegación admin consistente entre páginas.
- [x] Mantener `theme.css` y tipografías del proyecto principal.

## 2) Navegación y UX estructurada
- [x] Header estructurado con menú hamburguesa en páginas admin.
- [x] Añadir rutas de navegación:
  - [x] Dashboard Métricas (`pages/admin/AdminDashboard.html`)
  - [x] Administración de Usuarios (`pages/admin/AdminUsers.html`)
  - [x] Panel de Notificaciones (`pages/admin/AdminNotifications.html`)

## 3) Dashboard exclusivo de métricas
- [x] Mantener estratégico/táctico en dashboard principal.
- [x] Ocultar bloque operativo tradicional en modo métricas.
- [x] Agregar bloque operativo de métricas técnicas (runtime/costos).

## 4) Métricas operativas backend
- [x] Extender trazabilidad `llm.hito5.inference` con `flow` y `tokens`.
- [x] Crear endpoint `GET /api/admin/dashboard/ops_metrics/`.
- [x] Agregar a router API (`backend/api/urls.py`).

## 5) Métricas operativas frontend
- [x] Consumir `ops_metrics` desde `adminDashboard.js`.
- [x] Renderizar gráfica de líneas (13 experiencias + total).
- [x] Renderizar KPIs de benchmark y costo unitario por usuario.

## 6) Administración y notificaciones por páginas
- [x] Página dedicada de usuarios con tabla + auditoría + modales.
- [x] Página dedicada de notificaciones con broadcast y permisos push.
- [x] Reutilizar JS común con modo por página (`data-admin-page`).

## 7) Calidad técnica
- [x] Validar errores de archivos modificados con diagnóstico estático del entorno.
- [x] Mantener compatibilidad con endpoints existentes (`overview`, `signups_series`, `audit`, `push`).

## Archivos tocados
- `backend/api/views.py`
- `backend/api/urls.py`
- `frontend/js/adminDashboard.js`
- `frontend/css/adminDashboard.css`
- `frontend/pages/admin/AdminDashboard.html`
- `frontend/pages/admin/AdminUsers.html` (nuevo)
- `frontend/pages/admin/AdminNotifications.html` (nuevo)
