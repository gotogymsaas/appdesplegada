# Checklist — UX Chat (tipo ChatGPT, sin refactor grande)

## Objetivo
- Que el último mensaje nunca quede tapado por el footer (input/adjuntos/voz/teclado móvil).
- Mantener auto-scroll “inteligente” (solo si el usuario está cerca del final).
- No introducir rediseños grandes ni nuevas pantallas/componentes.

## Implementación (hecha)
- [x] Reservar espacio inferior del área de mensajes en móvil usando una variable dinámica `--chat-footer-height`.
- [x] Calcular en runtime la altura real del footer (`#chat-input-area`) y actualizar `--chat-footer-height`.
- [x] Recalcular al cambiar:
  - [x] Tamaño del textarea (auto-resize).
  - [x] VisualViewport (teclado móvil / rotación / resize).
  - [x] Preview de adjuntos (mostrar/ocultar, estado uploading).
  - [x] Estado de voz (recording + botones cancel/retry visibles).
  - [x] Apertura del chat.
- [x] Mantener el último mensaje visible si el usuario estaba cerca del final (no forzar scroll si está leyendo arriba).

## Verificación rápida (QA manual)
- [ ] Desktop (ancho > 480px)
  - [ ] Abrir/cerrar chat: el último mensaje queda visible.
  - [ ] Enviar texto largo (varias líneas): el textarea crece y no tapa el último mensaje.
- [ ] Mobile (<= 480px)
  - [ ] Abrir chat a pantalla completa: el input no tapa mensajes.
  - [ ] Abrir teclado: el chat se ajusta y el último mensaje sigue visible.
  - [ ] Adjuntar imagen/PDF: aparece preview y no se tapa el último mensaje.
  - [ ] Activar voz (🎤): al mostrar “Cancelar”/“Reintentar” no se tapa el último mensaje.

## Listo para ver en el teléfono (cómo probar)
1. En tu PC, levanta backend:
   - `cd backend && python manage.py runserver 0.0.0.0:8000`
2. En tu PC, levanta frontend:
   - `npx serve ./frontend -p 5500 --cors`
3. En el celular (misma Wi‑Fi), abre:
   - `http://<IP_DE_TU_PC>:5500/pages/auth/indexInicioDeSesion.html`

Notas:
- El frontend ahora calcula `API_URL` en LAN como `http://<host_del_frontend>:8000/api/` automáticamente.
- Si tu red usa otra IP/puerto, puedes override con `localStorage.setItem('api_url_override','http://X:8000/api/')`.

## Deploy (cuando se haga push a main)
- Frontend: GitHub Actions `Deploy GoToGym Frontend` (Azure Static Web Apps) se ejecuta en `push` a `main`.
- API: GitHub Actions `Deploy GoToGym API (auto)` se ejecuta en `push` a `main`.

## Archivos tocados
- `frontend/js/chat.js`
- `frontend/css/chat.css`
- `frontend/js/config.js`
