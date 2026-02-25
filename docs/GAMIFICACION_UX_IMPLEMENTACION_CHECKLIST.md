# Checklist de implementación — UX Gamificación (Perfil)

## Reglas de diseño (obligatorias)
- [x] Mantener tipografía base `Poppins`.
- [x] Usar únicamente tokens de color del tema (`--primary`, `--secondary`, `--text-main`, `--text-muted`, `--card-bg`).
- [x] No introducir componentes visuales fuera del sistema actual.
- [x] Mantener estilo de botones existente (outline/neón, bordes redondeados, interacción ligera).

## Backend (datos de juego)
- [x] Exponer estado `wow` dentro de `build_gamification_status`.
- [x] Incluir `points_total`, `today_events_count` y estado de recompensa diaria.
- [x] Mantener endpoint `POST /api/gamification/claim_daily/` operativo.

## Frontend Perfil (experiencia videojuego)
- [x] Mostrar puntos actuales en card de evolución.
- [x] Agregar CTA “Reclamar recompensa diaria”.
- [x] Bloquear CTA cuando ya fue reclamada hoy.
- [x] Mostrar feedback inmediato (toast + texto contextual).
- [x] Reintroducir sección de logros (`badges-container`) para visualizar badges.
- [x] Añadir acciones directas por misión (check-in / dispositivos / IF).
- [x] Mantener progreso semanal y estados de misión completada.

## Frontend Perfil (animación + sonido)
- [x] Añadir microanimaciones de recompensa en puntos (`wow bump` + flash del contenedor).
- [x] Añadir animación de atención al CTA de recompensa diaria cuando está disponible.
- [x] Añadir animación de misión recién completada.
- [x] Añadir toggle de sonido (`SFX ON/OFF`) persistido en `localStorage`.
- [x] Implementar SFX positivo usando WebAudio sin assets externos.
- [x] Implementar háptica de éxito (`navigator.vibrate`) cuando está disponible.
- [x] Respetar accesibilidad de movimiento (`prefers-reduced-motion`).

## Verificación técnica
- [x] Ejecutar pruebas backend relevantes.
- [x] Verificar estado limpio de lint/sintaxis en archivos modificados.
- [ ] Validar manual en móvil (flujo completo de claim + misiones + badges).

## Fusión y despliegue
- [x] Commit único de esta implementación UX.
- [x] Push a `main` para disparar despliegues automáticos (API + Frontend).
- [ ] Confirmar en teléfono URL productiva con hard refresh.
