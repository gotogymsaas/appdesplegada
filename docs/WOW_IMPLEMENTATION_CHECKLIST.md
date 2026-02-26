# WOW Implementation Checklist (Perfil)

Fecha: 2026-02-26
Objetivo: reforzar sensación "videojuego" sin rediseñar UI.

## Checklist
- [x] Mantener UI actual y usar solo micro-feedback (sin cambios de layout)
- [x] Consolidar feedback en un único canal visual (`showFxToast`)
- [x] Disparar feedback solo ante deltas reales (puntos, progreso, racha, misiones, experiencias, badges)
- [x] Añadir mensajes semánticos por evento (misión/experiencia/logro)
- [x] Añadir mapeo de recompensa WOW por tipo de evento para mensajes consistentes
- [x] Reforzar compatibilidad móvil/iOS con `warmupAudio()` en primera interacción
- [x] Mantener `claimDailyReward()` robusto con fallback a `gamification/status`
- [x] Validar que no haya errores de sintaxis/editor
- [x] Publicar cambios en `main`

## Notas de implementación
- Se reutilizan clases existentes: `newly-completed`, `newly-done`, `newly-unlocked`, `streak-up`, `claim-success`, `progress-pulse`.
- No se agregaron librerías nuevas; se conserva WebAudio API por rendimiento en PWA/Capacitor.
