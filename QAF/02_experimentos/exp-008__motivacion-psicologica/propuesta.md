# Propuesta — Exp-008 (QAF Motivación Psicológica)

## Inputs reales disponibles hoy (arquitectura actual)

- Texto del chat (proxy): se convierte a señales sin guardar texto completo.
- Memoria: `user.coach_state` / `coach_weekly_state`.
- Gamificación: `streak`, `badges`, misiones semanales.
- Estado físico proxy: Exp-007 (DHSS) si existe; si no, estado `energy_mode` del sync.
- Push: existen endpoints FCM/WebPush y scheduler interno (depende de config).

## Limitaciones MVP

- No hay registro confiable de workouts completados (series/reps/PRs).
- No hay modelo de equipos/retos grupales reales.

## Salida QAF

- `profile.vector`: logro, disciplina, salud, estética, comunidad (0..1)
- `state.mood`: eufórico/neutral/fatiga/frustrado/ansioso
- `intervention.level`: 0..3 (anti-abandono)
- `tone.style`: estilo de lenguaje
- `challenge`: reto sugerido (seguro)
- `reward`: reconocimiento (streak/badge/wellbeing)

## Integración

- Endpoint debug: `POST /api/qaf/motivation/`
- Chat: dentro de `/api/chat/` cuando llega `motivation_request` o el mensaje lo sugiere.
