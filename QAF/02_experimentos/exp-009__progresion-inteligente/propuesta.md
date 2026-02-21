# Propuesta — Exp-009 (QAF Progresión Fuerza+Cardio)

## Inputs mínimos (MVP)

- Fuerza: 1 ejercicio clave con `sets x reps x load_kg` (o `sets x reps` si no hay carga).
- Cardio: `minutes` + `RPE 1–10` (y `avg_hr` si existe).
- Post-sesión: `RPE 1–10`, `completion_pct`.
- Señales del día (si hay wearable): `sleep_minutes`, `steps`, `resting_heart_rate_bpm`.

## Outputs

- `readiness_score` (0..100) + explicación.
- `plateau` detectado (ventana 3 sesiones) + razón.
- decisión: `progress / deload / variation / minimum_viable / swap_exercise`.
- `micro_goal` (1 objetivo concreto siempre).
