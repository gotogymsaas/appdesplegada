# Propuesta — Exp-007 (QAF Lifestyle Intelligence)

## Lo que sí podemos hacer con datos reales hoy

- Sueño: `sleep_minutes` (wearable) y/o auto-reporte.
- Movimiento: `steps` (wearable).
- Estrés: proxy con `resting_heart_rate_bpm` si existe (Fitbit) y/o auto-reporte.
- Actividad previa: proxy con `calories`/`distance`.

## Lo que no podemos inferir con precisión (MVP)

- Hidratación real (solo auto-reporte).
- Sedentarismo/horas sentado (solo auto-reporte o futura integración).
- HRV (por ahora no está normalizada en `FitnessSync.metrics`).

## Diseño QAF

1) Capa 1 — DHSS:
- Score 0..100 con `confidence`.

2) Capa 2 — Patrones (7–14 días):
- sueño bajo consecutivo
- pasos bajos consecutivos
- RHR por encima del baseline (si existe)

3) Capa 3 — Ajuste automático:
- entrenamiento/hábitos del día (solo recomendaciones, sin “diagnóstico”).

4) Capa 4 — Micro-hábitos (máx 3):
- accionables, concretos, sin repetición 3 días seguidos.

5) Capa 5 — Personalización de lenguaje:
- `persona_style` guardado en memoria (directo/calmado/reto/bienestar).

## Integración

- Backend: endpoint `POST /api/qaf/lifestyle/`.
- Chat: usar `quick_actions` existentes para:
  - “Estado de hoy”
  - preguntas mínimas de confirmación (1–5)
  - “Marcar hábito como hecho”
