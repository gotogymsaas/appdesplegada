# Exp-012 — Shape & Presence Intelligence™ (MVP)

Objetivo: entregar feedback de **proporción** y **presencia postural** usando *keypoints 2D* (calculados en cliente) sin prometer medidas reales.

## Qué hace
- Acepta 1–2 fotos (frontal relajado recomendado; perfil derecho opcional).
- Calcula variables *proxy* (0–100) sobre alineación, silueta y base.
- Devuelve insights + acciones sugeridas (lenguaje profesional, no estético).

## Qué NO hace (guardrails)
- No hace segmentación corporal.
- No calcula centímetros reales.
- No juzga atractivo / belleza.
- No hace diagnóstico médico.

## Verificación backend (antes de deploy)
- Endpoint registrado: `POST /api/qaf/shape_presence/`.
- Chat integrado: `POST /api/chat/` acepta `shape_presence_request`.
- Persistencia semanal: `user.coach_weekly_state.shape_presence[week_id]`.
- Privacidad: solo se envían keypoints (no fotos) en este flujo.
- Guardrail de calidad: `decision=needs_confirmation` si no hay keypoints suficientes.

## Entrada (JSON)
Se usa desde chat o endpoint QAF.

```json
{
  "poses": {
    "front_relaxed": {"keypoints": [{"name":"left_shoulder","x":0.4,"y":0.3,"score":0.9}], "image": {"width": 1080, "height": 1920}},
    "side_right_relaxed": {"keypoints": [{"name":"right_ear","x":0.6,"y":0.2,"score":0.9}], "image": {"width": 1080, "height": 1920}}
  },
  "locale": "es-CO"
}
```

## Salida (resumen)
- `decision`: `accepted | needs_confirmation`
- `confidence.score` y `confidence.uncertainty_score`
- `variables.overall_presence` y sub-scores
- `insights` y `recommended_actions`

