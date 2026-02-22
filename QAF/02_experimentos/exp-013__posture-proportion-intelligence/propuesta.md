# Propuesta — Exp-013 Posture & Proportion Intelligence™

## Hipótesis
Unificar postura + proporción (proxy) aumenta el “WOW” porque el usuario:
1) entiende en 1 frase qué pasa,
2) siente mejora inmediata con 2 correcciones,
3) recibe un ajuste semanal claro.

## Contrato
`posture_proportion_request.v0` via `POST /api/chat/` (o endpoint QAF directo).

## Variables (0–100)
- `posture_score` (A-score)
- `proportion_score` (P-score)
- `alignment_silhouette_index` (unificado)

## Salidas
- `immediate_corrections` (2): ejercicios 30–60s + cue.
- `weekly_adjustment` (1): foco semanal y nota.

## Limitaciones explícitas
- Sin segmentación → no cintura real.
- Sin calibración → no distancia en metros.
- Confianza baja → pedir retomar fotos.

