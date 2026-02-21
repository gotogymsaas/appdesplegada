# Exp-006 — Postura: detección + rutina correctiva (QAF)

Objetivo: identificar **patrones posturales comunes** (no diagnóstico médico) usando señales de *pose estimation* (keypoints) y recomendar una rutina **segura** y accionable.

Este experimento está diseñado para encajar con la arquitectura actual:
- El chat ya soporta adjuntos (fotos) y “visión” descriptiva.
- El backend **aún no** tiene un modelo real de pose-estimation incorporado.

Por eso el motor de Exp-006 trabaja con la **salida estándar de pose** (keypoints + score) como entrada. La pose puede venir de:
- cliente (ideal: MediaPipe Pose / MoveNet en el móvil)
- microservicio de visión dedicado

## Qué hay aquí
- `scripts/qaf_posture_corrective.py`: motor (QC + métricas + etiquetas + recomendaciones).
- `scripts/run_local_eval.py`: runner simple para datasets JSONL.
- `data/dataset.sample.jsonl`: ejemplos mínimos de payload.
- `tests/test_qaf_posture_corrective.py`: tests unitarios.

## Entrada esperada (MVP)

El motor espera un objeto con:

```json
{
  "poses": {
    "front": {"keypoints": [{"name":"left_shoulder","x":0.5,"y":0.3,"score":0.9}], "image": {"width":1080,"height":1920}},
    "side":  {"keypoints": [{"name":"right_ear","x":0.62,"y":0.18,"score":0.9}], "image": {"width":1080,"height":1920}}
  },
  "user_context": {"pain_neck": false, "pain_low_back": false, "injury_recent": false, "level": "beginner"},
  "locale": "es-CO"
}
```

## Salida

Devuelve:
- `decision`: `accepted` | `needs_confirmation`
- `confidence`: score 0..1 + `uncertainty_score`
- `signals`: métricas posturales con valor + umbral + confianza
- `labels`: etiquetas detectadas (ej: `forward_head`, `rounded_shoulders`)
- `recommendations`: rutina sugerida + mensajes de seguridad

## Ejecución local

Desde la raíz del repo:

- `python QAF/02_experimentos/exp-006__postura-correctiva/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-006__postura-correctiva/data/dataset.sample.jsonl`
