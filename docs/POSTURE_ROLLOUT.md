# Postura (Exp-006) — Cómo probar en el celular

Este flujo usa el chat existente (botones + cámara) y calcula **pose estimation en el cliente** (MediaPipe Pose) para enviar solo **keypoints** al backend.

## Requisitos

- Backend levantado en tu PC en `:8000`
- Frontend servido en tu PC en `:5500`
- Celular en la misma Wi‑Fi
- Usuario logueado (para que el chat pueda llamar al backend con JWT)

## Pasos

1) Backend:

- `cd backend && python manage.py runserver 0.0.0.0:8000`

2) Frontend:

- `cd frontend && python -m http.server 5500 --bind 0.0.0.0`

3) En el celular abre:

- `http://<IP_DE_TU_PC>:5500/pages/auth/indexInicioDeSesion.html`

4) Abre el chat y toca el botón **Postura**.

5) Sigue el flujo:

- Tomar foto frontal (cuerpo completo, buena luz)
- Tomar foto lateral
- Responder la pregunta de seguridad

El chat debe responder con:

- `decision` + `confidence`
- etiquetas detectadas (ej. `forward_head`, `rounded_shoulders`)
- rutina recomendada (ejercicios básicos) + aviso de seguridad

## Contrato (posture_request v0)

El chat envía al backend (`POST /api/chat/`) un payload extra:

```json
{
  "posture_request": {
    "poses": {
      "front": {"keypoints": [{"name": "left_shoulder", "x": 0.43, "y": 0.26, "score": 0.92}], "image": {"width": 1080, "height": 1920}},
      "side":  {"keypoints": [{"name": "right_ear", "x": 0.66, "y": 0.14, "score": 0.92}], "image": {"width": 1080, "height": 1920}}
    },
    "user_context": {"injury_recent": false, "pain_neck": false, "pain_low_back": false, "level": "beginner"},
    "locale": "es-CO"
  }
}
```

Notas:
- Las coordenadas `x,y` son normalizadas 0..1 (MediaPipe Pose).
- El backend valida tamaño/estructura y no intenta inferir postura desde “descripción de imagen”.
