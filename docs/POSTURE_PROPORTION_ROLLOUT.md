````markdown
# Postura & Proporción (Exp-013) — Cómo probar en el celular

Este flujo usa el chat existente (botones + cámara) y calcula **pose estimation en el cliente** (MediaPipe Pose) para enviar solo **keypoints** al backend.

## Requisitos
- Backend en tu PC `:8000`
- Frontend en tu PC `:5500`
- Celular en la misma Wi‑Fi
- Usuario logueado (JWT)

## Pasos
1) Backend:
- `cd backend && python manage.py runserver 0.0.0.0:8000`

2) Frontend:
- `cd frontend && python -m http.server 5500 --bind 0.0.0.0`

3) En el celular:
- `http://<IP_DE_TU_PC>:5500/pages/auth/indexInicioDeSesion.html`

4) Abre el chat y escribe **“Postura & Proporción”**.

5) Sigue el flujo (3 fotos):
- Frente relajado
- Perfil derecho
- Espalda (opcional recomendado)

6) Toca **Analizar ahora**.

## Contrato (posture_proportion_request v0)
```json
{
  "posture_proportion_request": {
    "poses": {
      "front_relaxed": {"keypoints": [{"name": "left_shoulder", "x": 0.43, "y": 0.26, "score": 0.92}], "image": {"width": 1080, "height": 1920}},
      "side_right_relaxed": {"keypoints": [{"name": "right_ear", "x": 0.66, "y": 0.14, "score": 0.92}], "image": {"width": 1080, "height": 1920}},
      "back_relaxed": {"keypoints": [{"name": "left_shoulder", "x": 0.40, "y": 0.27, "score": 0.90}], "image": {"width": 1080, "height": 1920}}
    },
    "locale": "es-CO"
  }
}
```

## Endpoint directo (QAF)
- `POST /api/qaf/posture_proportion/`

````
