````markdown
# Shape & Presence (Exp-012) — Cómo probar en el celular

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

4) Abre el chat y escribe **“Shape & Presence”** (o sube una foto que el router clasifique como `route=training` y toca el botón **Shape & Presence**).

5) Sigue el flujo:

- Tomar/adjuntar foto **frente relajado** (cuerpo completo, buena luz)
- (Opcional) Tomar/adjuntar foto **perfil derecho**
- Toca **Analizar ahora**

El chat debe responder con:

- `decision` + `confidence`
- variables (0–100) tipo `overall_presence`, `alignment_symmetry`, etc.
- insights + acciones recomendadas

## Contrato (shape_presence_request v0)

El chat envía al backend (`POST /api/chat/`) un payload extra:

```json
{
  "shape_presence_request": {
    "poses": {
      "front_relaxed": {"keypoints": [{"name": "left_shoulder", "x": 0.43, "y": 0.26, "score": 0.92}], "image": {"width": 1080, "height": 1920}},
      "side_right_relaxed": {"keypoints": [{"name": "right_ear", "x": 0.66, "y": 0.14, "score": 0.92}], "image": {"width": 1080, "height": 1920}}
    },
    "locale": "es-CO"
  }
}
```

Notas:
- Las coordenadas `x,y` son normalizadas 0..1 (MediaPipe Pose).
- El backend **no** hace pose estimation; valida estructura y calcula métricas proxy.
- No se prometen centímetros reales.

## Endpoint directo (QAF)

También existe:
- `POST /api/qaf/shape_presence/`

con la misma estructura (`poses`, `locale`).

````
