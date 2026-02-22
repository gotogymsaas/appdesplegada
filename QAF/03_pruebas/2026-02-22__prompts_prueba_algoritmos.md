# Prompts de prueba — Algoritmos QAF + Motor de Cognición

Fecha: 2026-02-22

Objetivo: tener prompts (texto) listos para copiar/pegar en el chat y también ejemplos de llamadas API para probar cada algoritmo QAF de forma consistente.

---

## Reglas de uso

- **Chat (UX)**: pega el prompt tal cual en el chat de la app.
- **API (QA)**: usa `curl` contra `/api/qaf/...` si quieres aislar el algoritmo.
- Donde veas `<JWT>`, reemplaza con tu token.
- Base URL (prod): `https://api.gotogym.store/api`

---

## Exp-001 — Visión calorías (qaf_calories)

### Chat (con foto)
- Adjunta una foto de tu comida y escribe:
  - `Calcula mis calorías y macros aproximados.`
  - `¿Cuántas calorías tiene esto?` 

> Nota: este experimento depende de visión/adjuntos. Sin imagen, el resultado puede pedir confirmación.

### API (si estás usando qaf_context/vision)
- Suele entrar por el flujo de chat con `qaf_context.vision`. Para prueba directa, usa el endpoint de chat:
  - `POST /api/chat/` con `qaf_context` y/o adjunto (según tu setup).

---

## Exp-002 — Coherencia comida ↔ meta (qaf_goal_coherence)

### Chat
- `Mi meta es bajar grasa. Me comí: 2 empanadas, gaseosa y un brownie. ¿Qué tan coherente fue?`

### API
- `POST /api/qaf/meal_coherence/`

Ejemplo:
```bash
curl -sS -X POST 'https://api.gotogym.store/api/qaf/meal_coherence/' \
  -H 'Authorization: Bearer <JWT>' \
  -H 'Content-Type: application/json' \
  -d '{
    "goal": {"type": "fat_loss"},
    "meal": {"text": "2 empanadas, gaseosa y brownie"}
  }'
```

---

## Exp-003 — Perfil metabólico dinámico (qaf_metabolic_profile)

### Chat
- `Quiero mi perfil metabólico semanal.`
- `¿Cómo voy esta semana con mi perfil metabólico?`

### API
- `POST /api/qaf/metabolic_profile/`

---

## Exp-004 — Meal planner (qaf_meal_planner)

### Chat
- `Hazme un plan de comidas para 7 días para bajar grasa, con comidas fáciles y baratas.`
- `Cámbiame el almuerzo del martes por algo con pollo.`

### API
- `POST /api/qaf/meal_plan/`
- `POST /api/qaf/meal_plan/mutate/`
- `POST /api/qaf/meal_plan/apply/`

---

## Exp-005 — Predictor tendencias corporales 6 semanas (qaf_body_trend)

### Chat
- `Proyéctame mi tendencia corporal de las próximas 6 semanas.`
- `Si sigo igual, ¿qué debería esperar en 6 semanas?`

### API
- `POST /api/qaf/body_trend/`

---

## Exp-006 — Postura correctiva (qaf_posture)

### Chat
- (Con imagen/postura o keypoints ya calculados) `Evalúa mi postura y dame una rutina correctiva.`

### API
- `POST /api/qaf/posture/`

> Nota: este experimento suele requerir `keypoints` (pose estimation) ya calculados.

---

## Exp-007 — Lifestyle Intelligence (qaf_lifestyle)

### Chat
- `Dormí 5 horas, hoy caminé poco y estoy estresado. ¿Cómo debería entrenar hoy?`
- `Me siento con poca energía y con ansiedad, ¿qué me recomiendas para hoy?`

### API
- `POST /api/qaf/lifestyle/`

---

## Exp-008 — Motivación psicológica (qaf_motivation)

### Chat
- `No tengo ganas de entrenar. Estoy procrastinando hace días. Ayúdame con un plan simple para volver.`
- `Siento que no avanzo y eso me frustra. ¿Qué reto me recomiendas hoy?`

### API
- `POST /api/qaf/motivation/`

---

## Exp-009 — Evolución de Entrenamiento (progresión inteligente) (qaf_progression)

### Chat (flujo UX)
1) `Evolución de Entrenamiento`
2) Botón: `Fuerza` o `Cardio`
3) Botón RPE: `2 / 4 / 6 / 8 / 10`
4) Botón cumplimiento: `20% / 40% / 60% / 80% / 100%`

### API
- `POST /api/qaf/progression/`

---

## Exp-010 — Medición muscular (qaf_muscle_measure)

### Chat
- `Quiero medir mi músculo y ver cómo voy.`
- `Compárame músculo vs la semana pasada.`

### API
- `POST /api/qaf/muscle_measure/`

---

## Exp-011 — Skin health (qaf_skin_health)

### Chat
- `Evalúa mi piel (rostro) y dame recomendaciones de cuidado.`

### API
- `POST /api/qaf/skin_health/`

---

## Exp-012 — Shape & Presence Intelligence (qaf_shape_presence)

### Chat
- `Evalúa mi shape/presencia corporal y dime qué mejorar.`

### API
- `POST /api/qaf/shape_presence/`

---

## Exp-013 — Posture & Proportion Intelligence (qaf_posture_proportion)

### Chat
- `Analiza mis proporciones y postura, y dime ajustes concretos.`

### API
- `POST /api/qaf/posture_proportion/`

---

# Prompt FINAL — Motor de Cognición (QAF)

Hay dos formas recomendadas de probarlo:

## A) Directo al endpoint (QA)

Endpoint:
- `POST /api/qaf/cognition/evaluate/`

Prompt/Body sugerido (copia y ajusta):
```json
{
  "week_id": "2026-W08",
  "locale": "es-CO",
  "message": "Estoy intentando retomar. Tuve mala semana: dormí poco, me salté entrenos, y hoy me siento estresado. Quiero una decisión clara: ¿qué priorizo hoy?",
  "observations": {
    "nutrition": {"score": 0.35, "quality": 0.7},
    "training": {"score": 0.45, "quality": 0.7},
    "health": {"score": 0.40, "quality": 0.6},
    "quantum": {"score": 0.70, "quality": 0.8}
  }
}
```

## B) Desde chat (E2E con n8n)

1) Envía un mensaje que active al menos 1–2 motores (por ejemplo Lifestyle + Evolución de Entrenamiento).
2) Luego pega:
- `Quiero que actives el motor de cognición y me digas qué priorizar hoy en 3 acciones.`

> En el backend, el payload a n8n ya se enriquece con `qaf_cognition` y `system_rules.qaf_cognition_summary` cuando hay contexto.
