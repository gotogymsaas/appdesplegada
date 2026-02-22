# Contrato operativo — Motor de Cognición (QAF) — v0.1

Objetivo: unificar los outputs de los experimentos QAF en un **estado** $\psi(t)$ y una **decisión determinista**.

Reglas (Categoría V):
- El **LLM no decide**: solo narra/explica la decisión.
- El motor QAF devuelve: `state + indices + decision + policy`.
- Si hay señales de ruptura / baja calidad de datos: el motor **pide confirmación mínima** antes de proponer acciones.

---

## Endpoint

- `POST /api/qaf/cognition/evaluate/`
- Auth: JWT (mismo patrón que otros endpoints QAF)

### Uso desde n8n (recomendado)

- Nodo: **HTTP Request**
- Method: `POST`
- URL: `https://api.gotogym.store/api/qaf/cognition/evaluate/` (o tu URL de entorno)
- Headers:
  - `Authorization: Bearer <JWT>`
  - `Content-Type: application/json`
- Body JSON (mínimo):

```json
{
  "message": "{{$json.message}}",
  "week_id": "{{$json.week_id}}"
}
```

Resultado esperado:
- Guardar el JSON completo en una variable (por ejemplo `qaf_cognition`) y luego pasar **solo** `decision + state + indices` al LLM para narración.

### Input (JSON)

```json
{
  "week_id": "2026-W08",
  "locale": "es-CO",
  "message": "texto libre del usuario (opcional)",
  "observations": {
    "nutrition": {"score": 0.0},
    "training": {"score": 0.0},
    "health": {"score": 0.0},
    "quantum": {"score": 0.0}
  }
}
```

Notas:
- `observations` es opcional y sirve para pasar outputs “de esta interacción” cuando aún no están persistidos.
- Compatibilidad: si el caller envía `qaf_context` en lugar de `observations`, el endpoint lo usa.

### Output (JSON)

```json
{
  "success": true,
  "engine": {"name": "qaf-cognition", "version": "0.1"},
  "inputs": {"week_id": "2026-W08", "locale": "es-CO"},
  "dimensions": {
    "scores": {"nutrition": 0.5, "training": 0.5, "health": 0.5, "quantum": 0.5},
    "quality": {"nutrition": 0.6, "training": 0.6, "health": 0.6, "quantum": 0.6},
    "selected": "quantum"
  },
  "state": {
    "E": 0.6,
    "A": 0.6,
    "X": 0.5,
    "S": 0.4,

    "Psi": 0.6,
    "Omega": 0.6,
    "Iyo": 0.4,
    "Omega_IA": 0.7,
    "S_eff": 0.4,
    "C_align": 0.6,
    "G": 0.3,
    "Q_data": 0.5
  },
  "indices": {"coherence": 0.6, "impact": 0.6, "efficiency": 0.6, "CAP": 0.6},
  "flags": {"rupture_detected": false, "human_validation_required": false},
  "decision": {
    "mode": "quantum",
    "type": "proceed",
    "follow_up_questions": [],
    "next_3_actions": [
      {"id": "quantum_1", "title": "...", "timebox_minutes": 1}
    ]
  },
  "policy": {
    "llm_role": "narrate_only",
    "human_responsibility": true,
    "medical_disclaimer_required": true
  },
  "sources": {"persisted": {"lifestyle": false, "metabolic": false}}
}
```

---

## Mapeo: “13 experimentos → señales” (versión operativa)

Principio: cada experimento aporta 1–3 señales normalizadas (0..1) + calidad (`confidence`, `uncertainty_score`).

- Exp-001 (Visión calorías): `nutrition.score` (coherencia de porciones) + `quality` desde `uncertainty_score`.
- Exp-002 (Coherencia comida↔meta): `nutrition.score` desde `coherence_score`.
- Exp-003 (Perfil metabólico): `nutrition.quality` desde `confidence` + estabilidad (si existe).
- Exp-004 (Meal planner): `nutrition.score` según adherencia/variedad (si se modela).
- Exp-005 (Tendencia corporal): `nutrition/training.score` según trayectoria y adherencia.
- Exp-006 (Postura correctiva): `health.score` según `confidence` y severidad de labels.
- Exp-007 (Lifestyle/DHSS): aporta base de `E` y `S` (energía/estrés).
- Exp-008 (Motivación): aporta `quantum.score` (necesidad de claridad) y modula `G`.
- Exp-009 (Progresión): `training.score` desde readiness/adherencia.
- Exp-010 (Medición muscular): `health.score` (estado físico) + tendencia semanal.
- Exp-011 (Skin health): `health.score` (calidad de piel/guardrails, no diagnóstico).
- Exp-012 (Shape & Presence): `health.score` (presencia/alineación proxy).
- Exp-013 (Posture & Proportion): `health.score` (ASI) + ajustes de acción.

Este mapeo se implementa por etapas: primero consumiendo persistencia (`coach_state`, `coach_weekly_state`), y luego enriqueciendo con `observations` cuando aplique.
