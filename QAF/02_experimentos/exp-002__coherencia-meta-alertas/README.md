# Exp-002 — Coherencia comida ↔ meta (déficit/masa/mantenimiento) + alertas inteligentes (QAF)

Objetivo: dado el registro de una comida (idealmente con calorías + incertidumbre desde Exp-001), estimar si esa comida es **coherente** con la meta del usuario:

- `deficit` (déficit calórico)
- `maintenance` (mantenimiento)
- `gain` (ganancia/masa)

…y generar **alertas inteligentes** (claras, accionables y no clínicas) cuando:

- Falte información crítica (meta no declarada, peso ausente, calorías ausentes).
- La comida se salga del rango esperado para la meta.
- La incertidumbre (porciones) sea alta: primero confirmar antes de concluir.

## Principio QAF aplicado (operacional)

- **Estado**: `state = {goal, daily_target, meal, uncertainty, context_confidence}`.
- **Índice compuesto**: `coherence_score` en 0..1 (1 = coherente), con penalizaciones por exceso/defecto y por baja calidad de información.
- **Colapso**:
  - si `goal_confidence` bajo o `uncertainty` alta → el sistema “colapsa” en pedir confirmación mínima.
  - si es estable → entrega evaluación y alertas.

## Qué hay aquí

- `scripts/qaf_goal_coherence.py`: motor (coherencia + alertas).
- `scripts/run_local_eval.py`: runner JSONL (dataset) para pruebas rápidas.
- `data/dataset.sample.jsonl`: dataset de ejemplo.
- `tests/test_qaf_goal_coherence.py`: tests unitarios.

## Formato de dataset (JSONL)

Cada línea es un JSON con:

- `id` (string)
- `user_context` (object): datos mínimos disponibles del usuario
  - `weight_kg` (number|null)
  - `age` (number|null)
  - `height_cm` (number|null)
  - `goal_type` ("deficit"|"maintenance"|"gain"|null)
  - `goal_text` (string|null) — texto libre si existe (ej: del plan o del chat)
  - `activity_level` ("low"|"moderate"|"high"|null)
- `meal` (object)
  - `total_calories` (number|null)
  - `uncertainty_score` (number|null) — 0..1 (si viene desde Exp-001)
  - `needs_confirmation` (bool|null) — si viene desde Exp-001
  - `meal_slot` ("breakfast"|"lunch"|"dinner"|"snack"|"unknown")

## Ejecución local

Desde la raíz del repo:

- `python QAF/02_experimentos/exp-002__coherencia-meta-alertas/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-002__coherencia-meta-alertas/data/dataset.sample.jsonl`
