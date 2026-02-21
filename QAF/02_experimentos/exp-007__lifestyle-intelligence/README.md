# Exp-007 — Lifestyle Intelligence Engine (QAF)

Objetivo: convertir señales diarias (sueño, movimiento, estrés proxy, actividad) en:

- `DHSS` (Daily Human State Score) 0..100 (con `confidence` + `uncertainty_score`)
- patrones (tendencias multi-día)
- micro-hábitos concretos (máx 3/día) sin repetición excesiva

Principios QAF:
- Estado + incertidumbre.
- Colapso humano: si falta señal crítica, pedir 1 confirmación mínima.
- Recomendaciones seguras (no diagnóstico médico).

## Qué hay aquí

- `scripts/qaf_lifestyle_intelligence.py`: motor QAF Exp-007.
- `scripts/run_local_eval.py`: runner JSONL para pruebas rápidas.
- `data/dataset.sample.jsonl`: ejemplos mínimos.
- `tests/test_qaf_lifestyle_intelligence.py`: tests unitarios.

## Entrada (MVP)

El motor puede operar con:
- `daily_metrics`: lista por día (`date`) con `steps`, `sleep_minutes`, `resting_heart_rate_bpm`, `calories`, etc.
- `self_report` opcional: `stress_1_5`, `sleep_quality_1_5`, `hydration_ml`, `sedentary_hours`.
- `memory`: historial de micro-hábitos para evitar repetición.

## Ejecución local

- `python QAF/02_experimentos/exp-007__lifestyle-intelligence/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-007__lifestyle-intelligence/data/dataset.sample.jsonl`
