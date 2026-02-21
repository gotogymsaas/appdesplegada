# Exp-005 — Predictor de Tendencias Corporales (Time Series)

Tipo: modelo predictivo temporal (series de tiempo) + simulación de escenarios.

Objetivo UX: “Si continúas así…” con proyección a 6 semanas, con banda de incertidumbre y colapso humano mínimo cuando faltan datos (kcal promedio).

## Entradas

- Perfil / metabólico: `tdee_kcal_day` (ideal), `recommended_kcal_day` (opcional)
- Peso: promedio semanal actual + previo (o lista de pesos)
- Ingesta: `kcal_in_avg_day` (si no está, se pide confirmación)

## Salidas

- `scenarios`: baseline, follow_plan, minus_200, plus_200
- Cada escenario: trajectory semanal (peso + banda)
- `confidence`, `uncertainty_score`, `follow_up_questions`

## Ejecución

- `python QAF/02_experimentos/exp-005__predictor-tendencias-corporales/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-005__predictor-tendencias-corporales/data/dataset.sample.jsonl`
