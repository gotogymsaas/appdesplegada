# Exp-001 — Visión → porción → calorías (QAF) — ejecutable

Objetivo: tener un primer experimento **funcionando de extremo a extremo** con:
- entrada: salida “vision” (JSON) por imagen
- salida: items normalizados + porción + calorías + incertidumbre + `needs_confirmation`
- evaluación: métricas simples vs ground-truth

## Qué hay aquí
- `scripts/qaf_vision_calories.py`: motor QAF v0 (normalización + porción + calorías + incertidumbre).
- `scripts/run_local_eval.py`: runner que lee un dataset `.jsonl`, ejecuta el motor y calcula métricas.
- `data/calorie_db.csv`: base MVP de calorías por 100g / por unidad.
- `data/dataset.sample.jsonl`: ejemplo del formato de dataset.

## Formato de dataset (JSONL)
Cada línea es un JSON con:
- `id` (string)
- `vision` (object): el JSON que retorna Vision (o el que guardes del chat). Debe incluir `items` (lista) y opcional `portion_estimate`.
- `ground_truth` (object, opcional):
  - `items`: lista de `{item_id, grams, calories}`
  - `total_calories`

Ver `data/dataset.sample.jsonl`.

## Ejecución (local)
Desde la raíz del repo:
- `python QAF/02_experimentos/exp-001__vision-calorias/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-001__vision-calorias/data/dataset.sample.jsonl`

## Criterio de “experimento cumplido” (MVP)
- El runner procesa el dataset sin errores.
- Devuelve calorías para al menos el 80% de ejemplos (con la base MVP).
- Marca `needs_confirmation=true` cuando la incertidumbre sea alta.

Siguiente paso: reemplazar el input `vision` por llamadas reales al backend (chat o un endpoint dedicado) y crecer el dataset.
