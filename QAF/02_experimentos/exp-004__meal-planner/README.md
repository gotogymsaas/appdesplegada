# Exp-004 — Generador Inteligente de Menús (AI Meal Planner)

Tipo: algoritmo heurístico / evolutivo.

Objetivo: crear un menú semanal personalizado bajo restricciones, optimizando:

- coherencia calórica
- variedad (baja repetición)
- cobertura de micronutrientes (proxy por `micros_db.csv`)
- baja fricción (pocas decisiones)

Este experimento reutiliza el catálogo nutricional existente (mismo que Exp-001):
`backend/api/qaf_calories/data/{nutrition_db.csv,micros_db.csv,items_meta.csv,aliases.csv}`.

## Ejecución

- `python QAF/02_experimentos/exp-004__meal-planner/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-004__meal-planner/data/dataset.sample.jsonl`
