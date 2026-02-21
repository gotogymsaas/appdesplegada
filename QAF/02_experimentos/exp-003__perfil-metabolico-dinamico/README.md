# Exp-003 — Perfil Metabólico Dinámico (QAF)

Tipo: modelo predictivo adaptativo.

Objetivo: calcular y recalibrar semanalmente un perfil metabólico operativo:

- **TMB** (tasa metabólica basal)
- **TDEE** (gasto diario estimado)
- **adaptación metabólica** (factor latente que modula TDEE)
- **calorías recomendadas** ajustadas semanalmente según meta y tendencia de peso

Diseñado para UX: “nutricionista personal” con cambios conservadores, explicación corta y colapso humano si faltan datos.

## Principio QAF aplicado

- Estado: $x_t = \{\hat{TMB}_t, \hat{TDEE}_t, \hat{\alpha}_t, \hat{K}_{rec,t}, u_t\}$
- Índices:
  - `confidence` (0..1)
  - `uncertainty_score` (0..1)
  - `stability` (0..1)
- Colapso (humano-in-the-loop): si la calidad de datos es baja o el peso es inconsistente, pedir 1 confirmación mínima antes de ajustar.

## Qué hay aquí

- `scripts/qaf_metabolic_profile.py`: motor Exp-003.
- `scripts/run_local_eval.py`: runner JSONL.
- `data/dataset.sample.jsonl`: dataset de ejemplo.
- `tests/test_qaf_metabolic_profile.py`: tests unitarios.

## Ejecución local

Desde la raíz del repo:

- `python QAF/02_experimentos/exp-003__perfil-metabolico-dinamico/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-003__perfil-metabolico-dinamico/data/dataset.sample.jsonl`
