# Exp-008 — Motivación Psicológica Personalizada (QAF)

Objetivo: que el usuario sienta:

1) “Me entiende” (perfil motivacional real, no genérico)
2) “Me acompaña” (constancia + memoria útil)
3) “Me impulsa” (reto exacto, sin exagerar)
4) “Me recompensa” (reconocimiento que le importa)

Este experimento implementa:
- Perfil motivacional como vector (no etiqueta única)
- Estado emocional diario (cambio de marcha)
- Motor anti-abandono (niveles 1–3)
- Micro-interacciones por chat (3–5 preguntas repartidas, sin test largo)

## Contenido

- `scripts/qaf_motivation_psych.py`: motor QAF Exp-008.
- `scripts/run_local_eval.py`: runner JSONL.
- `data/dataset.sample.jsonl`: casos mínimos.
- `tests/test_qaf_motivation_psych.py`: tests unitarios.

## Ejecución local

`python QAF/02_experimentos/exp-008__motivacion-psicologica/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-008__motivacion-psicologica/data/dataset.sample.jsonl`
