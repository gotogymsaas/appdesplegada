# Checklist — Exp-001 (visión → calorías) — ejecución

Este checklist está pensado para iterar Exp-001 hasta tener un experimento **productizable**.

## P0 (hacer ahora)
- [x] Añadir `reasons`, `confidence_per_item` y `suggested_questions` al output.
- [x] Reemplazar la “entropía uniforme” por un `uncertainty_score` compuesto (item + porción).
- [x] Normalización con tabla de alias versionable (CSV) + fuzzy matching (sin dependencias pesadas).
- [x] Parser de porción por item (cuando el texto menciona el alimento).
- [x] Runner de benchmark: coverage, MAE, tasa de confirmación y cobertura del rango.

## P1 (siguiente iteración)
- [ ] Base nutricional versionada con fuente/país/fecha + macros (aunque no se muestren).
- [ ] Catálogo de medidas por alimento (taza/cucharada/vaso) y densidades.
- [ ] Dataset real (50–200) con ground-truth de gramos (balanza) y calorías.
- [ ] Baselines comparables: (1) Vision directo vs (2) Vision+QAF.

## P2 (si queremos ir más lejos)
- [ ] Integración con backend para capturar automáticamente salidas vision en JSONL.
- [ ] Memoria de porciones del usuario (sin guardar fotos) para reducir preguntas.
- [ ] Exploración de optimización discreta (clásica o cuántica) para selección bajo restricciones.
