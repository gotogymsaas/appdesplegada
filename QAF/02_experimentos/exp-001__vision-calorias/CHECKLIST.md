# Checklist — Exp-001 (visión → calorías) — implementación premium

Este checklist convierte Exp-001 en una experiencia “premium” (UX + motor), manteniendo compatibilidad con el JSON actual y sin añadir dependencias innecesarias.

## Fase 0 — Preparación (contrato y dataset)
- [ ] Definir el **contrato JSON premium** (entrada/salida) y reglas de compatibilidad.
	- Campos nuevos obligatorios/opcionales.
	- `decision`: `accepted|needs_confirmation|partial` + `decision_reason`.
	- `range_driver`: item que más mueve el rango.
	- `items[].portion_candidates[]`: exactamente 3 opciones.
	- `follow_up_questions[]`: lista ya priorizada para UI.
- [ ] Actualizar el dataset de ejemplo con 3 casos “oficiales”:
	- 1 ítem (huevo)
	- 2 ítems (arroz + pollo)
	- 3 ítems (ensalada + pan + leche)

## Fase 1 — UI premium sin escribir (candidates + follow-ups)
- [ ] Implementar `build_portion_candidates(item_id, locale="es-CO", *, calorie_db, memory_hint=None) -> list[dict]`.
	- Debe devolver **exactamente 3** opciones.
	- Cada opción debe incluir: `label`, `grams`, `confidence_hint`.
	- Si existe `unit_grams` → candidatos en unidades (1/2/3).
	- Si existe `default_serving_grams` → esa opción debe ser la central.
	- Si hay memoria reciente → opción #1 = “Repetir última porción”.
- [ ] Extender `items[]` en la salida:
	- `confidence` (ya existe)
	- `reasons` (ya existe a nivel global; duplicar/derivar por ítem si aplica)
	- `portion_candidates[]` (nuevo)
	- `selected_portion` (solo si el usuario confirmó o si se decide “aceptar” una porción inferida)
- [ ] Construir `follow_up_questions[]` listo para UI:
	- Una pregunta por ítem “driver” (máximo 1–2 preguntas por respuesta).
	- Reusar `portion_candidates` (no duplicar lógica).

## Fase 1.5 — Rango inteligente + “qué lo mueve”
- [ ] Cambiar el cálculo de `total_calories_range`:
	- Si falta porción o hay incertidumbre alta: rango = min/max derivado de candidates por ítem (no +/- fijo).
	- Si porción está clara: rango estrecho.
- [ ] Calcular `range_driver`:
	- ítem con mayor contribución a la varianza del total (aprox: spread_kcal del ítem * peso).
- [ ] Añadir `explainability` con máximo 2 bullets cortos orientados a usuario:
	- bullet 1: estimado y rango
	- bullet 2: qué ítem mueve el rango y qué confirmar

## Fase 2 — Memoria suave (sin guardar imágenes)
- [ ] Diseñar storage mínimo (sin dependencias):
	- archivo JSON local en `data/` (para experimento) con esquema `user_id + item_id -> grams + ts`.
	- funciones `get_last_portion(user_id, item_id)` y `save_confirmed_portion(user_id, item_id, grams)`.
	- ventana de recencia (14 días) para sugerir “Repetir”.
- [ ] Integrar memoria como `memory_hint` en `build_portion_candidates()`.

## Fase 3 — Colapso por optimización (score)
- [ ] Implementar `decision_score = w_uncertainty*U + w_default*D + w_missing*M + w_ask*A`.
	- U: incertidumbre total (agregada por kcal)
	- D: penalización por defaults usados
	- M: penalización por ítems faltantes en DB
	- A: costo de preguntar (cantidad de follow-ups)
- [ ] Derivar `decision`:
	- `accepted` si score bajo
	- `needs_confirmation` si score alto
	- `partial` si algunos ítems están OK y otros requieren confirmación
- [ ] Documentar pesos iniciales (`w_*`) y cómo calibrarlos con datos futuros.

## Tests mínimos (cada fase)
- [ ] Pruebas unitarias de `build_portion_candidates()` (exactamente 3 opciones y labels correctos).
- [ ] Pruebas de `range_driver` para casos 2 ítems y 3 ítems.
- [ ] Pruebas de `decision` (accepted/needs_confirmation/partial) con fixtures.
- [ ] Runner: imprimir métricas + un “golden output” (snapshot) para los 3 casos oficiales.

## Comandos de verificación (local)
- [ ] `python QAF/02_experimentos/exp-001__vision-calorias/scripts/run_local_eval.py --dataset QAF/02_experimentos/exp-001__vision-calorias/data/dataset.sample.jsonl`
- [ ] `python QAF/02_experimentos/exp-001__vision-calorias/scripts/qaf_vision_calories.py --calorie-db QAF/02_experimentos/exp-001__vision-calorias/data/calorie_db.csv --vision-json <JSON>`
