# Contrato JSON premium — Exp-001 (visión → porción → calorías)

Objetivo: mantener compatibilidad con la salida actual y extenderla para UI premium (botones, rango inteligente, colapso).

## Entrada (compatible)
`vision` (object) — igual que hoy:
- `is_food` (bool, opcional)
- `items` (list[str])
- `portion_estimate` (str, opcional)
- `notes` (str, opcional)

## Entrada (extensiones opcionales para premium)
- `user_id` (str, opcional): id del usuario para memoria suave.
- `locale` (str, opcional, default `es-CO`).
- `confirmed_portions` (list[object], opcional): confirmaciones del usuario en esta interacción.
  - `{ "item_id": string, "grams": number }`

## Salida (base, existente)
- `is_food` (bool|null)
- `items` (list[object])
  - `{ item_id, grams, calories }`
- `missing_items` (list[str])
- `total_calories` (number)
- `total_calories_range` (object): `{ low, high }`
- `uncertainty` (object): incluye `uncertainty_score`
- `needs_confirmation` (bool)
- `confidence` (object): `total` + `per_item`
- `reasons` (list[str])

## Salida (premium)
- `decision` (string): `accepted|needs_confirmation|partial`
- `decision_reason` (string)
- `range_driver` (string|null): `item_id` que más mueve el rango
- `explainability` (list[str]): máximo 2 bullets cortos

### `items[]` (premium)
Además de `{item_id, grams, calories}`, cada item puede incluir:
- `portion_candidates` (list[object]): **exactamente 3** opciones
  - `{ label: string, grams: number, confidence_hint: number }`
- `selected_portion` (object|null): si el usuario confirmó (o si el motor decide fijarla)
  - `{ grams: number, source: "user"|"memory"|"inferred"|"default" }`
- `item_confidence` (number)
- `portion_confidence` (number)
- `reasons` (list[str]) (opcional; razones específicas del ítem)

### Preguntas de seguimiento (UI)
- `follow_up_questions` (list[object]): lista priorizada lista para UI de botones
  - `{ type: "confirm_portion", item_id, prompt, options: portion_candidates }`

Compatibilidad:
- Se mantiene `suggested_questions` como alias de `follow_up_questions`.
