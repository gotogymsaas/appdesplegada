# Checklist — Memoria Suave (Soft Memory) QAF (100%)

Objetivo: reducir incertidumbre y taps reutilizando porciones confirmadas por el usuario como **prior suave**, sin convertirlo en “verdad” y sin arrastrar errores.

---

## 0) Definición de alcance (antes de código)

- [ ] Definir “evento de escritura” (solo confirmación explícita):
  - [ ] Click en botones (`confirmed_portions`) ✅
  - [ ] Texto con gramos/unidades claramente parseable (si se soporta)
  - [ ] No escribir desde estimaciones del modelo
- [ ] Definir “evento de lectura”: solo cuando **no** hay porción explícita y/o cuando `portion_method` es `default/unknown`.
- [ ] Definir granularidad: memoria por `user_id + item_id` (mínimo). Opcional:
  - [ ] `context_bucket` (desayuno/almuerzo/cena)
  - [ ] `plate_size` (small/normal/large)
- [ ] Definir retención y privacidad:
  - [ ] No guardar imágenes
  - [ ] Guardar solo valores numéricos y timestamps
  - [ ] Retención (p.ej. 90 días) y borrado con cuenta

---

## 1) Modelo de datos (Django) + migración

**Recomendado**: tabla propia (evitar JSON en `User` a largo plazo).

- [ ] Crear modelo `QAFSoftMemoryPortion` (nombre sugerido), con:
  - [ ] `user` (FK)
  - [ ] `item_id` (string)
  - [ ] `context_bucket` (string nullable) **si aplica**
  - [ ] `plate_size` (string nullable) **si aplica**
  - [ ] `grams_last` (float)
  - [ ] `grams_ema` (float) (promedio exponencial) **o** `grams_median` (si guardas histórico corto)
  - [ ] `count_confirmed` (int)
  - [ ] `updated_at` (datetime)
  - [ ] `created_at` (datetime)
  - [ ] `last_source` (enum/string: button|text)
  - [ ] `reliability` (float 0–1) (opcional pero útil)
  - [ ] `spread_hint` (float) (opcional: variabilidad estimada)
- [ ] Índices/constraints:
  - [ ] `UniqueConstraint(user, item_id, context_bucket, plate_size)` (según campos usados)
  - [ ] Index por `(user, updated_at)`
- [ ] Migración aplicada y verificada en local.

---

## 2) Servicio interno (lectura/escritura) — API de memoria

- [ ] Crear módulo servicio (p.ej. `backend/api/qaf_calories/memory_service.py`) con funciones puras:
  - [ ] `get_memory_hints(user, item_ids, context) -> dict[item_id] = grams_hint`
  - [ ] `upsert_confirmed_portions(user, confirmed_portions, context)`
  - [ ] `apply_decay(record, now) -> effective_weight`
- [ ] Reglas de decaimiento (mínimo):
  - [ ] `age_days <= 3` → peso alto
  - [ ] `3 < age_days <= 7` → peso medio
  - [ ] `> 14 días` → peso bajo / ignorar
- [ ] Robustez:
  - [ ] Validar `grams` > 0 y límites razonables (p.ej. 1g–2000g)
  - [ ] Manejar concurrencia (`update_or_create` + transacción)

---

## 3) Wiring en el flujo actual del chat

### Lectura (antes de `qaf_estimate_v2`)
- [ ] Identificar usuario autenticado (`user` ya existe en `chat_n8n`).
- [ ] Construir lista de `item_ids` candidatos (ideal: después de extractor, o mínimo: después de `vision_parsed.items`).
- [ ] Llamar `get_memory_hints(...)` y pasar al motor como `memory_hint_by_item`.
- [ ] Confirmar que **solo** se usa memoria si:
  - [ ] no hay `confirmed_portions` para ese item
  - [ ] no hay porción explícita parseada

### Escritura (después de confirmación)
- [ ] Cuando llega `confirmed_portions` desde frontend:
  - [ ] Validar payload
  - [ ] Guardar por item con `upsert_confirmed_portions`
- [ ] (Opcional) Si parseas porciones explícitas desde texto del usuario, decidir si eso también escribe memoria.

---

## 4) Integración con Portion Engine (QAF)

- [ ] Si hay `memory_hint_grams`:
  - [ ] Usar como centro para 3 candidatos (`Repetir`, `Menos`, `Más`)
  - [ ] Marcar `portion_method = memory` cuando se use como base
  - [ ] Ajustar `portion_confidence(memory)` (p.ej. 0.85) sin llegar a 1.0
- [ ] Si hay `scale_hints.plate`:
  - [ ] Ajustar centro ±10% (ya existe en `portion_engine.build_portion_candidates`)

---

## 5) Integración con Governor (decisión)

- [ ] Si `portion_method == memory` y el registro es reciente/confiable:
  - [ ] Reducir `uncertainty_score` levemente **o** reducir umbral para `needs_confirmation`
- [ ] Nunca suprimir confirmación si:
  - [ ] `missing_items` no vacío
  - [ ] rango ancho por otro item (range_driver distinto)

---

## 6) Observabilidad (imprescindible en producción)

- [ ] Logging estructurado (sin PII sensible):
  - [ ] `memory_used=true/false` por request
  - [ ] `memory_hit_rate` (items con hint / items totales)
  - [ ] `memory_age_days`
  - [ ] `memory_overridden_by_user` (si el usuario elige opción muy distinta)
- [ ] Métricas:
  - [ ] % de requests que requieren confirmación antes/después
  - [ ] taps promedio por estimación
  - [ ] error proxy: cambios grandes entre memory y confirmación

---

## 7) Seguridad / privacidad / retención

- [ ] No guardar imágenes ni texto de la conversación como parte de la memoria.
- [ ] Política de retención:
  - [ ] job de limpieza (cron/management command) borrando >90 días
- [ ] Borrado por usuario:
  - [ ] al eliminar usuario o por request explícito (si existe feature)

---

## 8) Tests (mínimo necesario)

- [ ] Unit tests para `memory_service`:
  - [ ] upsert crea/actualiza
  - [ ] decaimiento por edad
  - [ ] validación de grams
- [ ] Unit tests para motor:
  - [ ] con `memory_hint_by_item` cambia candidatos y `portion_method`
  - [ ] con `confirmed_portions` ignora memoria
- [ ] Test de integración (Django):
  - [ ] 1) POST `chat/` con imagen → devuelve pregunta
  - [ ] 2) POST `chat/` con `confirmed_portions` → escribe memoria
  - [ ] 3) POST `chat/` nueva imagen similar → usa memoria y pregunta menos

---

## 9) Rollout y fallback

- [ ] Feature flag (env var): `QAF_SOFT_MEMORY_ENABLED=true/false`
- [ ] Si hay excepción leyendo/escribiendo:
  - [ ] no romper chat
  - [ ] continuar sin memoria (`memory_hint_by_item={}`)
- [ ] Plan de rollback:
  - [ ] desactivar flag

---

## 10) QA manual (teléfono)

- [ ] Caso A (primer uso): foto de comida → muestra kcal/rango y pide 1 confirmación.
- [ ] Caso B (confirmación): tocar botón de porción → responde con kcal/macro actualizados.
- [ ] Caso C (memoria): repetir comida similar → ofrece “Repetir última porción” y (ideal) no pregunta si la incertidumbre baja.
- [ ] Caso D (cambio): confirmar porción muy distinta → memoria se actualiza y no se “empecina” en la anterior.

---

## Definición de “100% listo”

- [ ] Memoria persistente por usuario en DB + decaimiento
- [ ] Lectura/escritura integradas al flujo del chat
- [ ] Feature flag + observabilidad
- [ ] Tests mínimos pasando
- [ ] QA en móvil validado (A–D)
