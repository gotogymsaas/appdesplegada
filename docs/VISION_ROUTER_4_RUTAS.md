# Enrutamiento de imágenes (Vision Router) — 4 rutas finales

## Objetivo
Cuando el usuario adjunta una imagen al chat, el backend debe decidir **una** de estas rutas finales:

1. **Nutrición** (`route=nutrition`) — comida / etiqueta nutricional.
2. **Entrenamiento** (`route=training`) — persona en contexto de ejercicio o intención de técnica/postura.
3. **Salud** (`route=health`) — primer plano piel/músculo/rostro (salud/belleza/medición muscular).
4. **Quantum Coach (fallback)** (`route=quantum`) — cualquier otro caso o baja confianza.

La regla de oro es: **no intentar calorías si no es Nutrición**.

---

## Dónde vive
- Entrada principal: endpoint `POST /api/chat/`.
- Implementación: [backend/api/views.py](backend/api/views.py) en `chat_n8n()`.
- Vision call: `_describe_image_with_azure_openai()`.

---

## Contrato JSON esperado desde Vision
El backend pide a Vision que responda **solo JSON** con estas claves (compatibles con el histórico):

```json
{
  "route": "nutrition|training|health|quantum",
  "route_confidence": 0.0,
  "is_food": false,
  "items": [],
  "portion_estimate": "",
  "notes": "",
  "has_person": false,
  "has_nutrition_label": false,
  "is_closeup_skin_or_muscle": false
}
```

### Compatibilidad hacia atrás
- `is_food/items/portion_estimate/notes` se conservan porque alimentan Exp-001 (visión→calorías).
- `route/route_confidence` habilitan el router 4 rutas.

---

## Reglas de enrutamiento (decisión final)

### Nutrición
**Condición:** `route == nutrition` **o** `is_food == true`.

**Acciones:**
- Se permite ejecutar el motor de calorías (Exp-001 / `qaf_calories`).
- Se adjunta bloque `[CALORÍAS ESTIMADAS]` y opcionalmente coherencia con meta `[COHERENCIA CON META]`.

### Entrenamiento
**Condición:** `route == training`.

**Acciones:**
- No se ejecuta calorías.
- Se agrega bloque **Entrenamiento / Imagen** con guía de captura (frontal/lateral).
- Se agregan quick-actions: `posture_start`, `pp_start`, `shape_start`, `open_camera`, `open_attach`.
- Guardrail UX: se suprimen check-ins semanales/metabólico para no mezclar contexto.

### Salud
**Condición:** `route == health`.

**Acciones:**
- No se ejecuta calorías.
- Se agrega bloque **Salud / Imagen** que pide intención mínima:
  - “Comparar/medir músculo”
  - “Belleza/piel”
- Se agregan quick-actions de texto (message) con esas opciones.

### Quantum Coach
**Condición:** `route == quantum` o cuando Vision no está disponible.

**Acciones:**
- No se ejecuta calorías.
- No se fuerzan flujos; el mensaje se responde como coach general.

---

## Puntos críticos (guardrails)
- **Bloqueo de calorías:** el motor de calorías solo corre si se confirma Nutrición.
- **No diagnóstico médico:** en Salud se pide intención mínima y se responde prudente.
- **Privacidad/seguridad:** descarga de bytes de imagen solo si el blob pertenece al usuario.

---

## Cómo se refleja en el prompt hacia n8n
- `attachment_text` incluye un bloque `[DESCRIPCIÓN DE IMAGEN]` con `route`, `route_confidence`, etc.
- `qaf_context.vision` incluye el JSON parseado para reproducibilidad.

---

## Tests
- Test de regresión del router: [backend/api/test_chat_vision_router.py](backend/api/test_chat_vision_router.py)
  - Verifica que `route=training` **no** dispara `[CALORÍAS ESTIMADAS]`.
  - Verifica que se agrega **Entrenamiento / Imagen**.
