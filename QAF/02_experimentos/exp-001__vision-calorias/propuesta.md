# Exp-001 — Algoritmo QAF para visión computacional de calorías (propuesta v0)

## Objetivo
Dado 1 o más imágenes de un plato/comida, producir:
- `is_food`: bool
- `items`: lista de alimentos probables (normalizados)
- `portion`: estimación de porción por ítem (con unidad)
- `calories`: estimación total y por ítem **con incertidumbre** (rango/percentiles)
- `needs_confirmation`: bool (si incertidumbre alta)

## Estado QAF (definición operacional)
Definimos el estado como un vector de hipótesis:
$$|\psi\rangle \equiv \sum_{k \in \mathcal{H}} c_k |h_k\rangle$$
Donde cada hipótesis $h_k$ representa una combinación:
- alimento (clase),
- preparación (frito/hervido/horneado),
- porción (clase o valor continuo discretizado),
- (opcional) condimentos/salsas.

La salida de visión inicial (percepción) se interpreta como probabilidades $p_k \approx |c_k|^2$ (no necesitamos fases para empezar).

## Módulos del algoritmo
### 1) Percepción (observación)
Entrada: imagen.
Salida: candidatos `items` + pistas de porción.
Implementación actual en producto: Azure OpenAI Vision en el backend (`_describe_image_with_azure_openai`).

### 2) Normalización (Categoría I: Estructura)
- Mapear items detectados a un **vocabulario canónico** (p. ej. `arroz_blanco`, `pollo_asado`, `ensalada_mixta`).
- Definir unidades base de porción: g, ml, “unidad”, “taza”, “rebanada”.
- Restricciones: coherencia nutricional (porciones físicamente plausibles), dieta del usuario, historial.

### 3) Dinámica e interacciones (Categoría II: Dinámica)
Codificar acoplamientos entre hipótesis (co-ocurrencia / exclusión):
- Ej.: `hamburguesa` acopla con `papas_fritas` (co-ocurrencia),
- `sopa` excluye `pizza_entera` en la misma imagen (dependiendo del contexto).

Operacionalmente: una matriz $H$ (o un grafo de factores) que repondera $p_k$ usando contexto.

### 4) Principio variacional (Categoría IV: Información y cognición)
Reformular como minimización de “energía libre”/negativa de evidencia:
- término de ajuste a observación (likelihood de visión),
- penalización por alta entropía (incertidumbre),
- penalización por complejidad (no inventar 10 items sin evidencia),
- priors del usuario (preferencias, país, hábitos, hora).

Salida: distribución posterior sobre hipótesis $q(h_k)$.

### 5) Ruptura / umbrales (Categoría III: Ruptura del vacío)
Definir umbrales prácticos:
- si entropía alta o conflicto fuerte entre hipótesis → activar `needs_confirmation=true`.
- si aparece un ítem “nuevo/no visto” → crear entrada candidata en diccionario (nueva “realidad” del sistema) pero marcada como provisional.

### 6) Colapso en decisión (Categoría V: Coordinación humano–IA)
Colapso = reporte final listo para registrar:
- Seleccionar top-k hipótesis por probabilidad.
- Convertir porciones a gramos/ml.
- Calcular calorías vía base nutricional.
- Si `needs_confirmation=true`: pedir al usuario 1 pregunta mínima (ej. “¿era arroz o pasta?” / “¿1 taza o 2?”).

## Cálculo de calorías (knowledge layer)
Se requiere una capa de conocimiento con:
- calorías por 100g (o por unidad) por ítem.
- ajuste por preparación (frito vs asado) y salsas.

Estrategia MVP:
1) tabla curada con 100–300 items más comunes.
2) fallback a un buscador (API tipo FoodData Central) con normalización y caching.

## Métricas
- Error absoluto medio (MAE) de calorías por comida.
- Cobertura del intervalo (si damos rango 90%, cuántas veces contiene el valor real).
- Tasa de confirmación (cuándo pedimos ayuda humana).
- Latencia end-to-end.

## Integración con el backend (punto de anclaje real)
Hoy el backend ya produce una descripción JSON con:
- `is_food`, `items`, `portion_estimate`, `notes`.

La propuesta es agregar una etapa posterior (servicio o función) que:
1) parsea ese JSON,
2) normaliza items,
3) estima porciones en unidades,
4) calcula calorías + incertidumbre,
5) devuelve un JSON extendido.

## Riesgos/limitaciones (honestos)
- Con una sola foto sin referencia de escala, la porción es la parte más difícil.
- “Algoritmo cuántico” aquí es **marco de inferencia/optimización inspirado** en dinámica cuántica; puede implementarse con simulación clásica. Si luego se desea, se puede probar una capa variacional en circuitos cuánticos pequeños para calibración, pero no es requisito para el MVP.
