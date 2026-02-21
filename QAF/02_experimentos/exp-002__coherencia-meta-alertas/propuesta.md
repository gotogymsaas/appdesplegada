# Exp-002 — Propuesta v0 (QAF): coherencia comida ↔ meta + alertas

## Problema
Queremos evaluar si una comida registrada es coherente con la meta del usuario (déficit, masa, mantenimiento) y producir alertas inteligentes.

Restricción actual: no siempre existe la meta explícita (p. ej. no hay `goal_weight` o `target_kcal`), y a veces solo existe `weight`/`age`/`height` o texto libre.

## Estado QAF (definición operacional)

Definimos un estado mínimo para la evaluación:

$$x \equiv \{g,\ \hat{K}_{day},\ \hat{K}_{meal},\ u,\ c\}$$

- $g$: meta (deficit/maintenance/gain) con confianza $c_g$.
- $\hat{K}_{day}$: objetivo calórico diario estimado.
- $\hat{K}_{meal}$: calorías de la comida.
- $u$: incertidumbre (porciones / incompletitud), 0..1.
- $c$: confianza global del contexto.

## Índice compuesto (coherencia)

La coherencia se calcula contra un objetivo por comida $K^*_{meal}$ derivado del diario:

- $K^*_{meal} = w_{slot} \cdot \hat{K}_{day}$

El score se penaliza por desviación relativa:

- $r = \hat{K}_{meal} / (K^*_{meal} + \epsilon)$
- score base: $S = \exp(-|\ln r|/\sigma)$  (0..1)
- penalización por incertidumbre/contexto: $S' = S \cdot (1 - 0.6u) \cdot c$

## Variacional (decisión de alertar)

Definimos un costo simple (energía libre operacional):

- $F = w_{over}\,\max(0, r-1)^2 + w_{under}\,\max(0, 1-r)^2 + w_u\,u + w_{miss}\,\mathbb{1}[\text{missing}]$

Si $F$ es alto o $u$ alto, el sistema colapsa en pedir confirmación (1 pregunta mínima) antes de afirmar.

## Alertas inteligentes (no clínicas)

Alertas típicas:

- `missing_goal`: no hay meta → pedir confirmación.
- `missing_weight`: sin peso → target diario muy incierto → advertir.
- `high_uncertainty`: porciones inciertas → confirmar primero.
- `over_target_meal`: comida muy alta vs meta.
- `under_target_meal`: comida muy baja vs meta (riesgo de hambre/ansiedad/atracón como señal operacional, sin prescribir).

## Integración (posterior)

Este Exp-002 puede consumir:

- salida del Exp-001 (`total_calories`, `uncertainty_score`, `needs_confirmation`) y
- contexto del usuario (perfil + texto del plan/documentos).
