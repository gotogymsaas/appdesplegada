# Exp-003 — Propuesta v0 (QAF) — Perfil Metabólico Dinámico

## Estado QAF

Definimos el estado semanal como:

$$x_t = \{\hat{TMB}_t, \hat{TDEE}^{base}_t, \hat{\alpha}_t, \hat{TDEE}_t, \hat{K}_{rec,t}, u_t\}$$

- $\hat{TMB}_t$: basal (Mifflin St-Jeor) con sexo.
- $\hat{TDEE}^{base}_t$: gasto base por nivel de actividad.
- $\hat{\alpha}_t$: adaptación metabólica (0..0.25) — reduce/explica discrepancias.
- $\hat{TDEE}_t = \hat{TDEE}^{base}_t \cdot (1-\hat{\alpha}_t)$.
- $\hat{K}_{rec,t}$: recomendación calórica diaria para la semana.
- $u_t$: incertidumbre por calidad de datos.

## Observación robusta (peso)

En vez de usar un peso aislado, usamos:

- Mediana/promedio recortado de la semana actual y semana anterior.
- Detección de outliers por MAD (median absolute deviation).

Esto reduce cambios erráticos y mejora UX.

## Principio variacional

Definimos un costo operacional (energía libre):

$$F(\alpha, \Delta K) = w_e(\Delta W_{obs}-\Delta W_{target})^2 + w_a\alpha^2 + w_k\Delta K^2 + w_u u$$

Donde $\Delta K$ es el ajuste semanal de calorías recomendado (acotado a ±200 kcal por defecto).

El motor minimiza $F$ por búsqueda discreta (pequeña, estable, reproducible) y colapsa a confirmación humana cuando $u$ sea alta.
