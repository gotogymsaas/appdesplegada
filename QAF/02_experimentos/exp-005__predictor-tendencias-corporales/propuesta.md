# Exp-005 — Propuesta v0 (QAF)

## Estado

$$x_t = \{\hat{W}_t, \hat{TDEE}_t, \hat{\alpha}_t, K_{in,t}, u_t\}$$

## Dinámica

Balance energético (operacional):

$$\Delta W \approx \frac{(K_{in} - TDEE) \cdot 7}{7700}$$

Se simula a 6 semanas con escenarios:

- baseline: $K_{in}$ = ingesta promedio observada
- follow_plan: $K_{in}$ = recomendación actual
- ±200: sensibilidad

## Incertidumbre

La banda crece cuando faltan datos o hay alta variabilidad del peso. Usamos un ancho base (kg/semana) escalado por `uncertainty_score`.

## Colapso humano

Si no hay `kcal_in_avg_day`, pedir 1 dato mínimo: “¿Tu ingesta promedio diaria esta semana (kcal)?”
