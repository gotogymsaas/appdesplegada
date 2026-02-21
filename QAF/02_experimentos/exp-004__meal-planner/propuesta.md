# Exp-004 — Propuesta v0 (QAF) — AI Meal Planner

## Estado

Definimos el estado semanal:

$$x = \{P, R, E, M, u\}$$

- $P$ perfil (kcal objetivo, meta, actividad)
- $R$ restricciones (exclusiones, dieta)
- $E$ entorno (disponibilidad; MVP: omitido)
- $M$ menú candidato (7 días × slots)
- $u$ incertidumbre (faltantes/limitaciones)

## Funcional a minimizar

$$F(M)= w_k L_{kcal} + w_\mu L_{micro} + w_v L_{variedad} + w_f L_{friccion} + w_u u$$

MVP:
- $L_{kcal}$: error cuadrático kcal/día vs objetivo.
- $L_{micro}$: 1 - cobertura micro (alto/medio).
- $L_{variedad}$: penaliza repetición de items.
- $L_{friccion}$: penaliza demasiados items distintos (lista de compras enorme).

## Operadores evolutivos

- mutación: reemplazar un item por otro del mismo “pool” (proteína/carb/veg/fat/fruta) y reajustar gramos.
- elitismo: conservar top-k.
- generaciones cortas: 10–25 (para latencia <5s).
