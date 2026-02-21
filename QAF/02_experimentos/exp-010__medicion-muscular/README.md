# Exp-010 — Medición muscular (solo fotografía) (QAF)

## Objetivo
Crear un algoritmo **solo con fotografías** (sin prometer diagnóstico médico) para estimar cambios y balances corporales con un enfoque “coach + wow”, enrutado por el Vision Router a la ruta final **Salud**.

Salida esperada:
- 7 variables (0–100) + `confidence`.
- Top 3 insights tipo coach.
- Rutina sugerida alineada a hallazgos.
- Soporte de comparación semanal (baseline vs semana actual) cuando exista historial.

## Entradas
- 4 fotos (misma sesión):
  1) Frente relajado
  2) Perfil derecho relajado
  3) Espalda relajado
  4) Frente “flex suave”
- Metadatos mínimos:
  - `week_id`
  - `user_id`/`username`
  - opcional: `height_cm` (solo para normalizar ratios; no para cm exactos)

## Restricción clave
Sin referencia física de escala (regla, cinta, marcador), **no** se puede prometer medición en centímetros fiable. El MVP debe reportar **métricas relativas** (ratios y cambios porcentuales) + un `confidence` alto/bajo según calidad de captura.

## Principios QAF aplicados
- Estado + incertidumbre.
- Colapso humano mínimo: pedir confirmación o repetir captura cuando `confidence` sea baja.
- Guardrails de seguridad (no diagnóstico médico).

## Carpeta
- `propuesta.md`: diseño QAF v0 (variables, contrato JSON, guardrails, dataset).

