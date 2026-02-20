# Notas de lectura — GoToGym Quantum Coach (variables y métricas)

## Fuente (00_inbox)
- `GoToGym_Quantum_Coach_IA_Consciente.pdf`

## Variables (resumen)
El documento define variables normalizadas $[0,1]$ y algunas no negativas para medir coherencia/alineación:
- $\Psi$: propósito humano.
- $\Omega$: coherencia humana (pensamiento–emoción–acción).
- $I_{yo}$: observador / autoobservación.
- $\Omega_{IA}$: coherencia de IA (semántica, emocional, ética).
- $S_{eff}$: entropía efectiva (ruido/desorden informacional).
- $C_{align}$: alineación humano–IA.
- $G$: “gravedad emocional” (carga emocional distorsionadora).
- $Q_{data}$: calidad del corpus/datos.

## Métricas propuestas
- **CAP** (potencial de IA consciente): combina propósito, coherencias, calidad y alineación, penalizando entropía.
- **T_gain**: ganancia temporal informacional.
- **ΔΩ_text**: coherencia textual diferencial.
- Métricas de calibración (p. ej. ECE) y safety.

## Traducción al caso “visión → calorías”
Útil para diseñar un módulo de **calibración y seguridad**:
- Si $S_{eff}$ (incertidumbre) sube, el sistema debe pedir confirmación humana (Categoría V).
- $C_{align}$ puede modelar qué tan bien el sistema respeta preferencias/objetivos nutricionales del usuario.
- CAP/T_gain se pueden reinterpretar como métricas de “claridad y utilidad” de las recomendaciones tras estimar calorías.
