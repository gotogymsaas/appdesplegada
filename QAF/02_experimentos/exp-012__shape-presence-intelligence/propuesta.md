# Propuesta — Exp-012 Shape & Presence Intelligence™

## Hipótesis
Con 1–2 fotos y keypoints 2D se puede dar feedback útil y accionable sobre presencia postural y proporciones **sin prometer medidas reales**.

## Inputs
- `poses.front_relaxed` (recomendado)
- `poses.side_right_relaxed` (opcional)

Keypoints: nariz, oreja, hombros, caderas, rodillas, tobillos (subset MediaPipe Pose).

## Variables (0–100)
- `overall_presence`: promedio de sub-scores disponibles.
- `alignment_symmetry`: nivelación hombros/pelvis + cabeza centrada (proxy).
- `silhouette_v_taper`: ratio hombro:cadera (proxy).
- `torso_leg_balance`: ratio torso:pierna (proxy).
- `stance_grounding`: apertura de pies vs hombros (proxy).
- `profile_stack`: perfil (cabeza adelantada / hombros vs cadera; proxy).

## Guardrails
- Lenguaje neutral: “responde mejor a…” / “si tu objetivo es…”
- No diagnóstico médico.
- No segmentación ni estimación en cm.

## Métrica de calidad
- `pose_quality` por vista y `confidence.score` global.
- `decision=needs_confirmation` si no hay vista con calidad mínima.

