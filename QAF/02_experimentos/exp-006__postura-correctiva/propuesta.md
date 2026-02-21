# Propuesta — Exp-006 Postura (QAF)

## Problema
Los usuarios piden “corregir postura” pero el valor real viene de:
1) guiar la captura (frontal + lateral)
2) análisis robusto con *pose estimation* (keypoints)
3) recomendaciones seguras (sin diagnóstico médico)

## Restricciones actuales (arquitectura)
- Frontend: vanilla JS + Capacitor/PWA (sin pipeline complejo).
- Backend: Django/DRF.
- Chat: `/api/chat/` proxya a n8n, y ya puede procesar adjuntos (fotos) usando un descriptor de visión.

Lo que NO está hoy:
- un modelo real de *pose estimation* integrado en backend.
- validación “física” 3D (solo 2D → limita curvaturas profundas).

## Enfoque MVP (realista)
1) Captura guiada (mensaje + checklist): pedir frontal y lateral.
2) Pose Estimation: fuera del backend por ahora (cliente o servicio dedicado). El motor de Exp-006 asume que recibe keypoints.
3) Métricas: simetrías (frontal) + adelantamiento cabeza/hombros (lateral) con normalización por escala corporal.
4) Clasificación híbrida (reglas): etiquetas simples y trazables.
5) Recomendaciones: catálogo de ejercicios + filtros por seguridad (dolor, lesión reciente) + “si hay dolor agudo consulta profesional”.

## Integración futura con chat
- El backend puede exponer `POST /api/qaf/posture/`.
- El chat puede enviar `posture_request` (keypoints) como payload, y el backend adjunta el resultado al mensaje.

## Métricas objetivo
- tasa de `needs_confirmation` cuando falta calidad (mejor que falsos positivos)
- estabilidad: resultados similares en capturas repetidas
- satisfacción UX (claridad y acción)
