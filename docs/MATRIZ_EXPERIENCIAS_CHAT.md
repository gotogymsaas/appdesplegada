# Matriz de experiencias y botones UX del chat

## Fuentes revisadas
- Frontend (menú de servicios y botones): `frontend/js/chat.js`.
- Backend (catálogo oficial y router de quick actions): `backend/api/views.py`.

## Matriz Exp-001 a Exp-013

| Exp | Nombre (backend) | Presencia en menú frontend (`showServicesMenu`) | Payload frontend al iniciar | `start_actions` backend (primer CTA) | Notas UX de router backend |
|---|---|---|---|---|---|
| Exp-001 | Calorías Inteligentes (QAF) | Página `final` (`Calorías Inteligentes (QAF)`) | `service_intent: {experience: exp-001_calories, action: start_new}` | `Tomar foto de comida` (`open_camera`) | Desambiguación estándar: `Ver último` / `Iniciar evaluación nueva`. |
| Exp-002 | Coherencia Nutricional | Página `final` (`Coherencia Nutricional`) | `service_intent: {experience: exp-002_meal_coherence, action: start_new}` | `Evaluar coherencia de comida` (`message`) | Desambiguación estándar. |
| Exp-003 | Perfil Metabólico | Página `more` (`Perfil metabólico`) | `service_intent: {experience: exp-003_metabolic_profile, action: start_new}` | `Ejecutar evaluación metabólica` (`message`) con `metabolic_profile_request.start=true` | En desambiguación usa etiqueta especial `Ejecutar evaluación metabólica` (no `Iniciar evaluación nueva`). |
| Exp-004 | Menú Semanal | Página `more` (`Menú semanal`) | `service_intent: {experience: exp-004_meal_plan, action: start_new}` | `Generar menú semanal` (`message`) con `meal_plan_request` | Texto de introducción especial antes de CTAs. |
| Exp-005 | Tendencia 6 Semanas | Página `more` (`Tendencia 6 semanas`) | `service_intent: {experience: exp-005_body_trend, action: start_new}` | `Calcular tendencia 6 semanas` (`message`) | Desambiguación estándar. |
| Exp-006 | Corrección de Postura | Página `core` (`Corrección de postura`) | `service_intent: {experience: exp-006_posture, action: start_new}` | `Tomar foto frontal` (`posture_capture`) | Flujo guiado por capturas y botones de cancelación. |
| Exp-007 | Estado de Hoy (Lifestyle) | Página `more` (`Estado de hoy`) | `service_intent: {experience: exp-007_lifestyle, action: start_new}` | `Iniciar estado de hoy` (`message`) con `lifestyle_request.days=14` | Texto de introducción especial antes de CTAs. |
| Exp-008 | Motivación | Página `final` (`Motivación`) | `service_intent: {experience: exp-008_motivation, action: start_new}` | `Iniciar motivación` (`message`) con `motivation_request.preferences.pressure=suave` | Desambiguación estándar. |
| Exp-009 | Evolución de Entrenamiento | Página `more` (`Evolución entrenamiento`) | `service_intent: {experience: exp-009_progression, action: start_new}` | `Iniciar evolución` (`message`) | Desambiguación estándar. |
| Exp-010 | Progreso Muscular | Página `core` (`Progreso muscular`) | `service_intent: {experience: exp-010_muscle_measure, action: start_new}` | `Tomar foto frontal` (`muscle_capture`) | Flujo guiado por capturas y cancelación. |
| Exp-011 | Vitalidad de la Piel | Página `core` (`Vitalidad de la Piel`) | `service_intent: {experience: exp-011_skin_health, action: start_new}` | `Tomar foto` (`open_camera`) | Fast-path determinista adicional para evitar respuestas fuera de flujo. |
| Exp-012 | Alta Costura Inteligente | Página `core` (`Alta Costura Inteligente`) | `service_intent: {experience: exp-012_shape_presence, action: start_new}` | `Tomar foto frontal` (`shape_capture`) | Flujo de captura (1–2 vistas) con cancelación. |
| Exp-013 | Arquitectura Corporal | Página `core` (`Arquitectura Corporal`) | `service_intent: {experience: exp-013_body_architecture, action: start_new}` | `Tomar foto frente` (`pp_capture`) | Flujo de captura (2 obligatorias + opcional) con cancelación. |

## Mapa UX del menú frontend (13 experiencias)

- `core`: Exp-013, Exp-012, Exp-011, Exp-006, Exp-010.
- `more`: Exp-007, Exp-004, Exp-003, Exp-005, Exp-009.
- `final`: Exp-008, Exp-001, Exp-002.

## Hallazgos de consistencia UX

1. Hay doble fuente de verdad del catálogo:
   - Frontend define botones visibles por página.
   - Backend define catálogo oficial (`_service_spec`) y CTAs de inicio (`start_actions`).
2. El frontend incluye una normalización defensiva para Exp-003 en `appendQuickActions` para convertir CTAs viejos a `Ejecutar evaluación metabólica` y adjuntar payload metabólico.
3. El backend también protege Exp-003 con CTA y payload especial en la desambiguación (`Ver último` vs `Ejecutar evaluación metabólica`).

## Recomendación operativa

- Mantener sincronizado cualquier cambio de etiqueta/payload de experiencia en ambos lados (frontend + backend) para evitar loops o botones inconsistentes.
