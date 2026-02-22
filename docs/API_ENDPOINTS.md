# API Endpoints (GoToGym)

Base backend: Django + Django REST Framework.

Prefijos:
- `/api/` → endpoints principales (la mayoría con JWT)
- `/oauth/` → callbacks/authorize OAuth “directos” (GET)
- `/admin/` → Django Admin

Autenticación:
- JWT: `Authorization: Bearer <access>`
- Muchos endpoints aceptan `OPTIONS` para CORS.

## Auth

- `POST /api/register/`
- `POST /api/login/`
- `POST /api/token/` (SimpleJWT)
- `POST /api/token/refresh/` (SimpleJWT)
- `POST /api/password/reset/request/`
- `POST /api/password/reset/confirm/`

## Perfil y contexto (JWT)

- `PUT /api/update_profile/`
- `GET /api/user_profile/?username=...`
- `GET /api/coach_context/?username=...&include_text=0|1`

## IF / felicidad (JWT)

- `POST /api/predict_if/`
- `GET /api/if/question/`
- `POST /api/if/answer/`
- `GET /api/users/history/`
- `POST /api/update_score/`
- `GET /api/stats/global_history/`

## Chat (n8n)

- `POST /api/chat/`

Nota: el handler puede enriquecer la respuesta con resultados QAF (exp-003/004/005) y/o sobreescribir el texto `output`.

## Documentos (JWT)

- `GET /api/user_documents/`
- `POST /api/user_documents/delete/`

## Contacto

- `POST /api/contact/`

## Push (JWT)

- `POST /api/push/register/`
- `POST /api/push/unregister/`
- `POST /api/push/send_test/`
- `GET /api/push/web/key/`
- `POST /api/push/web/subscribe/`
- `POST /api/push/web/unsubscribe/`

### Push admin (JWT + superuser)

- `POST /api/push/admin/broadcast/` (requiere `reason`)

## Admin (JWT + superuser)

- `GET /api/admin/dashboard/overview/?date_from=YYYY-MM-DD&date_to=YYYY-MM-DD&tz=...&compare=0|1`
- `GET /api/admin/dashboard/signups_series/?date_from=...&date_to=...&tz=...`
- `GET /api/admin/audit/?page=1&pageSize=50&action=...&entity_type=...&entity_id=...`

## Billing

- `POST /api/billing/webhook/mercadopago/`

## QAF (JWT)

- `POST /api/qaf/meal_coherence/` (Exp-002)
- `POST /api/qaf/metabolic_profile/` (Exp-003)
- `POST /api/qaf/meal_plan/` (Exp-004)
- `POST /api/qaf/meal_plan/apply/` (Exp-004)
- `POST /api/qaf/meal_plan/mutate/` (Exp-004)
- `POST /api/qaf/body_trend/` (Exp-005)
- `POST /api/qaf/posture/` (Exp-006)
- `POST /api/qaf/lifestyle/` (Exp-007)
- `POST /api/qaf/motivation/` (Exp-008)
- `POST /api/qaf/progression/` (Exp-009)
- `POST /api/qaf/muscle_measure/` (Exp-010)
- `POST /api/qaf/skin_health/` (Exp-011)
- `POST /api/qaf/shape_presence/` (Exp-012)
- `POST /api/qaf/posture_proportion/` (Exp-013)

## Dispositivos / salud (JWT)

- `GET /api/devices/`
- `POST /api/devices/<provider>/connect/`
- `POST /api/devices/<provider>/disconnect/`
- `POST /api/devices/<provider>/sync/`

### Scheduler interno

- `POST /api/devices/internal/sync/run_due/` (protegido por `X-Internal-Token`)

## OAuth

Rutas disponibles por dos vías:

1) Bajo `/api/` (stub genérico):
- `GET /api/oauth/<provider>/authorize/`
- `GET /api/oauth/<provider>/callback/`

2) Bajo `/oauth/` (rutas “directas”):
- `GET /oauth/google_fit/authorize/`
- `GET /oauth/google_fit/callback/`
- `GET /oauth/fitbit/authorize/`
- `GET /oauth/fitbit/callback/`
- `GET /oauth/garmin/authorize/`
- `GET /oauth/garmin/callback/`
- `GET /oauth/whoop/authorize/`
- `GET /oauth/whoop/callback/`
- `GET /oauth/<provider>/authorize/` (genérico)
- `GET /oauth/<provider>/callback/` (genérico)

## Endpoints internos (AllowAny)

- `POST /api/internal/bootstrap_superuser/` (requiere header `X-Internal-Token`)
- `GET /api/internal/ocr_health/`
- `GET /api/internal/vision_health/`
- `POST /api/internal/ocr_extract/`
