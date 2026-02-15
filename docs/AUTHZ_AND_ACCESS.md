# Autorizaciones y accesos (GitHub + Azure)

Este documento describe el estado actual y el camino recomendado para administrar:
- Edición de documentación/código
- Control de merges a `main`
- Despliegues a Azure
- Gobierno para uso de Copilot en el repositorio

## Estado actual (detectado en este entorno)

### GitHub
- El repo `gotogymsaas/appdesplegada` es **público**.
- En este entorno, `gh` quedó autenticado y el permiso efectivo es **ADMIN**.
- Se habilitó protección de rama en `main` (ver sección de gobierno).
- Se crearon variables de repositorio para OIDC hacia Azure: `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_CLIENT_ID`.

### Azure
- Suscripción activa: `92b318a9-86bc-4734-9cc9-821767f6084f`.
- El usuario actual tiene **Owner** a nivel suscripción.

## Azure: OIDC recomendado (sin secretos largos)

### Qué se aplicó ya
Se creó una **User Assigned Managed Identity** y se configuró confianza OIDC (GitHub Actions → Azure) sin requerir permisos de directorio (Graph):
- Managed Identity: `gotogym-gha-oidc-mi`
- Resource group: `gotogym-prod-rg`
- Tenant: `b3dfdbaa-a2d7-421c-8d25-712e07a6c6c8`
- Subscription: `92b318a9-86bc-4734-9cc9-821767f6084f`
- `clientId`: `b5534ff1-ffa6-4700-b8d9-12a8d1dcb145`
- `principalId`: `36394ad3-695e-4d0b-93c4-46c58edbb300`

RBAC aplicado:
- `Website Contributor` sobre el Web App `gotogym-api-ca-7581`.
- `Contributor` sobre el Static Web App `GitHub`.

Credenciales federadas creadas:
- `repo:gotogymsaas/appdesplegada:ref:refs/heads/main`
- `repo:gotogymsaas/appdesplegada:pull_request`

### Cómo usarlo en GitHub Actions
Para autenticación OIDC (ejemplo conceptual), normalmente se requiere:
- Permisos del workflow: `id-token: write`
- Valores:
  - `AZURE_TENANT_ID=b3dfdbaa-a2d7-421c-8d25-712e07a6c6c8`
  - `AZURE_SUBSCRIPTION_ID=92b318a9-86bc-4734-9cc9-821767f6084f`
  - `AZURE_CLIENT_ID=b5534ff1-ffa6-4700-b8d9-12a8d1dcb145`

Nota: los workflows actuales usan `publish-profile` y token de SWA; la migración a OIDC puede hacerse por etapas para no interrumpir despliegues.

## GitHub: autorizaciones recomendadas (gobierno)

### Recomendado para administración
- Proteger `main`:
  - Requerir PR (sin pushes directos)
  - Requerir 1-2 approvals
  - Requerir status checks (CI)
  - Bloquear force-push
- Usar `CODEOWNERS` para rutas críticas (backend/infra/workflows).
- Separar permisos por equipo: `Admin/Maintain` solo para pocas cuentas.

#### Configuración aplicada
- `main` protegido con:
  - 1 aprobación requerida
  - Code owners requeridos
  - Historial lineal requerido
  - Resolución de conversaciones requerida
  - Admins también sujetos a la regla
- `CODEOWNERS` agregado en `.github/CODEOWNERS`.

### Copilot (administración)
- Habilitar Copilot para la organización/repositorio.
- Definir políticas:
  - Revisión obligatoria para cambios generados
  - Exclusiones de rutas sensibles (por ejemplo `.env`, credenciales, llaves)
  - Lineamientos de “no secrets en repo”

## Cómo completar la parte GitHub en este entorno
Auditoría rápida recomendada:
- `gh repo view gotogymsaas/appdesplegada --json nameWithOwner,viewerPermission,isPrivate,defaultBranchRef`
- `gh api repos/gotogymsaas/appdesplegada/branches/main/protection`
- `gh secret list -R gotogymsaas/appdesplegada`

Variables creadas para OIDC (no secretas):
- `gh variable set AZURE_TENANT_ID -R gotogymsaas/appdesplegada --body "b3dfdbaa-a2d7-421c-8d25-712e07a6c6c8"`
- `gh variable set AZURE_SUBSCRIPTION_ID -R gotogymsaas/appdesplegada --body "92b318a9-86bc-4734-9cc9-821767f6084f"`
- `gh variable set AZURE_CLIENT_ID -R gotogymsaas/appdesplegada --body "b5534ff1-ffa6-4700-b8d9-12a8d1dcb145"`

(Variables preferibles a secrets para estos valores porque **no son secretos**.)
