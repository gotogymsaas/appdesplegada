# Investigación de costos Azure — Admin (27-02-2026)

## Contexto de autenticación y alcance
- Tenant: `Directorio predeterminado` (`b3dfdbaa-a2d7-421c-8d25-712e07a6c6c8`)
- Suscripción seleccionada: `Azure subscription 1` (`92b318a9-86bc-4734-9cc9-821767f6084f`)
- Alcance de consulta: **todas las suscripciones accesibles** (en este contexto, 1 suscripción).

## Inventario real detectado (Azure Resource Graph, muestra retornada)
> Nota: la respuesta de Resource Graph fue truncada en la consulta agregada; se listan aquí los 10 registros devueltos por la herramienta.

- `microsoft.cdn/profiles` en `gotogym-prod-rg` (1)
- `microsoft.cdn/profiles/afdendpoints` en `gotogym-prod-rg` (1)
- `microsoft.cognitiveservices/accounts` en `gotogym-speech` (1)
- `microsoft.cognitiveservices/accounts` en `n8n-gotogym` (1)
- `microsoft.communication/communicationservices` en `rg-gotogym-email` (1)
- `microsoft.communication/emailservices` en `rg-gotogym-email` (1)
- `microsoft.communication/emailservices/domains` en `rg-gotogym-email` (1)
- `microsoft.compute/disks` en `n8n-gotogym` (1)
- `microsoft.compute/sshpublickeys` en `n8n-gotogym` (2)
- `microsoft.compute/virtualmachines` en `n8n-gotogym` (1)

## Resource groups detectados
- `gotogym-prod-rg`
- `rg-gotogym-prod`
- `rg-gotogym-email`
- `gotogym-speech`
- `n8n-gotogym`
- `api`
- `gotogymweb`
- `rg-gotogym-llm-base`
- `ai_appi-gotogym-llm_*_managed`
- (otros grupos operativos/históricos)

## Mapa de costo sugerido para panel Admin

### 1) Compute/API
- App Service Plan + Web App (si aplica en `rg-gotogym-prod`/`api`)
- VM n8n + disco administrado (`n8n-gotogym`)
- Métrica panel: `monthly_cost_compute_cop`, `cost_per_request_compute_cop`

### 2) Datos
- PostgreSQL Flexible Server (si aplica)
- Métrica panel: `monthly_cost_db_cop`, `cost_per_active_user_db_cop`

### 3) Storage & Edge
- Blob Storage
- CDN / Front Door (`microsoft.cdn/*`)
- Métrica panel: `monthly_cost_storage_cop`, `monthly_cost_edge_cop`

### 4) IA/Servicios inteligentes
- Cognitive Services (`gotogym-speech`, `n8n-gotogym`)
- Métrica panel: `monthly_cost_ai_cop`, `cost_per_1k_tokens_or_calls_cop`

### 5) Comunicación
- Azure Communication Services Email (`rg-gotogym-email`)
- Métrica panel: `monthly_cost_email_cop`, `cost_per_email_cop`

### 6) Observabilidad
- Application Insights / Log Analytics (si aplica)
- Métrica panel: `monthly_cost_monitoring_cop`

## Estructura técnica recomendada para integrar costos reales

### Backend (nuevo endpoint recomendado)
`GET /api/admin/dashboard/cloud_costs/?days=30&currency=COP`

Respuesta sugerida:
```json
{
  "data": {
    "totals": {
      "monthly_cost_cop": 0,
      "monthly_cost_usd": 0,
      "cost_per_active_user_cop": 0
    },
    "by_service": [
      {"service": "compute", "cost_cop": 0, "cost_usd": 0},
      {"service": "database", "cost_cop": 0, "cost_usd": 0},
      {"service": "storage", "cost_cop": 0, "cost_usd": 0},
      {"service": "edge", "cost_cop": 0, "cost_usd": 0},
      {"service": "ai", "cost_cop": 0, "cost_usd": 0},
      {"service": "communication", "cost_cop": 0, "cost_usd": 0},
      {"service": "monitoring", "cost_cop": 0, "cost_usd": 0}
    ],
    "resource_groups": []
  },
  "meta": {
    "source": "azure_cost_management",
    "subscription_id": "92b318a9-86bc-4734-9cc9-821767f6084f"
  }
}
```

## Estado actual de la investigación
- ✅ Inventario base de recursos y RGs obtenido.
- ✅ Mapa de servicios para tablero definido.
- ⚠️ Falta consulta detallada de costos por recurso/servicio (Cost Management) para poblar valores reales.
- ⚠️ Resource Graph devolvió una respuesta truncada en la agregación global; se requiere extracción completa en siguiente iteración.
