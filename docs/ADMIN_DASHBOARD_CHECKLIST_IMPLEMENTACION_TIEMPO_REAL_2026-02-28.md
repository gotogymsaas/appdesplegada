# Checklist de implementación — Admin Dashboard (Consolidación operativa + costo unitario en tiempo real)

Fecha: 2026-02-28
Referencia: `ADMIN_DASHBOARD_COSTO_UNITARIO_TIEMPO_REAL_2026-02-28.md`

Objetivo: implementar extremo a extremo el panel operativo del dashboard con métricas runtime confiables y costo unitario por usuario activo, con reconciliación de costo real Azure.

---

## 0) Alineación inicial (alcance y criterios)
- [ ] Confirmar que el endpoint principal será `GET /api/admin/dashboard/ops_metrics/` (sin romper compatibilidad).
- [ ] Definir `days` soportados (30/84/180/365/730) y timezone de negocio (`America/Bogota`).
- [ ] Definir criterio de “tiempo real operativo”: latencia objetivo de actualización en dashboard (ej. <= 60s para datos runtime).
- [ ] Definir criterio de “costo real vigente”: máximo `lag_hours` aceptable (ej. <= 24h).
- [ ] Acordar tabla de “Done” para aprobar salida (API, UI, seguridad, datos, observabilidad).

---

## 1) Backend — consolidación operativa (`ops_metrics`)

### 1.1 Contrato de respuesta estable
- [ ] Mantener estructura actual: `data.experiences`, `data.series`, `data.benchmark`, `data.costs`, `meta`.
- [ ] Asegurar continuidad de serie diaria (incluir días sin eventos con `0`).
- [ ] Validar que se devuelvan las 13 experiencias (`exp-001` a `exp-013`) + total diario.

### 1.2 Benchmark técnico
- [ ] Validar cálculo de `requests_total`, `requests_success`, `requests_error`.
- [ ] Validar cálculo de `success_rate` y `error_rate` con división segura (`R=0`).
- [ ] Excluir latencias inválidas (null/negativas) del promedio.
- [ ] Validar tokens acumulados (`tokens_in_total`, `tokens_out_total`).

### 1.3 Usuarios activos y costo unitario
- [ ] Confirmar `active_users_range` por actor único en trazas, con fallback por `last_login`.
- [ ] Confirmar `cost_per_active_user_cop = cost_total_cop / active_users_range` cuando `U_a > 0`.
- [ ] Retornar `cost_per_active_user_cop = null` cuando `U_a = 0`.

### 1.4 Campos de reconciliación (nuevo)
- [ ] Agregar en `data.costs`:
  - [ ] `reconciled_total_cop`
  - [ ] `reconciled_total_usd`
  - [ ] `reconciliation_lag_hours`
  - [ ] `estimation_error_pct`
  - [ ] `cost_source` (`estimated_tokens` | `azure_cost_management` | `azure_billing_csv`)
- [ ] Mantener fallback robusto a `estimated_tokens` si no hay costo real disponible.

---

## 2) Backend — costo real Azure (servicio auxiliar)

### 2.1 Endpoint auxiliar recomendado
- [ ] Crear `GET /api/admin/dashboard/cloud_costs_realtime/?scope=subscription&days={n}`.
- [ ] Validar permisos admin (`JWT` + `is_superuser`) y respuesta 403 para no-admin.
- [ ] Implementar caché corta (TTL 5–15 min) para proteger costo/latencia de consultas.

### 2.2 Integración Azure Cost Management
- [ ] Implementar consulta a `Microsoft.CostManagement/query` por scope (`subscription` o `resource group`).
- [ ] Configurar granularidad diaria y agrupación por `ServiceName` + `ResourceGroup`.
- [ ] Normalizar moneda (`USD`/`COP`) y conversión con `USD_TO_COP`.

### 2.3 Persistencia y resiliencia
- [ ] Crear snapshot interno (`cloud_cost_snapshots`) con timestamp de refresco.
- [ ] Guardar: totales, moneda, scope, `by_service`, `by_resource_group`, `lag_hours`.
- [ ] En fallo Azure, responder último snapshot válido + bandera de fuente.

---

## 3) Frontend — render operativo y UX de fuente de costo

### 3.1 Consumo de datos
- [ ] Mantener consumo de `ops_metrics` sin breaking changes.
- [ ] Mostrar 13 líneas + línea total en gráfica operativa.
- [ ] Verificar mapeo estable de catálogo de experiencias y etiquetas.

### 3.2 KPIs operativos
- [ ] Renderizar: requests, success rate, latencia, tokens, costo período, usuarios activos, costo unitario.
- [ ] Mostrar formato COP consistente (locale `es-CO`).
- [ ] Manejar estado vacío/fallback (`--` o `0` según corresponda).

### 3.3 Transparencia de costo
- [ ] Mostrar badge de `cost_source` (Estimado / Azure real / Billing CSV).
- [ ] Mostrar “frescura” (`last_refresh_utc` / `lag_hours`) en UI.
- [ ] Mostrar aviso cuando el costo sea estimado y no real.

---

## 4) Seguridad y cumplimiento operativo
- [ ] Confirmar que ningún secreto Azure esté en frontend.
- [ ] Guardar secretos en backend (idealmente Key Vault / identidad administrada).
- [ ] Aplicar RBAC mínimo para consulta de costos en scope definido.
- [ ] Auditar cada consulta de costos: quién, cuándo, scope, resultado.
- [ ] Aplicar rate limit + timeout a endpoints admin de costo.

---

## 5) Observabilidad y alertas
- [ ] Instrumentar latencia y tasa de error del endpoint `ops_metrics`.
- [ ] Instrumentar latencia y tasa de error del endpoint `cloud_costs_realtime`.
- [ ] Crear alertas por:
  - [ ] `success_rate` por debajo de umbral.
  - [ ] `reconciliation_lag_hours` por encima de umbral.
  - [ ] caída del pipeline de costo real.
- [ ] Registrar versión de payload en logs para trazabilidad de cambios.

---

## 6) Pruebas (obligatorias antes de producción)

### 6.1 Backend
- [ ] Test unitario de fórmulas (`success_rate`, `error_rate`, `cost_per_active_user_cop`).
- [ ] Test de rango/serie con días sin actividad (deben aparecer en `series`).
- [ ] Test de permisos (403 para no-admin).
- [ ] Test de fallback: sin costo real => `cost_source=estimated_tokens`.

### 6.2 Frontend
- [ ] Test de render de KPI con payload completo.
- [ ] Test de render con payload parcial/vacío.
- [ ] Test visual de gráfica 13+1 (sin romper en móvil).
- [ ] Test de badge de fuente y frescura.

### 6.3 Integración extremo a extremo
- [ ] `overview` responde 200.
- [ ] `signups_series` responde 200.
- [ ] `ops_metrics` responde 200.
- [ ] (nuevo) `cloud_costs_realtime` responde 200 para admin.
- [ ] Comparar costo estimado vs costo real en una ventana de prueba.

---

## 7) Despliegue y rollback
- [ ] Activar por feature flag (si aplica) los campos de reconciliación.
- [ ] Desplegar primero backend, luego frontend.
- [ ] Verificar integridad del dashboard con usuario admin real.
- [ ] Preparar rollback: volver a contrato anterior de `ops_metrics` en caso crítico.
- [ ] Documentar incidencia conocida y plan de mitigación si el costo real viene con atraso.

---

## 8) Definición de “DONE”
- [ ] Dashboard muestra KPIs operativos correctos y consistentes con backend.
- [ ] Gráfico operativo renderiza total + 13 experiencias en todos los rangos.
- [ ] Costo unitario se muestra con fuente (`cost_source`) y frescura (`lag_hours`).
- [ ] Seguridad validada (solo admin, sin secretos expuestos, RBAC mínimo).
- [ ] Pruebas backend/frontend/integración aprobadas.
- [ ] Monitoreo y alertas activas en producción.

---

## 9) Checklist rápido de validación final (Go/No-Go)
- [ ] API estable (sin breaking changes).
- [ ] Datos confiables (sin huecos, sin NaN, sin división por cero).
- [ ] UX clara (fuente de costo visible, fallback entendible).
- [ ] Seguridad aprobada.
- [ ] Monitoreo activo.
- [ ] Aprobación de negocio/operación.

Resultado:
- [ ] GO
- [ ] GO con observaciones
- [ ] NO-GO

---

## 10) Ejecución rápida para probar el servicio (implementado)

Script disponible:
- `scripts/e2e_smoke_admin_dashboard_ops.py`

Variables mínimas:
- `GTG_ADMIN_USERNAME`
- `GTG_ADMIN_PASSWORD`

Variables opcionales:
- `GTG_API_BASE` (default: `https://api.gotogym.store/api/`)
- `GTG_DAYS` (default: `30`)
- `GTG_TZ` (default: `America/Bogota`)
- `GTG_TIMEOUT_SEC` (default: `30`)
- `GTG_INCLUDE_OPTIONAL_CLOUD_COSTS` (`true`/`false`, default: `true`)

Comando de ejecución:

```bash
export GTG_ADMIN_USERNAME='tu_admin'
export GTG_ADMIN_PASSWORD='tu_password'
python scripts/e2e_smoke_admin_dashboard_ops.py
```

Qué valida automáticamente:
- Login admin
- `GET /api/admin/dashboard/overview/`
- `GET /api/admin/dashboard/signups_series/`
- `GET /api/admin/dashboard/ops_metrics/`
- `GET /api/admin/dashboard/cloud_costs_realtime/` (opcional; 404 se considera omitido)

Salida esperada:
- `RESULTADO FINAL: PASS` (exit code 0)
- Si falla algo: `RESULTADO FINAL: FAIL (...)` (exit code 1)
