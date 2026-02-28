# Checklist de implementación y validación en producción — Operativo Métricas

Fecha: 2026-02-28
Alcance: habilitar y verificar en producción `GET /api/admin/dashboard/ops_metrics/` para mostrar costos y variables operativas reales.

---

## 1) Verificación inicial (estado real PROD)
- [ ] Confirmar commit en `main` con endpoint:
  - [ ] `backend/api/views.py` contiene `def admin_dashboard_ops_metrics`
  - [ ] `backend/api/urls.py` contiene ruta `admin/dashboard/ops_metrics/`
- [ ] Confirmar que workflow API de deploy ejecuta sobre `main`.
- [ ] Confirmar estado endpoint sin auth:
  - [ ] `overview` => 401 (existe)
  - [ ] `signups_series` => 401 (existe)
  - [ ] `ops_metrics` => **debe dejar de estar en 404**

---

## 2) Implementación operativa para ver valores en PROD
- [ ] Ejecutar deploy API manual (`Deploy GoToGym API (auto)`).
- [ ] Esperar `conclusion=success` en GitHub Actions.
- [ ] Reiniciar App Service si endpoint sigue 404 tras deploy exitoso.
- [ ] Verificar nuevamente `ops_metrics` hasta obtener 401/403 sin token o 200 con token.

---

## 3) Validación automática (implementado)
Script disponible:
- `scripts/verify_admin_ops_prod.sh`

Qué hace:
- Login admin con password oculta.
- Verifica `overview`, `signups_series`, `ops_metrics`.
- Si `ops_metrics=200` imprime:
  - `costos` (`data.costs`)
  - `variables_operativas` (`data.benchmark`)
  - `usuarios_activos`
  - cantidad de días de serie y total de experiencias.
- Si `ops_metrics=404`, imprime diagnóstico y pasos de remediación.

Comando:

```bash
cd /home/juan/azuredev-1152/appdesplegada
./scripts/verify_admin_ops_prod.sh
```

Variables opcionales:
- `GTG_API_BASE` (default: `https://api.gotogym.store/api`)
- `GTG_DAYS` (default: `30`)
- `GTG_TZ` (default: `America/Bogota`)

---

## 4) Criterio de éxito para “ya se ve en producción”
- [ ] `ops_metrics` responde 200 con token admin.
- [ ] `data.costs` presente con al menos:
  - [ ] `estimated_total_cop`
  - [ ] `active_users_range`
  - [ ] `cost_per_active_user_cop`
  - [ ] `source`
- [ ] `data.benchmark` presente con al menos:
  - [ ] `requests_total`
  - [ ] `success_rate`
  - [ ] `avg_latency_ms_total`
  - [ ] `tokens_in_total`
  - [ ] `tokens_out_total`
- [ ] `data.series` con continuidad de días del rango.
- [ ] `data.experiences` con 13 experiencias.

---

## 5) Resultado actual (captura de diagnóstico)
- [x] Deploy workflow ejecutado con `success`.
- [x] `overview` y `signups_series` existen (401 sin auth).
- [ ] `ops_metrics` habilitado en producción (actualmente 404).
- [ ] Valores de costos/variables visibles en dashboard PROD.

Conclusión actual:
- Hasta que `ops_metrics` deje de responder 404, el bloque Operativo seguirá en fallback (ceros) y no habrá valores reales para mostrar.
