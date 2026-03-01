#!/usr/bin/env bash
set -euo pipefail

API_BASE="${GTG_API_BASE:-https://api.gotogym.store/api}"
API_BASE="${API_BASE%/}"
DAYS="${GTG_DAYS:-30}"
TZ_NAME="${GTG_TZ:-America/Bogota}"

GTG_ADMIN_USERNAME="${GTG_ADMIN_USERNAME:-}"
GTG_ADMIN_PASSWORD="${GTG_ADMIN_PASSWORD:-}"

if [[ -z "${GTG_ADMIN_USERNAME}" ]]; then
  printf "GTG_ADMIN_USERNAME: "
  read -r GTG_ADMIN_USERNAME
fi

if [[ -z "${GTG_ADMIN_PASSWORD}" ]]; then
  printf "GTG_ADMIN_PASSWORD: "
  stty -echo
  read -r GTG_ADMIN_PASSWORD
  stty echo
  printf "\n"
fi

if [[ -z "${GTG_ADMIN_USERNAME}" || -z "${GTG_ADMIN_PASSWORD}" ]]; then
  echo "ERROR: Usuario/contraseña vacíos."
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq no está instalado."
  exit 2
fi

LOGIN_JSON=$(curl -sS -X POST "${API_BASE}/login/" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"${GTG_ADMIN_USERNAME}\",\"password\":\"${GTG_ADMIN_PASSWORD}\"}")

TOKEN=$(printf '%s' "$LOGIN_JSON" | jq -r '.access // empty')
if [[ -z "$TOKEN" ]]; then
  echo "ERROR: Login inválido o respuesta inesperada"
  echo "$LOGIN_JSON"
  unset GTG_ADMIN_PASSWORD
  exit 1
fi

check_endpoint() {
  local name="$1"
  local url="$2"
  local body hdr code ct
  body=$(mktemp)
  hdr=$(mktemp)
  code=$(curl -sS -o "$body" -D "$hdr" -w "%{http_code}" "$url" -H "Authorization: Bearer $TOKEN")
  ct=$(awk -F': ' 'tolower($1)=="content-type"{print tolower($2)}' "$hdr" | tr -d '\r' | head -n1)
  echo "$name|$code|$ct|$body"
}

OV=$(check_endpoint "overview" "${API_BASE}/admin/dashboard/overview/?days=${DAYS}&timezone=${TZ_NAME}&compare=true")
SG=$(check_endpoint "signups_series" "${API_BASE}/admin/dashboard/signups_series/?days=${DAYS}&timezone=${TZ_NAME}")
OP=$(check_endpoint "ops_metrics" "${API_BASE}/admin/dashboard/ops_metrics/?days=${DAYS}&timezone=${TZ_NAME}")

print_status() {
  local row="$1"
  local name code ct
  IFS='|' read -r name code ct _ <<< "$row"
  echo "- ${name}: HTTP ${code} (${ct:-sin-content-type})"
}

echo "=== ESTADO ENDPOINTS ADMIN DASHBOARD ==="
print_status "$OV"
print_status "$SG"
print_status "$OP"

echo ""
IFS='|' read -r _ op_code op_ct op_body <<< "$OP"

if [[ "$op_code" != "200" ]]; then
  echo "=== DIAGNÓSTICO OPS_METRICS ==="
  if [[ "$op_code" == "404" ]]; then
    echo "- El endpoint ops_metrics NO existe en la versión backend actualmente activa en producción."
    echo "- Acciones recomendadas:" 
    echo "  1) Ejecutar deploy API en main (workflow Deploy GoToGym API (auto))."
    echo "  2) Reiniciar App Service si el deploy marca success y sigue en 404."
    echo "  3) Re-validar hasta ver 401/403 sin token o 200 con token."
  else
    echo "- ops_metrics respondió HTTP ${op_code}."
  fi
  echo ""
  echo "Primeras líneas de respuesta:"
  sed -n '1,20p' "$op_body"
  unset GTG_ADMIN_PASSWORD TOKEN
  exit 1
fi

if ! echo "$op_ct" | grep -q "application/json"; then
  echo "ERROR: ops_metrics respondió 200 pero no JSON."
  sed -n '1,20p' "$op_body"
  unset GTG_ADMIN_PASSWORD TOKEN
  exit 1
fi

echo "=== VALORES BLOQUE OPERATIVO (PROD) ==="
jq '{
  costos: .data.costs,
  variables_operativas: .data.benchmark,
  usuarios_activos: .data.costs.active_users_range,
  series_dias: (.data.series | length),
  experiencias_total: (.data.experiences | length)
}' "$op_body"

echo ""
echo "=== VERIFICACIÓN AUTOMÁTICA (PASS/FAIL) ==="

source_value=$(jq -r '.data.costs.source // ""' "$op_body")
active_users=$(jq -r '.data.costs.active_users_range // 0' "$op_body")
cost_per_user=$(jq -r '.data.costs.cost_per_active_user_cop // "null"' "$op_body")
tokens_in=$(jq -r '.data.benchmark.tokens_in_total // 0' "$op_body")
tokens_out=$(jq -r '.data.benchmark.tokens_out_total // 0' "$op_body")
req_total=$(jq -r '.data.benchmark.requests_total // 0' "$op_body")
series_days=$(jq -r '(.data.series | length) // 0' "$op_body")
exp_total=$(jq -r '(.data.experiences | length) // 0' "$op_body")

pass=0
fail=0

check_item() {
  local ok="$1"
  local label="$2"
  local detail="$3"
  if [[ "$ok" == "1" ]]; then
    echo "[PASS] $label: $detail"
    pass=$((pass+1))
  else
    echo "[FAIL] $label: $detail"
    fail=$((fail+1))
  fi
}

check_item "$( [[ "$req_total" -gt 0 ]] && echo 1 || echo 0 )" "Actividad" "requests_total=$req_total"
check_item "$( [[ "$series_days" -gt 0 ]] && echo 1 || echo 0 )" "Serie temporal" "series_dias=$series_days"
check_item "$( [[ "$exp_total" -ge 13 ]] && echo 1 || echo 0 )" "Catálogo experiencias" "experiencias_total=$exp_total"
check_item "$( [[ "$source_value" == "azure_billing_csv" || "$source_value" == "azure_cost_management" ]] && echo 1 || echo 0 )" "Fuente de costo real" "source=$source_value"
check_item "$( [[ "$active_users" -gt 0 ]] && echo 1 || echo 0 )" "Usuarios activos" "active_users_range=$active_users"
check_item "$( [[ "$cost_per_user" != "null" && "$cost_per_user" != "0" && "$cost_per_user" != "0.0" ]] && echo 1 || echo 0 )" "Costo unitario" "cost_per_active_user_cop=$cost_per_user"
check_item "$( [[ "$tokens_in" -gt 0 || "$tokens_out" -gt 0 ]] && echo 1 || echo 0 )" "Tokens medidos" "tokens_in_total=$tokens_in tokens_out_total=$tokens_out"

echo ""
echo "Resumen: PASS=$pass FAIL=$fail"
if [[ "$fail" -gt 0 ]]; then
  echo "Acción: hay faltantes para un reporte financiero confiable en tiempo real."
else
  echo "Estado: bloque operativo listo para lectura ejecutiva/inversionistas."
fi

unset GTG_ADMIN_PASSWORD TOKEN
