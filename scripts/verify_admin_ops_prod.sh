#!/usr/bin/env bash
set -euo pipefail

API_BASE="${GTG_API_BASE:-https://api.gotogym.store/api}"
API_BASE="${API_BASE%/}"
DAYS="${GTG_DAYS:-30}"
TZ_NAME="${GTG_TZ:-America/Bogota}"

printf "GTG_ADMIN_USERNAME: "
read -r GTG_ADMIN_USERNAME
printf "GTG_ADMIN_PASSWORD: "
stty -echo
read -r GTG_ADMIN_PASSWORD
stty echo
printf "\n"

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

unset GTG_ADMIN_PASSWORD TOKEN
