import json
import os
import sys
from dataclasses import dataclass
from typing import Any

import requests


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _safe_json(response: requests.Response) -> dict[str, Any] | list[Any] | None:
    try:
        return response.json()
    except Exception:
        return None


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _bool_flag(name: str, default: bool = False) -> bool:
    raw = (_env(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _fail(message: str, code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def _expect_admin_login(api_base: str, username: str, password: str, timeout_sec: int) -> str:
    login_url = api_base + "login/"
    response = requests.post(login_url, json={"username": username, "password": password}, timeout=timeout_sec)
    if response.status_code != 200:
        body = (response.text or "").strip()
        _fail(f"Login falló: HTTP {response.status_code}: {body[:500]}")

    payload = _safe_json(response)
    if not isinstance(payload, dict):
        _fail("Login respondió sin JSON válido")

    access_token = payload.get("access")
    if not isinstance(access_token, str) or not access_token.strip():
        _fail(f"Login OK pero no vino access token. Body: {json.dumps(payload)[:500]}")
    return access_token


def _check_overview(api_base: str, headers: dict[str, str], days: int, timezone_name: str, timeout_sec: int) -> CheckResult:
    url = f"{api_base}admin/dashboard/overview/?days={days}&timezone={timezone_name}&compare=true"
    response = requests.get(url, headers=headers, timeout=timeout_sec)
    payload = _safe_json(response)
    if response.status_code != 200:
        return CheckResult("overview", False, f"HTTP {response.status_code} body={(response.text or '')[:180]}")
    if not isinstance(payload, dict):
        return CheckResult("overview", False, "respuesta no es JSON objeto")
    data = payload.get("data")
    required_keys = ["total_users", "premium_active"]
    if not isinstance(data, dict):
        return CheckResult("overview", False, "falta data en respuesta")
    missing = [key for key in required_keys if key not in data]
    if missing:
        return CheckResult("overview", False, f"faltan campos: {', '.join(missing)}")
    return CheckResult("overview", True, "ok")


def _check_signups(api_base: str, headers: dict[str, str], days: int, timezone_name: str, timeout_sec: int) -> CheckResult:
    url = f"{api_base}admin/dashboard/signups_series/?days={days}&timezone={timezone_name}"
    response = requests.get(url, headers=headers, timeout=timeout_sec)
    payload = _safe_json(response)
    if response.status_code != 200:
        return CheckResult("signups_series", False, f"HTTP {response.status_code} body={(response.text or '')[:180]}")
    if not isinstance(payload, dict):
        return CheckResult("signups_series", False, "respuesta no es JSON objeto")
    data = payload.get("data")
    if not isinstance(data, list):
        return CheckResult("signups_series", False, "data no es lista")
    return CheckResult("signups_series", True, f"ok rows={len(data)}")


def _check_ops_metrics(api_base: str, headers: dict[str, str], days: int, timezone_name: str, timeout_sec: int) -> CheckResult:
    url = f"{api_base}admin/dashboard/ops_metrics/?days={days}&timezone={timezone_name}"
    response = requests.get(url, headers=headers, timeout=timeout_sec)
    payload = _safe_json(response)
    if response.status_code != 200:
        return CheckResult("ops_metrics", False, f"HTTP {response.status_code} body={(response.text or '')[:180]}")
    if not isinstance(payload, dict):
        return CheckResult("ops_metrics", False, "respuesta no es JSON objeto")

    data = payload.get("data")
    if not isinstance(data, dict):
        return CheckResult("ops_metrics", False, "falta data")

    experiences = data.get("experiences")
    series = data.get("series")
    benchmark = data.get("benchmark")
    costs = data.get("costs")

    if not isinstance(experiences, list) or len(experiences) < 13:
        return CheckResult("ops_metrics", False, f"experiences inválido (esperado >=13, actual={0 if not isinstance(experiences, list) else len(experiences)})")
    if not isinstance(series, list):
        return CheckResult("ops_metrics", False, "series no es lista")
    if not isinstance(benchmark, dict):
        return CheckResult("ops_metrics", False, "benchmark inválido")
    if not isinstance(costs, dict):
        return CheckResult("ops_metrics", False, "costs inválido")

    required_benchmark = ["requests_total", "success_rate", "tokens_in_total", "tokens_out_total"]
    missing_benchmark = [key for key in required_benchmark if key not in benchmark]
    if missing_benchmark:
        return CheckResult("ops_metrics", False, f"faltan benchmark.{', benchmark.'.join(missing_benchmark)}")

    required_costs = ["estimated_total_cop", "active_users_range", "cost_per_active_user_cop", "source"]
    missing_costs = [key for key in required_costs if key not in costs]
    if missing_costs:
        return CheckResult("ops_metrics", False, f"faltan costs.{', costs.'.join(missing_costs)}")

    return CheckResult("ops_metrics", True, f"ok experiences={len(experiences)} series_days={len(series)} source={costs.get('source')}")


def _check_cloud_costs_optional(api_base: str, headers: dict[str, str], days: int, timeout_sec: int) -> CheckResult:
    url = f"{api_base}admin/dashboard/cloud_costs_realtime/?scope=subscription&days={days}"
    response = requests.get(url, headers=headers, timeout=timeout_sec)
    if response.status_code == 404:
        return CheckResult("cloud_costs_realtime", True, "omitido (endpoint opcional aún no implementado, 404)")

    payload = _safe_json(response)
    if response.status_code != 200:
        return CheckResult("cloud_costs_realtime", False, f"HTTP {response.status_code} body={(response.text or '')[:180]}")
    if not isinstance(payload, dict):
        return CheckResult("cloud_costs_realtime", False, "respuesta no es JSON objeto")

    data = payload.get("data")
    if not isinstance(data, dict):
        return CheckResult("cloud_costs_realtime", False, "falta data")
    totals = data.get("totals")
    if not isinstance(totals, dict):
        return CheckResult("cloud_costs_realtime", False, "falta data.totals")

    required = ["actual_cost_usd", "actual_cost_cop", "last_refresh_utc", "lag_hours"]
    missing = [key for key in required if key not in totals]
    if missing:
        return CheckResult("cloud_costs_realtime", False, f"faltan totals.{', totals.'.join(missing)}")
    return CheckResult("cloud_costs_realtime", True, "ok")


def main() -> None:
    api_base = (_env("GTG_API_BASE", "https://api.gotogym.store/api/") or "").rstrip("/") + "/"
    username = _env("GTG_ADMIN_USERNAME")
    password = _env("GTG_ADMIN_PASSWORD")
    timezone_name = _env("GTG_TZ", "America/Bogota") or "America/Bogota"
    days = int(_env("GTG_DAYS", "30") or "30")
    timeout_sec = int(_env("GTG_TIMEOUT_SEC", "30") or "30")
    include_optional_cloud_costs = _bool_flag("GTG_INCLUDE_OPTIONAL_CLOUD_COSTS", default=True)

    if not username or not password:
        _fail(
            "Faltan credenciales admin en env vars. Exporta:\n"
            "  export GTG_ADMIN_USERNAME='...'; export GTG_ADMIN_PASSWORD='...'\n"
            "Opcional: export GTG_API_BASE='https://api.gotogym.store/api/'\n"
            "Opcional: export GTG_DAYS='30'; export GTG_TZ='America/Bogota'"
        )

    access_token = _expect_admin_login(api_base, username, password, timeout_sec)
    headers = {"Authorization": f"Bearer {access_token}"}

    checks: list[CheckResult] = []
    checks.append(_check_overview(api_base, headers, days, timezone_name, timeout_sec))
    checks.append(_check_signups(api_base, headers, days, timezone_name, timeout_sec))
    checks.append(_check_ops_metrics(api_base, headers, days, timezone_name, timeout_sec))
    if include_optional_cloud_costs:
        checks.append(_check_cloud_costs_optional(api_base, headers, days, timeout_sec))

    print("=== SMOKE TEST ADMIN DASHBOARD (OPS) ===")
    print(f"API_BASE: {api_base}")
    print(f"DAYS: {days} | TZ: {timezone_name}")
    print("----------------------------------------")

    failures = 0
    for result in checks:
        status_text = "PASS" if result.ok else "FAIL"
        print(f"[{status_text}] {result.name}: {result.detail}")
        if not result.ok:
            failures += 1

    print("----------------------------------------")
    if failures == 0:
        print("RESULTADO FINAL: PASS")
        raise SystemExit(0)

    print(f"RESULTADO FINAL: FAIL ({failures} checks fallaron)")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
