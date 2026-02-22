import os
import sys
import json
from typing import Any

import requests


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def _die(msg: str, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _safe_json(resp: requests.Response) -> dict[str, Any] | None:
    try:
        return resp.json()
    except Exception:
        return None


def main() -> None:
    api_base = (_env("GTG_API_BASE", "https://api.gotogym.store/api/") or "").rstrip("/") + "/"
    username = _env("GTG_TEST_USERNAME")
    password = _env("GTG_TEST_PASSWORD")

    if not username or not password:
        _die(
            "Faltan credenciales en env vars. En tu terminal exporta:\n"
            "  export GTG_TEST_USERNAME='...'; export GTG_TEST_PASSWORD='...'\n"
            "Opcional: export GTG_API_BASE='https://api.gotogym.store/api/'"
        )

    # 1) Login
    login_url = api_base + "login/"
    r = requests.post(login_url, json={"username": username, "password": password}, timeout=30)
    if r.status_code != 200:
        body = (r.text or "").strip()
        _die(f"Login falló: HTTP {r.status_code}: {body[:500]}")

    data = _safe_json(r) or {}
    access = data.get("access")
    if not access:
        _die(f"Login OK pero no vino token access. Body: {json.dumps(data)[:500]}")

    headers = {"Authorization": f"Bearer {access}"}

    # 2) Cognición
    # Importante: en Django, si falta el slash final y APPEND_SLASH está activo,
    # puede ocurrir un redirect 301 que transforma el POST en GET (requests lo sigue).
    # Para evitar falsos 405, usamos slash final y evitamos seguir redirects aquí.
    cog_url = api_base + "qaf/cognition/evaluate/"
    r2 = requests.post(
        cog_url,
        headers=headers,
        json={"message": "hazme un análisis QAF"},
        timeout=30,
        allow_redirects=False,
    )
    if r2.status_code in (301, 302, 307, 308):
        loc = r2.headers.get("Location")
        if loc:
            r2 = requests.post(
                loc,
                headers=headers,
                json={"message": "hazme un análisis QAF"},
                timeout=30,
                allow_redirects=False,
            )

    cog_ok = r2.status_code == 200
    cog_json = _safe_json(r2) if cog_ok else None

    # 3) Chat (backend → n8n)
    chat_url = api_base + "chat/"
    r3 = requests.post(
        chat_url,
        headers=headers,
        json={
            "message": "Hola, prueba end-to-end. ¿Qué me recomiendas hoy?",
            "sessionId": "e2e_smoke",
            "attachment": "",
            "attachment_text": "",
            "username": username,
        },
        timeout=60,
    )
    chat_ok = r3.status_code == 200
    chat_json = _safe_json(r3) if chat_ok else None

    # Reporte (sin imprimir token)
    print("=== RESULTADOS ===")
    print(f"API_BASE: {api_base}")
    print(f"LOGIN: OK (HTTP {r.status_code})")

    if cog_ok and isinstance(cog_json, dict):
        decision = cog_json.get("decision") if isinstance(cog_json.get("decision"), dict) else {}
        print(f"COGNITION: OK (HTTP {r2.status_code}) mode={decision.get('mode')} type={decision.get('type')}")
    else:
        print(f"COGNITION: FAIL (HTTP {r2.status_code})")
        print((r2.text or "")[:500])

    if chat_ok and isinstance(chat_json, dict):
        out = chat_json.get("output")
        out_preview = (out or "")[:180].replace("\n", " ") if isinstance(out, str) else "(sin output string)"
        print(f"CHAT: OK (HTTP {r3.status_code}) output≈ {out_preview}")
    else:
        print(f"CHAT: FAIL (HTTP {r3.status_code})")
        print((r3.text or "")[:500])


if __name__ == "__main__":
    main()
