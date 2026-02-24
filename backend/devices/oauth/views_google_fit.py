# backend/devices/oauth/views_google_fit.py
import requests
import logging
from datetime import timedelta
from urllib.parse import urlencode
import os

from django.conf import settings
from django.core import signing
from django.contrib.auth import get_user_model
from django.shortcuts import redirect
from django.utils import timezone
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import AccessToken

from devices.models import DeviceConnection
logger = logging.getLogger(__name__)
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny

GGOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def google_fit_authorize(request):
    """
    Construye el redirect REAL a Google OAuth:
    https://accounts.google.com/o/oauth2/v2/auth
    """

    logger.info("🚀 Entrando a google_fit_authorize")

    # Modo stub (QA local): permite probar UX sin credenciales reales.
    # IMPORTANT: No usar en producción.
    dev_stub = (os.getenv("OAUTH_DEV_STUB", "").strip() == "1")
    logger.info(f"📥 Query params: {request.query_params}")

    token = request.query_params.get("token")
    logger.info(f"🔑 Token presente: {'SI' if token else 'NO'}")

    if not token:
        logger.warning("❌ Falta token en la URL")
        return Response(
            {"detail": "Falta token en URL (?token=...)"},
            status=401
        )

    try:
        logger.info("🧪 Validando AccessToken(token)")
        access = AccessToken(token)  # valida firma y expiración

        user_id = access.get("user_id")
        logger.info(f"✅ user_id en token: {user_id}")

        User = get_user_model()
        user = User.objects.get(id=user_id)

    except Exception as e:
        logger.exception("❌ Token inválido o expirado")
        return Response(
            {"detail": "Token inválido o expirado"},
            status=401
        )

    if dev_stub:
        conn, _ = DeviceConnection.objects.get_or_create(user=user, provider="google_fit")
        conn.status = "connected"
        conn.access_token = "stub-access-token-google_fit"
        conn.refresh_token = "stub-refresh-token-google_fit"
        conn.token_expires_at = timezone.now() + timedelta(days=30)
        conn.save(update_fields=["status", "access_token", "refresh_token", "token_expires_at", "updated_at"])

        frontend_url = getattr(
            settings,
            "GOOGLE_FIT_FRONTEND_REDIRECT",
            "http://127.0.0.1:5500/pages/settings/Dispositivos.html",
        )
        return redirect(f"{frontend_url}?oauth=success&provider=google_fit")

    client_id = settings.GOOGLE_FIT["WEB"]["CLIENT_ID"]
    redirect_uri = settings.GOOGLE_FIT["WEB"]["REDIRECT_URI"]
    scope = settings.GOOGLE_FIT["SCOPE"]

    if not client_id or not redirect_uri:
        logger.error("❌ Faltan variables GOOGLE_FIT_* en settings.py")
        return Response(
            {"detail": "Configuración Google Fit incompleta"},
            status=500
        )

    state = signing.dumps({"uid": user.id}, salt="google-fit-oauth")

    # (Opcional) state para mantener referencia del usuario
    # Puedes usar request.user.id o un token temporal.

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
        "include_granted_scopes": "true",
    }

    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    return redirect(url)



def _get_uid_from_state(state: str) -> int:
    try:
        data = signing.loads(state, salt="google-fit-oauth", max_age=900)
        return int(data["uid"])
    except Exception:
        if state.isdigit():
            return int(state)
        raise ValueError("State inválido")

@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def google_fit_callback(request):
    dev_stub = (os.getenv("OAUTH_DEV_STUB", "").strip() == "1")

    error = request.query_params.get("error")
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if error:
        return Response({"ok": False, "error": error}, status=400)

    if not code or not state:
        if dev_stub and state:
            # Permite callback "manual" en QA local aunque no haya code.
            try:
                user_id = _get_uid_from_state(state)
                User = get_user_model()
                user = User.objects.get(id=user_id)
                conn, _ = DeviceConnection.objects.get_or_create(user=user, provider="google_fit")
                conn.status = "connected"
                conn.access_token = "stub-access-token-google_fit"
                conn.refresh_token = "stub-refresh-token-google_fit"
                conn.token_expires_at = timezone.now() + timedelta(days=30)
                conn.save(update_fields=["status", "access_token", "refresh_token", "token_expires_at", "updated_at"])
                frontend_url = getattr(
                    settings,
                    "GOOGLE_FIT_FRONTEND_REDIRECT",
                    "http://127.0.0.1:5500/pages/settings/Dispositivos.html",
                )
                return redirect(f"{frontend_url}?oauth=success&provider=google_fit")
            except Exception:
                pass

        return Response({"ok": False, "error": "Faltan code o state"}, status=400)

    try:
        user_id = _get_uid_from_state(state)
        User = get_user_model()
        user = User.objects.get(id=user_id)
    except Exception:
        return Response({"ok": False, "error": "State inválido"}, status=400)

    client_id = (settings.GOOGLE_FIT.get("WEB", {}).get("CLIENT_ID") or "").strip()
    client_secret = (settings.GOOGLE_FIT.get("WEB", {}).get("CLIENT_SECRET") or "").strip()
    redirect_uri = (settings.GOOGLE_FIT.get("WEB", {}).get("REDIRECT_URI") or "").strip()
    token_url = (settings.GOOGLE_FIT.get("TOKEN_URL") or "").strip() or "https://oauth2.googleapis.com/token"

    if not client_id or not client_secret or not redirect_uri:
        logger.error(
            "Google Fit OAuth config incompleta en callback (client_id=%s, secret_len=%s, redirect_uri=%s)",
            (client_id[:10] + "…") if client_id else "",
            len(client_secret or ""),
            redirect_uri,
        )
        return Response(
            {
                "ok": False,
                "error": "google_fit_config_incomplete",
                "detail": "Faltan GF_WEB_CLIENT_ID / GF_WEB_CLIENT_SECRET / GF_WEB_REDIRECT_URI en el backend.",
            },
            status=500,
        )

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }

    r = requests.post(token_url, data=payload, timeout=15)
    data = r.json() if r.content else {}

    if r.status_code != 200:
        err = (data.get("error") if isinstance(data, dict) else None) or "token_exchange_failed"
        desc = (data.get("error_description") if isinstance(data, dict) else None) or ""
        hint = ""

        if err == "invalid_client":
            hint = (
                "Google devolvió invalid_client. Verifica que GF_WEB_CLIENT_ID y GF_WEB_CLIENT_SECRET "
                "corresponden al MISMO OAuth Client de tipo 'Web application' en Google Cloud Console, "
                "y que no estás usando credenciales de Android/iOS. Si rotaste el secret, actualízalo en App Service y reinicia."
            )
        elif err in ("invalid_grant", "redirect_uri_mismatch"):
            hint = (
                "Revisa que GF_WEB_REDIRECT_URI coincida EXACTAMENTE (incluye https, dominio y trailing slash) "
                "con el Redirect URI configurado en el OAuth Client de Google."
            )

        logger.warning(
            "Google Fit token exchange failed (http=%s, error=%s, desc=%s, redirect_uri=%s, client_id_prefix=%s)",
            r.status_code,
            err,
            (desc or "")[:200],
            redirect_uri,
            (client_id[:10] + "…") if client_id else "",
        )

        return Response(
            {
                "ok": False,
                "error": "google_fit_token_exchange_failed",
                "google": data,
                "hint": hint,
            },
            status=400,
        )

    expires_at = timezone.now() + timedelta(seconds=int(data.get("expires_in", 0)))

    conn, _ = DeviceConnection.objects.get_or_create(
        user=user,
        provider="google_fit",
    )

    conn.access_token = data.get("access_token", "")
    if data.get("refresh_token"):
        conn.refresh_token = data["refresh_token"]

    conn.token_expires_at = expires_at
    conn.status = "connected"
    conn.save()

    frontend_url = getattr(
        settings,
        "GOOGLE_FIT_FRONTEND_REDIRECT",
        "http://127.0.0.1:5500/pages/settings/Dispositivos.html",
    )
    return redirect(f"{frontend_url}?oauth=success&provider=google_fit")
