# backend/devices/oauth/views_google_fit.py
import requests
import logging
from datetime import timedelta
from urllib.parse import urlencode

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
    error = request.query_params.get("error")
    code = request.query_params.get("code")
    state = request.query_params.get("state")

    if error:
        return Response({"ok": False, "error": error}, status=400)

    if not code or not state:
        return Response({"ok": False, "error": "Faltan code o state"}, status=400)

    try:
        user_id = _get_uid_from_state(state)
        User = get_user_model()
        user = User.objects.get(id=user_id)
    except Exception:
        return Response({"ok": False, "error": "State inválido"}, status=400)

    payload = {
        "client_id": settings.GOOGLE_FIT["WEB"]["CLIENT_ID"],
        "client_secret": settings.GOOGLE_FIT["WEB"]["CLIENT_SECRET"],
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": settings.GOOGLE_FIT["WEB"]["REDIRECT_URI"],
    }

    r = requests.post(
        settings.GOOGLE_FIT["TOKEN_URL"],
        data=payload,
        timeout=15,
    )
    data = r.json()


    if r.status_code != 200:
        return Response({"ok": False, "google": data}, status=400)

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
