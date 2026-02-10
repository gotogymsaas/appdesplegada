import base64
import logging
from datetime import timedelta

import requests
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core import signing
from django.shortcuts import redirect
from django.utils import timezone
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import AccessToken

from devices.models import DeviceConnection

logger = logging.getLogger(__name__)


@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def fitbit_authorize(request):
    token = request.query_params.get("token")
    if not token:
        return Response({"detail": "Falta token en URL (?token=...)"}, status=401)

    try:
        access = AccessToken(token)
        user_id = access.get("user_id")
        User = get_user_model()
        user = User.objects.get(id=user_id)
    except Exception:
        return Response({"detail": "Token inválido o expirado"}, status=401)

    client_id = settings.FITBIT["CLIENT_ID"]
    redirect_uri = settings.FITBIT["REDIRECT_URI"]
    scope = settings.FITBIT["SCOPE"]
    auth_url = settings.FITBIT["AUTH_URL"]

    if not client_id or not redirect_uri:
        return Response({"detail": "Configuración Fitbit incompleta"}, status=500)

    state = signing.dumps({"uid": user.id}, salt="fitbit-oauth")

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "state": state,
    }

    url = auth_url + "?" + requests.compat.urlencode(params)
    return redirect(url)


def _get_uid_from_state(state: str) -> int:
    try:
        data = signing.loads(state, salt="fitbit-oauth", max_age=900)
        return int(data["uid"])
    except Exception:
        if state.isdigit():
            return int(state)
        raise ValueError("State inválido")


@api_view(["GET"])
@authentication_classes([])
@permission_classes([AllowAny])
def fitbit_callback(request):
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

    client_id = settings.FITBIT["CLIENT_ID"]
    client_secret = settings.FITBIT["CLIENT_SECRET"]
    token_url = settings.FITBIT["TOKEN_URL"]
    redirect_uri = settings.FITBIT["REDIRECT_URI"]

    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {
        "client_id": client_id,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
        "code": code,
    }

    r = requests.post(token_url, data=payload, headers=headers, timeout=15)
    data = r.json()

    if r.status_code != 200:
        return Response({"ok": False, "fitbit": data}, status=400)

    expires_at = timezone.now() + timedelta(seconds=int(data.get("expires_in", 0) or 0))

    conn, _ = DeviceConnection.objects.get_or_create(
        user=user,
        provider="fitbit",
    )
    conn.access_token = data.get("access_token", "")
    if data.get("refresh_token"):
        conn.refresh_token = data["refresh_token"]
    conn.token_expires_at = expires_at
    conn.status = "connected"
    conn.save()

    frontend_url = getattr(
        settings,
        "FITBIT_FRONTEND_REDIRECT",
        "http://127.0.0.1:5500/pages/settings/Dispositivos.html",
    )
    return redirect(f"{frontend_url}?oauth=success&provider=fitbit")
