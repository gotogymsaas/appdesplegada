from django.shortcuts import render
from django.utils import timezone
import requests
from django.conf import settings
import base64
import os
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication

import secrets


from .models import DeviceConnection, FitnessSync
from api.models import User
from .serializers import DeviceConnectionSerializer

from .sync_service import sync_device
from .scheduler_service import enqueue_sync_request, run_due_sync

PROVIDERS = [
    {"provider": "apple_health", "label": "Apple Health"},
    {"provider": "google_fit", "label": "Google Fit"},
    {"provider": "fitbit", "label": "Fitbit"},
    {"provider": "garmin", "label": "Garmin"},
    {"provider": "whoop", "label": "WHOOP"},
]
PROVIDER_KEYS = {p["provider"] for p in PROVIDERS}


def _provider_catalog():
    # Apple Health no está soportado vía web OAuth en este MVP.
    apple_enabled = False
    apple_reason = "Próximamente (requiere integración nativa iOS)."

    # Importante: no dependemos de defaults en settings.py para "enabled".
    # Solo habilitamos si hay variables de entorno configuradas (producción real).
    gf_enabled = bool((os.getenv("GF_WEB_CLIENT_ID", "") or "").strip()) and bool((os.getenv("GF_WEB_CLIENT_SECRET", "") or "").strip()) and bool((os.getenv("GF_WEB_REDIRECT_URI", "") or "").strip())
    gf_reason = "Próximamente (Google Fit en configuración)." if not gf_enabled else ""

    fb_enabled = bool((os.getenv("FITBIT_CLIENT_ID", "") or "").strip()) and bool((os.getenv("FITBIT_CLIENT_SECRET", "") or "").strip()) and bool((os.getenv("FITBIT_REDIRECT_URI", "") or "").strip())
    fb_reason = "Configura credenciales de Fitbit" if not fb_enabled else ""

    garmin_enabled = bool((os.getenv("GARMIN_CLIENT_ID", "") or "").strip()) and bool((os.getenv("GARMIN_CLIENT_SECRET", "") or "").strip()) and bool((os.getenv("GARMIN_REDIRECT_URI", "") or "").strip()) and bool((os.getenv("GARMIN_AUTH_URL", "") or "").strip()) and bool((os.getenv("GARMIN_TOKEN_URL", "") or "").strip())
    garmin_reason = "Próximamente (Garmin en configuración)." if not garmin_enabled else ""

    whoop_enabled = bool((os.getenv("WHOOP_CLIENT_ID", "") or "").strip()) and bool((os.getenv("WHOOP_CLIENT_SECRET", "") or "").strip()) and bool((os.getenv("WHOOP_REDIRECT_URI", "") or "").strip())
    whoop_reason = "Próximamente (WHOOP en configuración)." if not whoop_enabled else ""

    return [
        {"provider": "apple_health", "label": "Apple Health", "enabled": apple_enabled, "disabled_reason": apple_reason},
        {"provider": "google_fit", "label": "Google Fit", "enabled": gf_enabled, "disabled_reason": gf_reason},
        {"provider": "fitbit", "label": "Fitbit", "enabled": fb_enabled, "disabled_reason": fb_reason},
        {"provider": "garmin", "label": "Garmin", "enabled": garmin_enabled, "disabled_reason": garmin_reason},
        {"provider": "whoop", "label": "WHOOP", "enabled": whoop_enabled, "disabled_reason": whoop_reason},
    ]

def _validate_provider(provider):
    if provider not in PROVIDER_KEYS:
        return False
    return True

def _clamp_score(value, min_val=0, max_val=10):
    try:
        return max(min_val, min(max_val, int(round(value))))
    except Exception:
        return min_val

def _score_from_steps(steps):
    # 0 steps -> 0, 12k steps -> 10
    return _clamp_score((steps / 12000.0) * 10.0)

def _score_from_sleep_minutes(minutes):
    # 0h -> 0, 8h -> 10
    hours = minutes / 60.0
    return _clamp_score((hours / 8.0) * 10.0)

def _refresh_google_fit_token(conn):
    if not conn.refresh_token:
        return False, {"error": "Falta refresh_token"}

    payload = {
        "client_id": settings.GOOGLE_FIT["WEB"]["CLIENT_ID"],
        "client_secret": settings.GOOGLE_FIT["WEB"]["CLIENT_SECRET"],
        "refresh_token": conn.refresh_token,
        "grant_type": "refresh_token",
    }

    r = requests.post(
        settings.GOOGLE_FIT["TOKEN_URL"],
        data=payload,
        timeout=15,
    )
    data = r.json()
    if r.status_code != 200:
        return False, {"error": "Refresh failed", "google": data}

    conn.access_token = data.get("access_token", conn.access_token)
    expires_in = int(data.get("expires_in", 0) or 0)
    if expires_in:
        conn.token_expires_at = timezone.now() + timezone.timedelta(seconds=expires_in)
    conn.save()
    return True, {"ok": True}

def _refresh_fitbit_token(conn):
    if not conn.refresh_token:
        return False, {"error": "Falta refresh_token"}

    client_id = settings.FITBIT["CLIENT_ID"]
    client_secret = settings.FITBIT["CLIENT_SECRET"]
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": conn.refresh_token,
    }

    r = requests.post(settings.FITBIT["TOKEN_URL"], data=payload, headers=headers, timeout=15)
    data = r.json()
    if r.status_code != 200:
        return False, {"error": "Refresh failed", "fitbit": data}

    conn.access_token = data.get("access_token", conn.access_token)
    if data.get("refresh_token"):
        conn.refresh_token = data["refresh_token"]
    expires_in = int(data.get("expires_in", 0) or 0)
    if expires_in:
        conn.token_expires_at = timezone.now() + timezone.timedelta(seconds=expires_in)
    conn.save()
    return True, {"ok": True}

def _refresh_garmin_token(conn):
    if not conn.refresh_token:
        return False, {"error": "Falta refresh_token"}

    client_id = settings.GARMIN.get("CLIENT_ID")
    client_secret = settings.GARMIN.get("CLIENT_SECRET")
    token_url = settings.GARMIN.get("TOKEN_URL")
    if not client_id or not client_secret or not token_url:
        return False, {"error": "Configuración Garmin incompleta"}

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": conn.refresh_token,
    }

    r = requests.post(token_url, data=payload, timeout=15)
    data = r.json()
    if r.status_code != 200:
        return False, {"error": "Refresh failed", "garmin": data}

    conn.access_token = data.get("access_token", conn.access_token)
    if data.get("refresh_token"):
        conn.refresh_token = data["refresh_token"]
    expires_in = int(data.get("expires_in", 0) or 0)
    if expires_in:
        conn.token_expires_at = timezone.now() + timezone.timedelta(seconds=expires_in)
    conn.save()
    return True, {"ok": True}

@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def devices_list(request):
    # Solo considerar realmente conectados en la lista de "Conectados"
    qs = DeviceConnection.objects.filter(user=request.user, status="connected")

    # Sync invisible al abrir (no ejecuta sync inline; encola si está viejo)
    try:
        now = timezone.now()
        stale_after = timezone.timedelta(hours=3)
        for conn in qs:
            if not conn.last_sync_at or (now - conn.last_sync_at) >= stale_after:
                recently = (
                    request.user.sync_requests.filter(
                        status="pending",
                        provider=conn.provider,
                        requested_at__gte=now - timezone.timedelta(minutes=30),
                    ).exists()
                )
                if not recently:
                    enqueue_sync_request(
                        request.user,
                        provider=conn.provider,
                        reason="app_open_devices",
                        priority=7,
                    )
    except Exception:
        pass

    return Response({
        "providers": _provider_catalog(),
        "connected": DeviceConnectionSerializer(qs, many=True).data
    })


@api_view(["POST"])
@permission_classes([AllowAny])
def internal_run_due_sync(request):
    """Endpoint interno para ejecutar sync programado.

    Autorización: header X-Internal-Token debe coincidir con INTERNAL_SYNC_TOKEN.
    """

    expected = (getattr(settings, "INTERNAL_SYNC_TOKEN", "") or "").strip()
    if not expected:
        return Response({"ok": False, "error": "scheduler_not_configured"}, status=503)

    provided = (request.headers.get("X-Internal-Token") or request.META.get("HTTP_X_INTERNAL_TOKEN") or "").strip()
    if not provided or not secrets.compare_digest(provided, expected):
        return Response({"ok": False, "error": "unauthorized"}, status=401)

    payload = request.data if isinstance(request.data, dict) else {}
    window = int(payload.get("window_minutes") or 15)
    max_users = int(payload.get("max_users") or 50)
    max_requests = int(payload.get("max_requests") or 50)
    result = run_due_sync(window_minutes=window, max_users=max_users, max_requests=max_requests)
    return Response(result)

@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def device_connect(request, provider):

    if not _validate_provider(provider):
        return Response({"detail": "Proveedor inválido"}, status=400)

    obj, _ = DeviceConnection.objects.get_or_create(user=request.user, provider=provider)

    # TODO: aquí irá OAuth real. Por ahora devolvemos una URL simulada.
    obj.status = "pending"
    obj.save()

    return Response({
        "device": DeviceConnectionSerializer(obj).data,
        "redirect_url": f"/oauth/{provider}/authorize/"
    })


@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def device_disconnect(request, provider):
    obj, _ = DeviceConnection.objects.get_or_create(user=request.user, provider=provider)
    obj.status = "disconnected"
    obj.access_token = ""
    obj.refresh_token = ""
    obj.token_expires_at = None
    obj.save()
    return Response(DeviceConnectionSerializer(obj).data)

@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def device_sync(request, provider):
    res = sync_device(request.user, provider)
    payload = res.payload if isinstance(res.payload, dict) else {}

    # Gamificación: si el sync trae datos reales, cuenta como acción del día y protege la racha.
    try:
        metrics = payload.get("metrics") if isinstance(payload, dict) else None
        has_real_data = False
        if isinstance(metrics, dict):
            if metrics.get("note") != "provider_not_implemented":
                if metrics.get("data_quality") == "ok":
                    has_real_data = True
                else:
                    for k in ("steps", "sleep_minutes", "calories", "distance_m", "distance_km"):
                        try:
                            if float(metrics.get(k) or 0) > 0:
                                has_real_data = True
                                break
                        except Exception:
                            continue

        if res.ok and has_real_data:
            from api.gamification_service import update_user_streak
            update_user_streak(request.user, source=f"device_sync:{provider}")
            payload["streak"] = int(getattr(request.user, "current_streak", 0) or 0)
    except Exception:
        # Nunca bloquear sync por gamificación.
        pass

    return Response(payload, status=res.status)

    # Default: simulated sync for other providers
    obj.last_sync_at = timezone.now()
    obj.save()
    return Response({"ok": True, "device": DeviceConnectionSerializer(obj).data})
