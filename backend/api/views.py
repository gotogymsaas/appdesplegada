from rest_framework.decorators import api_view
from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.response import Response
from rest_framework import status
import logging
import re
import threading
import secrets
from typing import Any

from devices.models import DeviceConnection, FitnessSync as DevicesFitnessSync
from devices.scheduler_service import enqueue_sync_request

import requests
import json
import traceback
import hashlib
import ipaddress
from django.contrib.auth import authenticate
from .models import (
    User,
    HappinessRecord,
    IFQuestion,
    IFAnswer,
    UserDocument,
    ContactMessage,
    PushToken,
    TermsAcceptance,
    WebPushSubscription,
    AuditLog,
    QAFSoftMemoryPortion,
)
from .if_questions import IF_QUESTIONS
from .serializers import UserSerializer
# Importar el servicio ML
from .serializers import UserSerializer
# Importar el servicio ML
from .gamification_service import update_user_streak, build_gamification_status
from datetime import date, datetime, timedelta, timezone as dt_timezone
from django.utils import timezone
from django.db.models import Avg, Q, Count
from django.db import IntegrityError
from django.db.models.functions import TruncDate
import sys
try:
    from PIL import Image
except Exception:
    Image = None
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except Exception:
    pass
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    from azure.communication.email import EmailClient
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError
except Exception:
    EmailClient = None
    AzureKeyCredential = None
    HttpResponseError = None
import os
import uuid
import tempfile
import base64
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from django.conf import settings
from django.core import signing
from django.core.mail import EmailMessage, EmailMultiAlternatives
from django.utils.text import get_valid_filename
from zoneinfo import ZoneInfo
try:
    from pywebpush import webpush, WebPushException
except Exception:
    webpush = None
    WebPushException = Exception

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions
except Exception:
    BlobServiceClient = None
    ContentSettings = None
    generate_blob_sas = None
    BlobSasPermissions = None


logger = logging.getLogger(__name__)


def _normalize_height_cm_from_user_value(user_height_value):
    """Normaliza una altura almacenada en `User.height` a centímetros.

    En el sistema, `User.height` se espera en metros (p.ej. 1.75), pero pueden existir
    datos legacy en cm (p.ej. 175) o errores por doble conversión (p.ej. 17500, 1750000).
    """
    try:
        h = float(user_height_value)
    except Exception:
        return None
    if not h or h <= 0:
        return None

    # Caso nominal: metros
    height_cm = h * 100.0 if h <= 3.0 else h

    # Deshacer errores de escala comunes (cm*100, cm*100*100)
    try:
        while height_cm > 30000.0:
            height_cm = height_cm / 100.0
    except Exception:
        return None

    if 300.0 < height_cm <= 30000.0:
        height_cm = height_cm / 100.0
    if 0.5 < height_cm < 3.0:
        height_cm = height_cm * 100.0

    # Guardrail final: rango humano razonable
    if height_cm < 80.0 or height_cm > 260.0:
        return None
    return float(height_cm)


def _qaf_catalog_paths():
    base = Path(__file__).resolve().parent / "qaf_calories" / "data"
    return {
        "base": base,
        "aliases": base / "aliases.csv",
        "items_meta": base / "items_meta.csv",
        "calorie_db": base / "calorie_db.csv",
        "nutrition_db": base / "nutrition_db.csv",
        "micros_db": base / "micros_db.csv",
    }


class IsAuthenticatedOrOptions(IsAuthenticated):
    def has_permission(self, request, view):
        if request.method == "OPTIONS":
            return True
        return super().has_permission(request, view)


def _parse_timezone(tz_str: str) -> ZoneInfo:
    value = (tz_str or "").strip() or "America/Bogota"
    try:
        return ZoneInfo(value)
    except Exception:
        return ZoneInfo("America/Bogota")


def _parse_date_yyyy_mm_dd(value: str):
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except Exception:
        return None


def _date_range_from_request(request, default_days=90):
    qs = request.query_params if hasattr(request, "query_params") else request.GET
    tz = _parse_timezone(qs.get("timezone") or "")

    date_from = _parse_date_yyyy_mm_dd(qs.get("dateFrom") or qs.get("from") or "")
    date_to = _parse_date_yyyy_mm_dd(qs.get("dateTo") or qs.get("to") or "")

    if date_to is None:
        # hoy en TZ
        now_local = timezone.now().astimezone(tz)
        date_to = now_local.date()
    if date_from is None:
        try:
            days = int(qs.get("days") or default_days)
        except Exception:
            days = default_days
        days = max(1, min(days, 730))
        date_from = date_to - timedelta(days=days - 1)

    # convertimos a datetimes aware en UTC
    start_local = datetime(date_from.year, date_from.month, date_from.day, 0, 0, 0, tzinfo=tz)
    end_local = datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59, tzinfo=tz)
    start_utc = start_local.astimezone(dt_timezone.utc)
    end_utc = end_local.astimezone(dt_timezone.utc)

    return {
        "tz": tz,
        "date_from": date_from,
        "date_to": date_to,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "days": (date_to - date_from).days + 1,
    }


def _previous_period(range_info):
    days = int(range_info.get("days") or 1)
    date_from = range_info["date_from"]
    prev_to = date_from - timedelta(days=1)
    prev_from = prev_to - timedelta(days=days - 1)

    tz = range_info["tz"]
    start_local = datetime(prev_from.year, prev_from.month, prev_from.day, 0, 0, 0, tzinfo=tz)
    end_local = datetime(prev_to.year, prev_to.month, prev_to.day, 23, 59, 59, tzinfo=tz)
    return {
        "tz": tz,
        "date_from": prev_from,
        "date_to": prev_to,
        "start_utc": start_local.astimezone(dt_timezone.utc),
        "end_utc": end_local.astimezone(dt_timezone.utc),
        "days": days,
    }


def _delta(current, previous):
    try:
        cur = float(current)
        prev = float(previous)
    except Exception:
        return {"abs": None, "pct": None}
    abs_delta = cur - prev
    pct = None
    if prev != 0:
        pct = (abs_delta / prev) * 100.0
    return {"abs": abs_delta, "pct": pct}


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def admin_dashboard_overview(request):
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    if not getattr(request, "user", None) or not request.user.is_superuser:
        return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

    range_info = _date_range_from_request(request, default_days=90)
    qs = request.query_params
    compare = _as_bool(qs.get("compare"))

    users_all = User.objects.all()
    total_users = users_all.count()
    premium_active = users_all.filter(plan="Premium", is_active=True).count()

    users_period = users_all.filter(date_joined__gte=range_info["start_utc"], date_joined__lte=range_info["end_utc"])
    signups_total = users_period.count()
    signups_premium = users_period.filter(plan="Premium").count()
    conversion = (signups_premium / signups_total) if signups_total else 0.0

    # Actividad aproximada MVP: last_login en 7 días (si existe)
    last_7d_utc = timezone.now() - timedelta(days=7)
    active_7d = users_all.filter(last_login__isnull=False, last_login__gte=last_7d_utc, is_active=True).count()

    prev = None
    deltas = None
    if compare:
        prev_range = _previous_period(range_info)
        prev_period = users_all.filter(date_joined__gte=prev_range["start_utc"], date_joined__lte=prev_range["end_utc"])
        prev_signups_total = prev_period.count()
        prev_signups_premium = prev_period.filter(plan="Premium").count()
        prev_conversion = (prev_signups_premium / prev_signups_total) if prev_signups_total else 0.0

        prev = {
            "signups_total": prev_signups_total,
            "signups_premium": prev_signups_premium,
            "conversion_premium": prev_conversion,
        }
        deltas = {
            "signups_total": _delta(signups_total, prev_signups_total),
            "signups_premium": _delta(signups_premium, prev_signups_premium),
            "conversion_premium": _delta(conversion, prev_conversion),
        }

    return Response(
        {
            "data": {
                "total_users": total_users,
                "premium_active": premium_active,
                "active_users_7d": active_7d,
                "signups_total": signups_total,
                "signups_premium": signups_premium,
                "conversion_premium": round(conversion, 6),
            },
            "meta": {
                "dateFrom": str(range_info["date_from"]),
                "dateTo": str(range_info["date_to"]),
                "timezone": str(range_info["tz"]),
                "days": range_info["days"],
                "compare": compare,
                "previous": prev,
                "deltas": deltas,
            },
        }
    )


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def admin_dashboard_signups_series(request):
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    if not getattr(request, "user", None) or not request.user.is_superuser:
        return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

    range_info = _date_range_from_request(request, default_days=90)
    tz = range_info["tz"]

    base = User.objects.filter(date_joined__gte=range_info["start_utc"], date_joined__lte=range_info["end_utc"])
    daily = (
        base.annotate(day=TruncDate('date_joined', tzinfo=tz))
        .values('day', 'plan')
        .annotate(count=Count('id'))
        .order_by('day')
    )

    # Construir mapa day -> {total,premium,free}
    series_map = {}
    for row in daily:
        day = row.get('day')
        plan = row.get('plan')
        cnt = int(row.get('count') or 0)
        if not day:
            continue
        key = day.isoformat()
        if key not in series_map:
            series_map[key] = {"date": key, "total": 0, "premium": 0, "free": 0}
        series_map[key]["total"] += cnt
        if plan == "Premium":
            series_map[key]["premium"] += cnt
        else:
            series_map[key]["free"] += cnt

    # Rellenar días faltantes
    out = []
    cur = range_info["date_from"]
    while cur <= range_info["date_to"]:
        k = cur.isoformat()
        out.append(series_map.get(k) or {"date": k, "total": 0, "premium": 0, "free": 0})
        cur += timedelta(days=1)

    return Response({"data": out, "meta": {"dateFrom": str(range_info["date_from"]), "dateTo": str(range_info["date_to"]), "timezone": str(tz), "days": range_info["days"]}})


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def admin_audit_list(request):
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    if not getattr(request, "user", None) or not request.user.is_superuser:
        return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

    qs = request.query_params
    action = (qs.get('action') or '').strip()
    entity_type = (qs.get('entity_type') or '').strip()
    entity_id = (qs.get('entity_id') or '').strip()

    try:
        page = int(qs.get('page') or 1)
    except Exception:
        page = 1
    try:
        page_size = int(qs.get('pageSize') or 50)
    except Exception:
        page_size = 50
    page = max(1, page)
    page_size = max(1, min(200, page_size))

    logs = AuditLog.objects.all().select_related('actor').order_by('-occurred_at')
    if action:
        logs = logs.filter(action=action)
    if entity_type:
        logs = logs.filter(entity_type=entity_type)
    if entity_id:
        logs = logs.filter(entity_id=entity_id)

    total = logs.count()
    start = (page - 1) * page_size
    items = logs[start:start + page_size]
    data = []
    for it in items:
        data.append(
            {
                "occurred_at": _dt_iso(it.occurred_at),
                "action": it.action,
                "actor": {
                    "id": it.actor.id if it.actor else None,
                    "email": it.actor.email if it.actor else None,
                    "username": it.actor.username if it.actor else None,
                },
                "entity_type": it.entity_type,
                "entity_id": it.entity_id,
                "reason": it.reason,
                "ip": it.ip,
            }
        )

    return Response(
        {
            "data": data,
            "meta": {"page": page, "pageSize": page_size, "total": total},
        }
    )


# Ajustar path para importar if_model (que está en la raíz del proyecto backend/if_model?) No, está en feature_engineer_v6.py path
# Structure: backend/api/views.py. if_model is in c:/Users/PC/Desktop/GoToGym_Project/if_model
# But django runs from backend/
# We need to add parent dir to path or move logic. 
# Better: User "feature_engineer_v6.py" wrapper in root? No.
# Let's insert the path dynamically or assume user copied if_model inside backend/api or root.
# Given analysis, if_model is at root. runserver is at backend/. So '../if_model'
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from if_model.service import predict_if_from_scores

def _apply_trial_status(user):
    if user.trial_active and user.trial_ends_at and timezone.now() >= user.trial_ends_at:
        user.trial_active = False
        if user.plan == "Premium":
            user.plan = "Gratis"
        if user.billing_status == "trial":
            user.billing_status = "expired"
        user.save()


def _is_premium_active(user):
    _apply_trial_status(user)
    if user.plan == "Premium":
        return True
    if user.trial_active and user.trial_ends_at and timezone.now() < user.trial_ends_at:
        return True
    return False

def _week_id(dt=None):
    dt = dt or date.today()
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _ensure_if_questions():
    if IFQuestion.objects.exists():
        return
    for q in IF_QUESTIONS:
        IFQuestion.objects.update_or_create(
            key=q["id"],
            defaults={"label": q["label"], "order": q["order"], "active": True},
        )


def _compute_final_score(scores: dict):
    if not scores:
        return 0.0
    if all(v == 0 for v in scores.values()):
        return 0.0
    if all(v == 10 for v in scores.values()):
        return 10.0
    val_ridge, val_stack = predict_if_from_scores(scores)
    return val_stack if val_stack is not None else val_ridge


def _get_client_ip(request):
    def _normalize_ip(value):
        if not value:
            return None
        candidate = str(value).strip()
        if not candidate:
            return None

        if candidate.startswith("[") and "]" in candidate:
            candidate = candidate[1:candidate.index("]")]

        if ":" in candidate and "." in candidate and candidate.count(":") == 1:
            host, _port = candidate.rsplit(":", 1)
            if host:
                candidate = host

        try:
            ipaddress.ip_address(candidate)
            return candidate
        except Exception:
            return None

    forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
    if forwarded:
        normalized = _normalize_ip(forwarded.split(",")[0].strip())
        if normalized:
            return normalized

    return _normalize_ip(request.META.get("REMOTE_ADDR"))


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _password_meets_policy(password: str) -> bool:
    if not password:
        return False
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[0-9]", password):
        return False
    return True


def _frontend_base_url(request) -> str:
    base = (os.getenv("PASSWORD_RESET_FRONTEND_URL", "") or os.getenv("FRONTEND_BASE_URL", "")).strip()
    if base:
        return base.rstrip("/")
    try:
        return request.build_absolute_uri("/").rstrip("/")
    except Exception:
        return ""


def _audit_log(request, action, entity_type="", entity_id="", before=None, after=None, reason=""):
    try:
        actor = getattr(request, "user", None) if request else None
        if actor and not getattr(actor, "is_authenticated", False):
            actor = None
        AuditLog.objects.create(
            actor=actor,
            action=str(action)[:80],
            entity_type=(str(entity_type or "")[:40]),
            entity_id=(str(entity_id or "")[:120]),
            before_json=before,
            after_json=after,
            reason=(str(reason or "")[:2000]),
            ip=_get_client_ip(request) if request else None,
            user_agent=(request.META.get("HTTP_USER_AGENT", "")[:1000] if request else ""),
        )
    except Exception:
        # Nunca debe romper la request por auditoría.
        pass


def _soft_delete_user(user: User):
    """Borrado lógico básico sin cambiar el esquema: desactiva usuario y anonimiza credenciales visibles."""
    if not user:
        return
    stamp = timezone.now().strftime("%Y%m%d%H%M%S")
    original_email = user.email
    original_username = user.username
    user.is_active = False
    try:
        user.set_unusable_password()
    except Exception:
        pass
    # Mantener unicidad: reemplazar email/username
    user.email = f"deleted_{user.id}_{stamp}@example.invalid"
    user.username = f"deleted_{user.id}_{stamp}"
    user.full_name = None
    user.save(update_fields=["is_active", "email", "username", "full_name", "password"])
    return {"email": original_email, "username": original_username}


def _send_password_reset_email(to_email: str, reset_url: str) -> bool:
    subject = "Recupera tu contraseña - GoToGym"
    plain = (
        "Hola,\n\n"
        "Recibimos una solicitud para recuperar tu contraseña.\n"
        "Abre este enlace para crear una nueva contraseña:\n\n"
        f"{reset_url}\n\n"
        "Si tú no solicitaste esto, puedes ignorar este correo.\n"
    )
    html = (
        "<p>Hola,</p>"
        "<p>Recibimos una solicitud para recuperar tu contraseña.</p>"
        "<p>Abre este enlace para crear una nueva contraseña:</p>"
        f"<p><a href=\"{reset_url}\">{reset_url}</a></p>"
        "<p>Si tú no solicitaste esto, puedes ignorar este correo.</p>"
    )

    sender = (getattr(settings, "CONTACT_EMAIL_FROM", "") or getattr(settings, "DEFAULT_FROM_EMAIL", "") or "").strip()
    if not sender:
        sender = "DoNotReply@gotogym.store"

    # ACS si está configurado
    try:
        conn_str = (getattr(settings, "ACS_EMAIL_CONNECTION_STRING", "") or "").strip()
        if conn_str and EmailClient is not None:
            client = EmailClient.from_connection_string(conn_str)
            message = {
                "senderAddress": sender,
                "recipients": {"to": [{"address": to_email}]},
                "content": {"subject": subject, "plainText": plain, "html": html},
            }
            poller = client.begin_send(message)
            poller.result()
            return True
    except Exception as exc:
        # No exponer detalles al usuario; dejar evidencia en Log Stream.
        logger.exception("password_reset: ACS send failed")

    # SMTP / backend Django
    try:
        msg = EmailMultiAlternatives(subject=subject, body=plain, from_email=sender, to=[to_email])
        msg.attach_alternative(html, "text/html")
        msg.send(fail_silently=False)
        return True
    except Exception as exc:
        logger.exception("password_reset: SMTP send failed")
        return False


def _clean_extracted_text(text):
    if not text:
        return ""
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return cleaned.strip()


def _ocr_params():
    """Parámetros de OCR configurables por env.

    - CHAT_OCR_LANG: por ejemplo "eng" o "spa" o "spa+eng" (si están instalados en tesseract).
    - CHAT_OCR_TESSERACT_CONFIG: por ejemplo "--psm 6".
    """

    lang = (os.getenv("CHAT_OCR_LANG", "") or "").strip()
    if not lang:
        lang = (os.getenv("TESSERACT_LANG", "") or "").strip()

    config = (os.getenv("CHAT_OCR_TESSERACT_CONFIG", "") or "").strip() or "--psm 6"
    return {
        "lang": lang or None,
        "config": config,
    }


def _is_attachment_text_placeholder(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return (
        t.startswith("[No se pudo extraer texto del adjunto automáticamente")
        or t.startswith("[No se pudo extraer texto automáticamente")
        or t.startswith("[OCR no disponible")
    )


def _is_image_filename(name: str) -> bool:
    ext = os.path.splitext((name or "").lower())[1]
    return ext in (".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif")


def _azure_openai_vision_config():
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT", "") or "").strip()
    api_key = (os.getenv("AZURE_OPENAI_API_KEY", "") or "").strip()
    deployment = (
        (os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "") or "").strip()
        or (os.getenv("AZURE_OPENAI_DEPLOYMENT", "") or "").strip()
    )
    api_version = (os.getenv("AZURE_OPENAI_API_VERSION", "") or "").strip() or "2024-06-01"
    enabled = _as_bool(os.getenv("CHAT_ATTACHMENT_VISION", "true"))
    if not (enabled and endpoint and api_key and deployment):
        return None
    return {
        "endpoint": endpoint.rstrip("/"),
        "api_key": api_key,
        "deployment": deployment,
        "api_version": api_version,
    }


def _azure_openai_vision_diagnostic() -> str:
    """Diagnóstico legible cuando Vision no está disponible.

    No incluye secretos, solo nombres de variables faltantes.
    """

    enabled = _as_bool(os.getenv("CHAT_ATTACHMENT_VISION", "true"))
    if not enabled:
        return "vision_disabled"

    missing = []
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT", "") or "").strip()
    api_key = (os.getenv("AZURE_OPENAI_API_KEY", "") or "").strip()
    deployment = (
        (os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "") or "").strip()
        or (os.getenv("AZURE_OPENAI_DEPLOYMENT", "") or "").strip()
    )

    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not deployment:
        missing.append("AZURE_OPENAI_VISION_DEPLOYMENT")

    if missing:
        return "vision_not_configured:missing=" + ",".join(missing)
    return "vision_not_configured"


def _describe_image_with_azure_openai(image_url: str, *, image_bytes: bytes | None = None, content_type: str | None = None):
    """Genera una descripción de una imagen usando Azure OpenAI (Vision).

    - Si image_bytes es provisto, se manda como data URL para evitar que el modelo haga fetch.
    - Retorna (descripcion, diagnostic).
    """

    cfg = _azure_openai_vision_config()
    if not cfg:
        return "", _azure_openai_vision_diagnostic()

    try:
        url = (
            f"{cfg['endpoint']}/openai/deployments/{cfg['deployment']}/chat/completions"
            f"?api-version={cfg['api_version']}"
        )

        final_image_url = (image_url or "").strip()
        if image_bytes:
            safe_ct = (content_type or "image/png").split(";")[0].strip() or "image/png"
            b64 = base64.b64encode(image_bytes).decode("ascii")
            final_image_url = f"data:{safe_ct};base64,{b64}"

        if not final_image_url:
            return "", "vision_missing_image_url"

        # Nota: esta salida se usa como "router" de imágenes en `chat_n8n`.
        # Mantener compatibilidad con `is_food/items/portion_estimate/notes`.
        system_msg = (
            "Eres un router multimodal para un coach fitness. Tu trabajo es clasificar "
            "una imagen en una de 4 rutas finales de negocio y extraer señales mínimas. "
            "Responde en español, pero tu salida DEBE ser SOLO JSON válido."
        )
        user_text = (
            "Analiza la imagen y responde SOLO en JSON con estas claves:\n"
            "- route (string): una de 'nutrition', 'training', 'health', 'quantum'\n"
            "- route_confidence (number 0..1)\n"
            "- is_food (boolean)\n"
            "- items (array de strings): SOLO si is_food=true; si no, []\n"
            "- portion_estimate (string)\n"
            "- notes (string): breve, p.ej. 'selfie/rostro', 'cuerpo completo', 'primer plano piel/músculo', 'etiqueta nutricional', 'contexto gimnasio'\n"
            "- has_person (boolean)\n"
            "- has_nutrition_label (boolean)\n"
            "- is_closeup_skin_or_muscle (boolean)\n"
            "Reglas: si ves comida o etiqueta nutricional => route='nutrition'. Si es una persona entrenando o en contexto gimnasio => route='training'. "
            "Si es primer plano de piel/músculo/rostro para salud/belleza => route='health'. Si no encaja => route='quantum'."
        )

        payload = {
            "messages": [
                {"role": "system", "content": system_msg},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": final_image_url}},
                    ],
                },
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": cfg["api_key"],
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=25)
        if resp.status_code != 200:
            return "", f"vision_http_{resp.status_code}: {resp.text[:200]}"

        data = resp.json() if resp.content else {}
        content = (
            (data.get("choices") or [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return (content or "").strip(), ""
    except Exception as ex:
        return "", f"vision_failed:{ex}"


def _extract_text_from_file_bytes(original_name: str, file_bytes: bytes):
    """Extrae texto de PDFs/imágenes desde bytes.

    Retorna (extracted_text, diagnostic).
    diagnostic ayuda a entender por qué podría venir vacío.
    """

    original_name = (original_name or "").strip() or "adjunto"
    extension = os.path.splitext(original_name)[1].lower()
    extracted_text = ""
    diagnostic = ""
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension or "") as tmp_file:
            tmp_file.write(file_bytes or b"")
            temp_path = tmp_file.name

        is_pdf = extension == ".pdf"
        # OCR: por defecto lo habilitamos también para PDF, porque muchos PDFs son escaneados
        # y `pypdf` no extrae texto.
        enable_ocr = _as_bool(os.getenv("CHAT_ATTACHMENT_OCR", "true"))

        if is_pdf:
            try:
                from pypdf import PdfReader

                reader = PdfReader(temp_path)
                for page in reader.pages:
                    extracted_text += (page.extract_text() or "") + "\n"
            except Exception as ex:
                diagnostic = f"pypdf_failed:{ex}"
                extracted_text = ""

            # Si el PDF viene sin capa de texto y OCR está apagado, dejar diagnóstico explícito.
            if (not enable_ocr) and (not extracted_text.strip()) and (not diagnostic):
                diagnostic = "pdf_ocr_disabled"

            if enable_ocr and not extracted_text.strip():
                if pytesseract and convert_from_path:
                    try:
                        pages = convert_from_path(temp_path, dpi=200, first_page=1, last_page=10)
                        ocr_opts = _ocr_params()
                        ocr_chunks = []
                        for img in pages:
                            try:
                                ocr_chunks.append(
                                    pytesseract.image_to_string(
                                        img,
                                        lang=ocr_opts["lang"],
                                        config=ocr_opts["config"],
                                    )
                                )
                            except Exception:
                                pass
                        extracted_text = "\n".join([c for c in ocr_chunks if c.strip()])
                    except Exception as ex:
                        diagnostic = f"pdf_ocr_failed:{ex}"
                else:
                    diagnostic = "pdf_ocr_unavailable"
        else:
            if enable_ocr and pytesseract and Image is not None:
                try:
                    # Preprocesado básico para OCR: corrección de rotación EXIF y mejora de contraste.
                    from PIL import ImageEnhance, ImageOps

                    img = Image.open(temp_path)
                    img = ImageOps.exif_transpose(img)

                    if img.mode not in ("RGB", "L"):
                        img = img.convert("RGB")

                    gray = img.convert("L")
                    gray = ImageEnhance.Contrast(gray).enhance(1.8)

                    ocr_opts = _ocr_params()
                    try:
                        extracted_text = pytesseract.image_to_string(
                            gray,
                            lang=ocr_opts["lang"],
                            config=ocr_opts["config"],
                        )
                    except Exception:
                        # Si el lang configurado no existe en el runtime, reintenta sin lang.
                        extracted_text = pytesseract.image_to_string(
                            gray,
                            config=ocr_opts["config"],
                        )
                except Exception as ex:
                    diagnostic = f"img_ocr_failed:{ex}"
            elif not enable_ocr:
                diagnostic = "img_ocr_disabled"
            else:
                diagnostic = "img_ocr_unavailable"
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    extracted_text = _clean_extracted_text(extracted_text or "")
    max_chars = int(
        os.getenv(
            "CHAT_ATTACHMENT_TEXT_MAX_CHARS",
            os.getenv("N8N_DOC_TEXT_MAX_CHARS", "5000"),
        )
        or 5000
    )
    if max_chars and len(extracted_text) > max_chars:
        extracted_text = extracted_text[:max_chars]

    if not extracted_text.strip() and not diagnostic:
        diagnostic = "empty"

    return extracted_text, diagnostic


def _get_media_public_base():
    base = (
        os.getenv("MEDIA_PUBLIC_BASE", "").strip()
        or getattr(settings, "MEDIA_PUBLIC_BASE", "").strip()
        or "https://gotogymweb3755.blob.core.windows.net/media/"
    )
    if not base.endswith("/"):
        base = f"{base}/"
    return base


def _canonical_profile_picture_url(request, user):
    if not getattr(user, "profile_picture", None):
        return ""

    try:
        candidate = (user.profile_picture.url or "").strip()
    except Exception:
        candidate = ""

    if not candidate:
        candidate = (getattr(user.profile_picture, "name", "") or "").strip()

    if not candidate:
        return ""

    if candidate.startswith("blob:") or candidate.startswith("data:"):
        return candidate

    if candidate.lower().startswith("http://") or candidate.lower().startswith("https://"):
        return quote(candidate, safe=":/?&=#%")

    cleaned = candidate.lstrip("/")
    if cleaned.lower().startswith("media/"):
        cleaned = cleaned[6:]

    if cleaned.startswith("/"):
        absolute = request.build_absolute_uri(cleaned) if request else cleaned
        return quote(absolute, safe=":/?&=#%")

    absolute = f"{_get_media_public_base()}{cleaned}"
    return quote(absolute, safe=":/?&=#%")


def _profile_picture_db_value(user):
    if not getattr(user, "profile_picture", None):
        return ""
    return (getattr(user.profile_picture, "name", "") or "").strip()


def _storage_target_info():
    container = (
        os.getenv("AZURE_STORAGE_CONTAINER", "").strip()
        or getattr(settings, "AZURE_STORAGE_CONTAINER", "").strip()
        or "media"
    )
    backend_name = "azure" if os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip() else "local"
    return {
        "storage_backend": backend_name,
        "storage_container": container,
    }


def _doc_container_map():
    return {
        "nutrition_plan": os.getenv("AZURE_CONTAINER_NUTRITION", "nutricion").strip() or "nutricion",
        "training_plan": os.getenv("AZURE_CONTAINER_TRAINING", "entrenamiento").strip() or "entrenamiento",
        "medical_history": os.getenv("AZURE_CONTAINER_MEDICAL", "historiaclinica").strip() or "historiaclinica",
    }


def _chat_attachment_container():
    return os.getenv("AZURE_CONTAINER_ATTACHMENTS", "attachments").strip() or "attachments"


def _is_azure_blob_enabled():
    return (
        bool(getattr(settings, "AZURE_STORAGE_ACCOUNT_NAME", "").strip())
        and bool(getattr(settings, "AZURE_STORAGE_ACCOUNT_KEY", "").strip())
        and BlobServiceClient is not None
    )


def _build_blob_url(container, blob_name):
    account_name = getattr(settings, "AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    encoded_parts = [quote(p, safe="") for p in blob_name.split("/")]
    encoded_blob_name = "/".join(encoded_parts)
    return f"https://{account_name}.blob.core.windows.net/{container}/{encoded_blob_name}"


def _extract_blob_ref_from_url(file_url):
    if not file_url:
        return None, None
    try:
        parsed = urlparse(file_url)
        path = (parsed.path or "").strip("/")
        if not path:
            return None, None
        parts = path.split("/", 1)
        if len(parts) < 2:
            return None, None
        return parts[0], parts[1]
    except Exception:
        return None, None


def _resolve_blob_name(container_name, blob_name):
    if not container_name or not blob_name:
        return None

    candidates = [blob_name]
    decoded = unquote(blob_name)
    if decoded and decoded != blob_name:
        candidates.insert(0, decoded)

    account_name = getattr(settings, "AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    account_key = getattr(settings, "AZURE_STORAGE_ACCOUNT_KEY", "").strip()
    if not account_name or not account_key or BlobServiceClient is None:
        return candidates[0]

    try:
        account_url = f"https://{account_name}.blob.core.windows.net"
        blob_service = BlobServiceClient(account_url=account_url, credential=account_key)
        for candidate in candidates:
            try:
                blob_client = blob_service.get_blob_client(container=container_name, blob=candidate)
                blob_client.get_blob_properties()
                return candidate
            except Exception:
                continue
    except Exception:
        return candidates[0]

    return candidates[0]


def _build_signed_blob_url(file_url):
    if not file_url or not _is_azure_blob_enabled() or generate_blob_sas is None or BlobSasPermissions is None:
        return file_url

    account_name = getattr(settings, "AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    account_key = getattr(settings, "AZURE_STORAGE_ACCOUNT_KEY", "").strip()
    container_name, blob_name_raw = _extract_blob_ref_from_url(file_url)
    if not container_name or not blob_name_raw:
        return file_url

    blob_name = _resolve_blob_name(container_name, blob_name_raw)
    if not blob_name:
        return file_url

    try:
        expiry = timezone.now() + timedelta(seconds=getattr(settings, "AZURE_SAS_EXPIRATION", 3600))
        sas_token = generate_blob_sas(
            account_name=account_name,
            account_key=account_key,
            container_name=container_name,
            blob_name=blob_name,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        if not sas_token:
            return file_url
        separator = "&" if "?" in file_url else "?"
        return f"{file_url}{separator}{sas_token}"
    except Exception:
        return file_url


def _upload_file_to_blob(container_name, blob_name, file_bytes, content_type):
    account_name = getattr(settings, "AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    account_key = getattr(settings, "AZURE_STORAGE_ACCOUNT_KEY", "").strip()
    account_url = f"https://{account_name}.blob.core.windows.net"
    blob_service = BlobServiceClient(account_url=account_url, credential=account_key)
    blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)

    kwargs = {"overwrite": True}
    if ContentSettings is not None:
        kwargs["content_settings"] = ContentSettings(content_type=content_type or "application/octet-stream")
    blob_client.upload_blob(file_bytes, **kwargs)
    return _build_blob_url(container_name, blob_name)


def _delete_blob_if_exists(file_url):
    if not _is_azure_blob_enabled() or not file_url:
        return
    container_name, blob_name_raw = _extract_blob_ref_from_url(file_url)
    if not container_name or not blob_name_raw:
        return
    try:
        account_name = getattr(settings, "AZURE_STORAGE_ACCOUNT_NAME", "").strip()
        account_key = getattr(settings, "AZURE_STORAGE_ACCOUNT_KEY", "").strip()
        account_url = f"https://{account_name}.blob.core.windows.net"
        blob_service = BlobServiceClient(account_url=account_url, credential=account_key)
        candidates = [blob_name_raw]
        decoded = unquote(blob_name_raw)
        if decoded and decoded != blob_name_raw:
            candidates.insert(0, decoded)
        for candidate in candidates:
            try:
                blob_client = blob_service.get_blob_client(container=container_name, blob=candidate)
                blob_client.delete_blob(delete_snapshots="include")
                return
            except Exception:
                continue
    except Exception:
        pass


def _save_local_document(doc_type, username, safe_name, file_bytes):
    folder_map = {
        "nutrition_plan": "nutrition_plans",
        "training_plan": "training_plans",
        "medical_history": "medical_records",
    }
    folder_name = folder_map.get(doc_type, "medical_records")
    user_folder = os.path.join(settings.MEDIA_ROOT, folder_name, username)
    os.makedirs(user_folder, exist_ok=True)
    file_path = os.path.join(user_folder, safe_name)
    with open(file_path, "wb") as destination:
        destination.write(file_bytes)

    media_url = str(getattr(settings, "MEDIA_URL", "/media/") or "/media/")
    if not media_url.endswith("/"):
        media_url += "/"
    relative_url = f"{media_url}{folder_name}/{username}/{safe_name}".replace("//", "/")
    local_url = relative_url if relative_url.startswith("http") else f"https://api.gotogym.store{relative_url}"
    return file_path, local_url


def _resolve_request_user(request):
    token_user_id = None
    try:
        auth_token = getattr(request, "auth", None)
        if auth_token is not None:
            if isinstance(auth_token, dict):
                token_user_id = auth_token.get("user_id")
            else:
                token_user_id = auth_token.get("user_id")
    except Exception:
        token_user_id = None

    try:
        user = getattr(request, "user", None)
        if user is not None and getattr(user, "is_authenticated", False):
            _ = user.pk
            return user
    except Exception:
        pass

    if token_user_id is not None:
        try:
            user = User.objects.filter(id=int(token_user_id)).first()
            if user:
                return user
        except Exception:
            pass

    username = (request.data.get("username") or "").strip()
    if username:
        user = User.objects.filter(username=username).first()
        if not user:
            return None
        if token_user_id is not None:
            try:
                if int(token_user_id) != int(user.id):
                    return None
            except Exception:
                return None
        return user

    return None


def _require_authenticated_user(request, requested_username=None):
    user = _resolve_request_user(request)
    if not user:
        return None, Response({'error': 'Authentication required'}, status=status.HTTP_401_UNAUTHORIZED)
    if requested_username and user.username != requested_username:
        return None, Response({'error': 'Forbidden'}, status=status.HTTP_403_FORBIDDEN)
    return user, None


def _safe_session_id(user):
    raw = f"{user.id}:{user.username}:{settings.SECRET_KEY}"
    return f"u_{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _range_bucket(value, step, lower=None, upper=None):
    if value is None:
        return None
    try:
        num = float(value)
    except Exception:
        return None
    if lower is not None:
        num = max(num, lower)
    if upper is not None:
        num = min(num, upper)
    low = int((num // step) * step)
    high = low + step
    return f"{low}-{high}"

@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def get_user_profile(request):
    """
    Endpoint dedicado para el Agente AI.
    Devuelve datos clave del usuario (peso, altura, edad, felicidad) para contexto.
    """
    username = request.query_params.get('username')
    if not username:
        return Response({'error': 'Username required'}, status=status.HTTP_400_BAD_REQUEST)

    user, auth_error = _require_authenticated_user(request, username)
    if auth_error:
        return auth_error

    try:
        profile_picture = _profile_picture_db_value(user)
        profile_picture_url = _canonical_profile_picture_url(request, user)
        gamification = build_gamification_status(user)
        return Response({
            'username': user.username,
            'age': user.age,
            'weight': user.weight,
            'height': user.height,
            'profession': getattr(user, 'profession', None),
            'full_name': getattr(user, 'full_name', None),
            'favorite_exercise_time': getattr(user, 'favorite_exercise_time', None),
            'favorite_sport': getattr(user, 'favorite_sport', None),
            'plan': user.plan,
            'happiness_index': user.happiness_index,
            'current_streak': user.current_streak,
            'badges': user.badges,
            'gamification': gamification,
            'happiness_scores': user.scores or {},
            'has_happiness_scores': bool(user.scores),
            'profile_picture': profile_picture,
            'profile_picture_url': profile_picture_url,
        })
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def gamification_status(request):
    user, auth_error = _require_authenticated_user(request)
    if auth_error:
        return auth_error
    try:
        return Response(build_gamification_status(user))
    except Exception as exc:
        return Response({"ok": False, "error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def coach_context(request):
    username = request.query_params.get('username')
    include_text = _as_bool(request.query_params.get('include_text'))
    if not username:
        return Response({'error': 'Username required'}, status=status.HTTP_400_BAD_REQUEST)

    user, auth_error = _require_authenticated_user(request, username)
    if auth_error:
        return auth_error

    def _dt_iso(value):
        return value.isoformat() if value else None

    documents_qs = UserDocument.objects.filter(user=user).order_by('-updated_at')
    latest_by_type = {}
    for doc in documents_qs:
        if doc.doc_type not in latest_by_type:
            summary = {
                "doc_type": doc.doc_type,
                "file_name": doc.file_name,
                "updated_at": _dt_iso(doc.updated_at),
            }
            if include_text:
                summary["extracted_text"] = doc.extracted_text
            latest_by_type[doc.doc_type] = summary
    documents_summary = list(latest_by_type.values())
    documents_types = list(latest_by_type.keys())

    device_qs = DeviceConnection.objects.filter(user=user).order_by('-updated_at')
    devices_payload = [
        {
            "provider": d.provider,
            "status": d.status,
            "last_sync_at": _dt_iso(d.last_sync_at),
            "updated_at": _dt_iso(d.updated_at),
        }
        for d in device_qs
    ]
    connected_providers = [
        d["provider"] for d in devices_payload if d["status"] == "connected"
    ]

    fitness_by_provider = {}
    recent_syncs = DevicesFitnessSync.objects.filter(user=user).order_by('-created_at')[:50]
    for sync in recent_syncs:
        if sync.provider not in fitness_by_provider:
            fitness_by_provider[sync.provider] = {
                "provider": sync.provider,
                "start_time": _dt_iso(sync.start_time),
                "end_time": _dt_iso(sync.end_time),
                "metrics": sync.metrics,
                "created_at": _dt_iso(sync.created_at),
            }

    latest_record = HappinessRecord.objects.filter(user=user).order_by('-date').first()
    week_id = _week_id()
    answers_qs = (
        IFAnswer.objects.filter(user=user, week_id=week_id)
        .select_related('question')
        .order_by('answered_at')
    )
    answers_payload = [
        {
            "question_id": a.question.key,
            "question_label": a.question.label,
            "value": a.value,
            "slot": a.slot,
            "answered_at": _dt_iso(a.answered_at),
            "answered_date": _dt_iso(a.answered_date),
            "source": a.source,
        }
        for a in answers_qs
    ]

    context_payload = {
        "profile": {
            "username": user.username,
            "plan": user.plan,
            "full_name": getattr(user, "full_name", None),
            "email": user.email,
            "timezone": getattr(user, "timezone", "") or "",
            "sex": getattr(user, "sex", None),
            "age": user.age,
            "weight": user.weight,
            "height": user.height,
            "profession": getattr(user, "profession", None),
            "favorite_exercise_time": getattr(user, "favorite_exercise_time", None),
            "favorite_sport": getattr(user, "favorite_sport", None),
            "goal_type": getattr(user, "goal_type", None),
            "activity_level": getattr(user, "activity_level", None),
            "daily_target_kcal_override": getattr(user, "daily_target_kcal_override", None),
            "age_range": _range_bucket(user.age, 5),
            "weight_range": _range_bucket(user.weight, 5, lower=30, upper=200),
            "height_range": _range_bucket(user.height, 5, lower=120, upper=230),
            "happiness_index": user.happiness_index,
            "scores": user.scores or {},
            "current_streak": user.current_streak,
            "badges": user.badges,
            "coach_state": getattr(user, "coach_state", {}) or {},
            "coach_state_updated_at": _dt_iso(getattr(user, "coach_state_updated_at", None)),
            "coach_weekly_state": getattr(user, "coach_weekly_state", {}) or {},
            "coach_weekly_updated_at": _dt_iso(getattr(user, "coach_weekly_updated_at", None)),
        },
        "documents": {
            "summary": documents_summary,
            "types": documents_types,
            "count": len(documents_summary),
        },
        "devices": {
            "connected_providers": connected_providers,
            "devices": devices_payload,
            "fitness": fitness_by_provider,
        },
        "if_snapshot": {
            "week_id": week_id,
            "scores": user.scores or {},
            "latest_record": {
                "value": latest_record.value if latest_record else None,
                "scores": latest_record.scores if latest_record else {},
                "date": _dt_iso(latest_record.date) if latest_record else None,
            },
            "answers": answers_payload,
        },
    }

    return Response(context_payload)

@api_view(['POST', 'OPTIONS'])
def register(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
        
    try:
        data = request.data
        print("Datos recibidos del frontend:", data)

        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        plan = data.get('plan', 'Gratis')
        full_name = data.get('fullName') or data.get('full_name')
        terms_accepted = _as_bool(data.get('termsAccepted'))
        terms_version = (data.get('termsVersion') or '').strip()
        terms_source = (data.get('termsSource') or 'web').strip() or 'web'

        if not username or not email or not password:
            return Response({'error': 'Faltan campos obligatorios'}, status=status.HTTP_400_BAD_REQUEST)
        if not terms_accepted:
            return Response({'error': 'Debes aceptar los términos y condiciones'}, status=status.HTTP_400_BAD_REQUEST)
        if not terms_version:
            terms_version = os.getenv("TERMS_VERSION", "2025-04-07")

        if User.objects.filter(username=username).exists():
            return Response({'error': 'Usuario ya existe'}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(email=email).exists():
            return Response({'error': 'El email ya esta registrado'}, status=status.HTTP_400_BAD_REQUEST)

        if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'[0-9]', password):
            return Response({'error': 'La contraseña no cumple los requisitos'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                plan=plan
            )
        except IntegrityError:
            return Response({'error': 'Usuario o email ya existe'}, status=status.HTTP_400_BAD_REQUEST)
        if plan == "Premium":
            now = timezone.now()
            user.trial_active = True
            user.trial_started_at = now
            user.trial_ends_at = now + timedelta(days=14)
            user.billing_status = "trial"
            user.plan = "Premium"
        if full_name:
            user.full_name = full_name

        user.terms_accepted_at = timezone.now()
        user.terms_accepted_version = terms_version
        user.terms_accepted_ip = _get_client_ip(request)
        user.terms_accepted_user_agent = request.META.get("HTTP_USER_AGENT", "")[:1000]
        user.terms_accepted_source = terms_source[:30]
        user.save()

        try:
            TermsAcceptance.objects.create(
                user=user,
                version=terms_version,
                ip_address=user.terms_accepted_ip,
                user_agent=user.terms_accepted_user_agent,
                source=user.terms_accepted_source,
            )
        except Exception as e:
            print("Error guardando TermsAcceptance:", str(e))
        
        try:
            serializer = UserSerializer(user, context={"request": request})
            user_payload = serializer.data
        except Exception as e:
            print("Error serializando usuario en registro:", str(e))
            traceback.print_exc()
            user_payload = {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "plan": user.plan,
                "trial_active": user.trial_active,
                "trial_ends_at": user.trial_ends_at,
                "billing_status": user.billing_status,
            }

        response = Response({
            'success': True,
            'message': 'Usuario registrado exitosamente',
            'user': user_payload
        }, status=status.HTTP_201_CREATED)
        
        response['Access-Control-Allow-Origin'] = '*'
        return response
        
    except Exception as e:
        print("Error en registro:", str(e))
        traceback.print_exc()
        return Response({'error': 'Error interno del servidor'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST', 'OPTIONS'])
def login(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
        
    try:
        data = request.data

        identifier = (data.get('username') or '').strip()
        password = data.get('password') or ''

        if not identifier or not password:
            return Response({
                'success': False,
                'error': 'Usuario y contraseña son requeridos'
            }, status=status.HTTP_400_BAD_REQUEST)

        # Acepta username o email (en el frontend, username técnico = email)
        user = User.objects.filter(
            Q(username__iexact=identifier) | Q(email__iexact=identifier)
        ).first()
        if not user:
            return Response({
                'success': False,
                'error': 'Usuario no encontrado'
            }, status=status.HTTP_400_BAD_REQUEST)

        if hasattr(user, "is_active") and not user.is_active:
            return Response({
                'success': False,
                'error': 'Usuario inactivo'
            }, status=status.HTTP_403_FORBIDDEN)

        if not user.check_password(password):
            return Response({
                'success': False,
                'error': 'Contrasena incorrecta'
            }, status=status.HTTP_400_BAD_REQUEST)

        _apply_trial_status(user)
        serializer = UserSerializer(user, context={"request": request})
        refresh = RefreshToken.for_user(user)
        access = refresh.access_token
        response = Response({
            'success': True,
            'message': 'Inicio de sesion correcto',
            'user': serializer.data,
            'access': str(access),
            'refresh': str(refresh)
        })
        response['Access-Control-Allow-Origin'] = '*'
        return response
        
            
    except Exception as e:
        print('Error en login:', str(e))
        return Response({'error': 'Error interno del servidor'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST", "OPTIONS"])
@permission_classes([AllowAny])
def password_reset_request(request):
    if request.method == "OPTIONS":
        return Response(status=status.HTTP_200_OK)

    payload = request.data if isinstance(request.data, dict) else {}
    email = (payload.get("email") or "").strip().lower()

    generic = {
        "ok": True,
        "message": "Si el correo existe en GoToGym, enviaremos instrucciones para recuperar tu contraseña.",
    }

    if not email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return Response(generic, status=status.HTTP_200_OK)

    user = User.objects.filter(email__iexact=email).first()
    if not user:
        return Response(generic, status=status.HTTP_200_OK)

    ttl = int(os.getenv("PASSWORD_RESET_TTL_SECONDS", str(60 * 60)) or (60 * 60))
    signer = signing.TimestampSigner(salt="pwd-reset")
    token_data = json.dumps({"uid": user.id, "nonce": secrets.token_hex(8)}, separators=(",", ":"))
    token = signer.sign(token_data)

    base = _frontend_base_url(request)
    if not base:
        return Response(generic, status=status.HTTP_200_OK)

    reset_url = f"{base}/pages/auth/indexInicioDeSesion.html?reset={quote(token)}"
    _send_password_reset_email(email, reset_url)
    return Response({**generic, "ttl_seconds": ttl}, status=status.HTTP_200_OK)


@api_view(["POST", "OPTIONS"])
@permission_classes([AllowAny])
def password_reset_confirm(request):
    if request.method == "OPTIONS":
        return Response(status=status.HTTP_200_OK)

    payload = request.data if isinstance(request.data, dict) else {}
    token = (payload.get("token") or "").strip()
    new_password = payload.get("password") or ""

    if not token or not _password_meets_policy(new_password):
        return Response({"ok": False, "error": "invalid_request"}, status=400)

    ttl = int(os.getenv("PASSWORD_RESET_TTL_SECONDS", str(60 * 60)) or (60 * 60))
    signer = signing.TimestampSigner(salt="pwd-reset")

    try:
        raw = signer.unsign(token, max_age=ttl)
        data = json.loads(raw)
        uid = int(data.get("uid"))
    except signing.SignatureExpired:
        return Response({"ok": False, "error": "token_expired"}, status=400)
    except Exception:
        return Response({"ok": False, "error": "token_invalid"}, status=400)

    user = User.objects.filter(id=uid).first()
    if not user:
        return Response({"ok": False, "error": "token_invalid"}, status=400)

    user.set_password(new_password)
    user.save()
    return Response({"ok": True}, status=200)

@api_view(['POST', 'OPTIONS'])
def contact_message(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    data = request.data or {}
    name = (data.get('name') or '').strip()
    email = (data.get('email') or '').strip()
    subject = (data.get('subject') or '').strip()
    message = (data.get('message') or '').strip()
    honeypot = (data.get('website') or '').strip()

    if honeypot:
        return Response({'success': True, 'message': 'Mensaje recibido'}, status=status.HTTP_200_OK)

    if not name or not email or not subject or not message:
        return Response({'error': 'Todos los campos son obligatorios'}, status=status.HTTP_400_BAD_REQUEST)

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return Response({'error': 'Correo inválido'}, status=status.HTTP_400_BAD_REQUEST)

    provider = (settings.CONTACT_EMAIL_PROVIDER or "acs").lower()

    def _store_message():
        try:
            ContactMessage.objects.create(
                name=name,
                email=email,
                subject=subject,
                message=message,
                status="received",
            )
        except Exception as e:
            print("Error guardando mensaje de contacto:", str(e))

    sender = settings.CONTACT_EMAIL_FROM
    recipient = settings.CONTACT_EMAIL_TO

    subject_line = f"{settings.CONTACT_EMAIL_SUBJECT_PREFIX} {subject}".strip()
    plain_text = (
        "Nuevo mensaje de contacto\n"
        f"Nombre: {name}\n"
        f"Email: {email}\n"
        f"Asunto: {subject}\n\n"
        f"Mensaje:\n{message}\n"
    )
    html_body = f"""
        <h2>Nuevo mensaje de contacto</h2>
        <p><strong>Nombre:</strong> {name}</p>
        <p><strong>Email:</strong> {email}</p>
        <p><strong>Asunto:</strong> {subject}</p>
        <p><strong>Mensaje:</strong></p>
        <pre style="white-space: pre-wrap;">{message}</pre>
    """

    confirmation_subject = "Recibimos tu mensaje - GoToGym"
    confirmation_plain = (
        f"Hola {name},\n\n"
        "Gracias por escribirnos. Ya recibimos tu mensaje y nuestro equipo lo revisara pronto.\n\n"
        "Si necesitas agregar mas informacion, puedes responder directamente a este correo.\n\n"
        "- Equipo GoToGym\n"
    )
    confirmation_html = f"""
        <p>Hola {name},</p>
        <p>Gracias por escribirnos. Ya recibimos tu mensaje y nuestro equipo lo revisara pronto.</p>
        <p>Si necesitas agregar mas informacion, puedes responder directamente a este correo.</p>
        <p>- Equipo GoToGym</p>
    """

    def _send_support_via_smtp():
        if not settings.EMAIL_HOST or not settings.EMAIL_HOST_USER or not settings.EMAIL_HOST_PASSWORD:
            print("SMTP no configurado")
            return
        msg = EmailMessage(
            subject=subject_line,
            body=plain_text,
            from_email=settings.CONTACT_EMAIL_FROM,
            to=[recipient],
            reply_to=[email],
        )
        msg.content_subtype = "plain"
        msg.send(fail_silently=False)

    def _send_confirmation_via_smtp():
        msg = EmailMessage(
            subject=confirmation_subject,
            body=confirmation_plain,
            from_email=settings.CONTACT_EMAIL_FROM,
            to=[email],
        )
        msg.content_subtype = "plain"
        msg.send(fail_silently=False)

    def _build_acs_client():
        conn = settings.ACS_EMAIL_CONNECTION_STRING or ""
        endpoint = None
        access_key = None
        for part in conn.split(";"):
            if not part:
                continue
            key, _, val = part.partition("=")
            if key.lower() == "endpoint":
                endpoint = val.strip()
            elif key.lower() == "accesskey":
                access_key = val.strip()

        if endpoint and access_key and AzureKeyCredential is not None:
            if not endpoint.endswith("/"):
                endpoint = endpoint + "/"
            return EmailClient(endpoint, AzureKeyCredential(access_key))
        return EmailClient.from_connection_string(conn)

    def _send_via_acs():
        if not settings.ACS_EMAIL_CONNECTION_STRING or EmailClient is None:
            print("Servicio de correo no configurado")
            return
        try:
            client = _build_acs_client()
        except Exception as build_err:
            print("Error creando cliente ACS:", str(build_err))
            return

        email_message = {
            "senderAddress": sender,
            "recipients": {
                "to": [{"address": recipient}]
            },
            "content": {
                "subject": subject_line,
                "plainText": plain_text,
                "html": html_body,
            },
            "replyTo": [{"address": email, "displayName": name}],
        }
        try:
            poller = client.begin_send(email_message)
            poller.result()
        except HttpResponseError as send_err:
            print("Error ACS (soporte):", str(send_err))
            return
        except Exception as send_err:
            print("Error enviando correo de soporte:", str(send_err))
            return

        try:
            confirmation_message = {
                "senderAddress": sender,
                "recipients": {
                    "to": [{"address": email}]
                },
                "content": {
                    "subject": confirmation_subject,
                    "plainText": confirmation_plain,
                    "html": confirmation_html,
                },
            }
            confirm_poller = client.begin_send(confirmation_message)
            confirm_poller.result()
        except HttpResponseError as confirm_err:
            print("Error enviando confirmacion:", str(confirm_err))
        except Exception as confirm_err:
            print("Error enviando confirmacion:", str(confirm_err))

    def _send_email_async():
        try:
            if provider == "smtp":
                _send_support_via_smtp()
                _send_confirmation_via_smtp()
            else:
                _send_via_acs()
        except Exception as e:
            print("Error enviando correo:", str(e))

    _store_message()
    threading.Thread(target=_send_email_async, daemon=True).start()
    return Response({'success': True, 'message': 'Mensaje recibido'}, status=status.HTTP_200_OK)


def _send_fcm(tokens, title, body, data=None):
    if not settings.FCM_SERVER_KEY:
        return False, "FCM_SERVER_KEY no configurado"
    if not tokens:
        return False, "No hay tokens"

    url = "https://fcm.googleapis.com/fcm/send"
    headers = {
        "Authorization": f"key={settings.FCM_SERVER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "registration_ids": tokens[:1000],
        "notification": {
            "title": title,
            "body": body,
        },
        "data": data or {},
    }

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=15)
        if not res.ok:
            return False, f"FCM error: {res.status_code}"
        result = res.json()
        return True, result
    except Exception as e:
        return False, str(e)


def _chunk_list(values, size):
    size = int(size) if size else 1000
    if size <= 0:
        size = 1000
    for i in range(0, len(values), size):
        yield values[i:i + size]


def _send_fcm_bulk(tokens, title, body, data=None):
    tokens = list(tokens or [])
    if not tokens:
        return False, "No hay tokens"

    total_sent = 0
    chunk_results = []
    any_ok = False
    for chunk in _chunk_list(tokens, 1000):
        ok, info = _send_fcm(chunk, title, body, data=data)
        chunk_results.append({"ok": ok, "info": info, "count": len(chunk)})
        if ok:
            any_ok = True
            total_sent += len(chunk)

    if not any_ok:
        return False, {"sent": 0, "chunks": chunk_results}
    return True, {"sent": total_sent, "chunks": chunk_results}


def _get_vapid_keys():
    public_key = os.getenv("VAPID_PUBLIC_KEY", "").strip()
    private_key = os.getenv("VAPID_PRIVATE_KEY", "").strip()
    email = os.getenv("VAPID_EMAIL", "").strip() or "mailto:support@gotogym.store"
    return public_key, private_key, email


def _send_web_push(subscriptions, title, body, data=None):
    public_key, private_key, email = _get_vapid_keys()
    if not public_key or not private_key:
        return False, "VAPID keys no configuradas"
    if not webpush:
        return False, "pywebpush no disponible"
    if not subscriptions:
        return False, "No hay suscripciones"

    payload = {
        "title": title,
        "body": body,
        "data": data or {},
    }

    sent = 0
    errors = []
    for sub in subscriptions:
        try:
            webpush(
                subscription_info={
                    "endpoint": sub.endpoint,
                    "keys": {
                        "p256dh": sub.p256dh,
                        "auth": sub.auth,
                    },
                },
                data=json.dumps(payload),
                vapid_private_key=private_key,
                vapid_claims={"sub": email},
            )
            sent += 1
        except WebPushException as exc:
            errors.append(str(exc))
    if sent == 0:
        return False, errors or "No se pudo enviar"
    return True, {"sent": sent, "errors": errors}


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def push_admin_broadcast(request):
    """Envía una notificación a TODOS los dispositivos registrados.

    - Web Push: WebPushSubscription
    - Mobile (Capacitor/FCM): PushToken

    Requiere: usuario superuser.
    """
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    if not getattr(request, 'user', None) or not request.user.is_authenticated:
        return Response({'error': 'No autenticado'}, status=status.HTTP_401_UNAUTHORIZED)

    if not getattr(request.user, 'is_superuser', False):
        return Response({'error': 'Solo administradores'}, status=status.HTTP_403_FORBIDDEN)

    data = request.data or {}
    title = (data.get('title') or 'GoToGym').strip()
    body = (data.get('body') or 'Tienes una notificación').strip()
    payload_data = data.get('data') or {}
    reason = (data.get('reason') or '').strip()
    if not reason:
        return Response({'error': 'reason requerido'}, status=status.HTTP_400_BAD_REQUEST)

    # Protección simple para evitar payloads gigantes.
    try:
        if json.dumps(payload_data).__len__() > 4000:
            return Response({'error': 'data demasiado grande'}, status=status.HTTP_400_BAD_REQUEST)
    except Exception:
        payload_data = {}

    tokens = list(
        PushToken.objects.filter(active=True).values_list('token', flat=True)
    )
    subs = list(
        WebPushSubscription.objects.filter(active=True)
    )

    ok_fcm, info_fcm = _send_fcm_bulk(tokens, title, body, data=payload_data)
    ok_web, info_web = _send_web_push(subs, title, body, data=payload_data)

    _audit_log(
        request,
        action="push.broadcast",
        entity_type="push",
        entity_id="all",
        before=None,
        after={
            "title": title,
            "body": body,
            "data": payload_data,
            "tokens_total": len(tokens),
            "subs_total": len(subs),
            "fcm": info_fcm,
            "web": info_web,
        },
        reason=reason,
    )

    if not ok_fcm and not ok_web:
        return Response(
            {
                'success': False,
                'error': str(info_fcm or info_web),
                'result': {'fcm': info_fcm, 'web': info_web},
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    return Response(
        {
            'success': True,
            'counts': {
                'tokens_total': len(tokens),
                'subs_total': len(subs),
            },
            'result': {
                'fcm': info_fcm,
                'web': info_web,
            },
        },
        status=status.HTTP_200_OK,
    )


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def push_register(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    data = request.data or {}
    token = (data.get('token') or '').strip()
    platform = (data.get('platform') or 'unknown').strip().lower()
    device_id = (data.get('device_id') or '').strip()

    if not token:
        return Response({'error': 'Token requerido'}, status=status.HTTP_400_BAD_REQUEST)

    obj, created = PushToken.objects.update_or_create(
        token=token,
        defaults={
            'user': request.user,
            'platform': platform,
            'device_id': device_id,
            'active': True,
        }
    )

    return Response(
        {
            'success': True,
            'created': created,
            'id': obj.id,
        },
        status=status.HTTP_200_OK
    )


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def push_unregister(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    data = request.data or {}
    token = (data.get('token') or '').strip()
    if not token:
        return Response({'error': 'Token requerido'}, status=status.HTTP_400_BAD_REQUEST)

    updated = PushToken.objects.filter(token=token, user=request.user).update(active=False)
    return Response({'success': True, 'updated': updated}, status=status.HTTP_200_OK)


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def push_send_test(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    data = request.data or {}
    title = (data.get('title') or 'GoToGym').strip()
    body = (data.get('body') or 'Notificación de prueba').strip()

    tokens = list(
        PushToken.objects.filter(user=request.user, active=True).values_list('token', flat=True)
    )
    subs = list(
        WebPushSubscription.objects.filter(user=request.user, active=True)
    )

    ok_fcm, info_fcm = _send_fcm(tokens, title, body, data={"source": "test"})
    ok_web, info_web = _send_web_push(subs, title, body, data={"source": "test"})

    if not ok_fcm and not ok_web:
        return Response({'success': False, 'error': str(info_fcm or info_web)}, status=status.HTTP_400_BAD_REQUEST)

    return Response(
        {
            'success': True,
            'result': {
                'fcm': info_fcm,
                'web': info_web,
            }
        },
        status=status.HTTP_200_OK
    )


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def push_web_public_key(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    public_key, _private_key, _email = _get_vapid_keys()
    if not public_key:
        return Response({'error': 'VAPID key no configurada'}, status=503)

    return Response({'public_key': public_key})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def push_web_subscribe(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    data = request.data or {}
    endpoint = (data.get('endpoint') or '').strip()
    keys = data.get('keys') or {}
    p256dh = (keys.get('p256dh') or '').strip()
    auth = (keys.get('auth') or '').strip()
    device_id = (data.get('device_id') or '').strip()

    if not endpoint or not p256dh or not auth:
        return Response({'error': 'Datos incompletos'}, status=status.HTTP_400_BAD_REQUEST)

    obj, created = WebPushSubscription.objects.update_or_create(
        endpoint=endpoint,
        defaults={
            'user': request.user,
            'p256dh': p256dh,
            'auth': auth,
            'device_id': device_id,
            'active': True,
        }
    )

    return Response({'success': True, 'created': created, 'id': obj.id})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def push_web_unsubscribe(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    data = request.data or {}
    endpoint = (data.get('endpoint') or '').strip()
    if not endpoint:
        return Response({'error': 'Endpoint requerido'}, status=status.HTTP_400_BAD_REQUEST)

    updated = WebPushSubscription.objects.filter(endpoint=endpoint, user=request.user).update(active=False)
    return Response({'success': True, 'updated': updated})

@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def device_connect(request, provider):
    # Preflight CORS
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    allowed = {'apple_health', 'google_fit', 'garmin', 'fitbit'}
    if provider not in allowed:
        return Response(
            {'success': False, 'error': 'Proveedor no soportado'},
            status=status.HTTP_400_BAD_REQUEST
        )

    return Response(
        {
            'success': True,
            'provider': provider,
            'connected': True
        },
        status=status.HTTP_200_OK
    )

@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def devices(request):
    connections = DeviceConnection.objects.filter(
        user=request.user
    ).order_by('-updated_at')

    connected = [
        {
            'provider': c.provider,
            'status': c.status,
            'connected': c.status == 'connected',
            'last_sync_at': c.last_sync_at,
            'updated_at': c.updated_at,
        }
        for c in connections
        if c.status != 'disconnected'  # opcional, pero recomendado
    ]

    providers = [
        {'provider': 'apple_health', 'label': 'Apple Health'},
        {'provider': 'google_fit', 'label': 'Google Fit'},
        {'provider': 'garmin', 'label': 'Garmin Connect'},
        {'provider': 'fitbit', 'label': 'Fitbit'},
    ]

    return Response({
        'providers': providers,
         'connected': connected
    }, status=status.HTTP_200_OK)

@api_view(['PUT', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def update_profile(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'PUT, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    try:
        if not getattr(request, "user", None) or not request.user.is_authenticated:
            return Response({'success': False, 'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

        user = request.user

        email = request.data.get('email')
        if email:
            exists = User.objects.filter(email=email).exclude(id=user.id).exists()
            if exists:
                return Response({'success': False, 'error': 'El email ya esta registrado'}, status=status.HTTP_400_BAD_REQUEST)
            user.email = email

        # Update fields
        if 'age' in request.data:
            try:
                user.age = int(request.data['age']) if str(request.data['age']).strip() != '' else None
            except Exception:
                pass
        if 'weight' in request.data:
            try:
                user.weight = float(request.data['weight']) if str(request.data['weight']).strip() != '' else None
            except Exception:
                pass
        if 'height' in request.data:
            try:
                user.height = float(request.data['height']) if str(request.data['height']).strip() != '' else None
            except Exception:
                pass

        if 'profile_picture' in request.FILES:
            upload = request.FILES['profile_picture']
            max_size_bytes = 8 * 1024 * 1024
            content_type = (upload.content_type or '').lower()
            if not content_type.startswith('image/'):
                return Response({'success': False, 'error': 'Formato de imagen no permitido'}, status=status.HTTP_400_BAD_REQUEST)
            if upload.size and upload.size > max_size_bytes:
                return Response({'success': False, 'error': 'La imagen supera el maximo permitido (8MB)'}, status=status.HTTP_400_BAD_REQUEST)
            base, ext = os.path.splitext(upload.name)
            ext = ext.lower() if ext else '.jpg'
            upload.name = f"profile_{user.id}_{uuid.uuid4().hex}{ext}"
            user.profile_picture = upload
        
        user.save()
        
        serializer = UserSerializer(user, context={"request": request})
        return Response({
            'success': True,
            'message': 'Perfil actualizado correctamente',
            'user': serializer.data
        })

    except Exception as e:
        print("Error updating profile:", str(e))
        return Response({'success': False, 'error': 'Error interno'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def get_users(request):
    try:
        if not getattr(request, "user", None) or not request.user.is_superuser:
            return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

        qs = request.query_params
        q = (qs.get('q') or qs.get('query') or '').strip()
        plan = (qs.get('plan') or '').strip()
        status_filter = (qs.get('status') or '').strip().lower()

        # Paginación opcional: si page/pageSize vienen, respondemos con {data,meta}
        page_raw = qs.get('page')
        page_size_raw = qs.get('pageSize') or qs.get('page_size')
        use_paging = page_raw is not None or page_size_raw is not None

        users = User.objects.all().order_by('-date_joined')
        if q:
            users = users.filter(Q(username__icontains=q) | Q(email__icontains=q) | Q(full_name__icontains=q))
        if plan in ("Gratis", "Premium"):
            users = users.filter(plan=plan)
        if status_filter:
            if status_filter == 'active':
                users = users.filter(is_active=True)
            elif status_filter in ('inactive', 'disabled', 'deleted'):
                users = users.filter(is_active=False)

        if not use_paging:
            serializer = UserSerializer(users, many=True, context={"request": request})
            return Response(serializer.data)

        try:
            page = int(page_raw or 1)
        except Exception:
            page = 1
        try:
            page_size = int(page_size_raw or 100)
        except Exception:
            page_size = 100

        page = max(1, page)
        page_size = max(1, min(500, page_size))
        total = users.count()
        start = (page - 1) * page_size
        items = users[start:start + page_size]
        serializer = UserSerializer(items, many=True, context={"request": request})
        return Response({"data": serializer.data, "meta": {"page": page, "pageSize": page_size, "total": total}})
    except Exception as e:
        print("Error fetching users:", str(e))
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def delete_user(request, user_id):
    try:
        if not getattr(request, "user", None) or not request.user.is_superuser:
            return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)
            
        if user.is_superuser:
            return Response({'error': 'No se puede eliminar un superusuario'}, status=status.HTTP_403_FORBIDDEN)

        reason = (request.data.get('reason') or '').strip() if isinstance(request.data, dict) else ''
        if not reason:
            return Response({'error': 'reason requerido'}, status=status.HTTP_400_BAD_REQUEST)
        before = {"id": user.id, "username": user.username, "email": user.email, "plan": user.plan, "is_active": getattr(user, "is_active", True)}
        original = _soft_delete_user(user)
        after = {"id": user_id, "is_active": False, "anonymized": True, "original": original}
        _audit_log(request, action="users.soft_delete", entity_type="user", entity_id=str(user_id), before=before, after=after, reason=reason)

        return Response({'success': True, 'message': 'Usuario eliminado (lógico) correctamente'})
    except Exception as e:
        return Response({'error': 'Error al eliminar usuario'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['PUT', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def set_user_active_admin(request, user_id):
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    if not getattr(request, "user", None) or not request.user.is_superuser:
        return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

    try:
        user = User.objects.get(id=user_id)
    except User.DoesNotExist:
        return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)

    if user.is_superuser:
        return Response({'error': 'No se puede suspender/reactivar un superusuario'}, status=status.HTTP_403_FORBIDDEN)

    data = request.data if isinstance(request.data, dict) else {}
    if 'is_active' not in data:
        return Response({'error': 'is_active requerido'}, status=status.HTTP_400_BAD_REQUEST)

    reason = (data.get('reason') or '').strip()
    if not reason:
        return Response({'error': 'reason requerido'}, status=status.HTTP_400_BAD_REQUEST)

    before = {"id": user.id, "username": user.username, "email": user.email, "plan": user.plan, "is_active": getattr(user, "is_active", True)}
    user.is_active = _as_bool(data.get('is_active'))
    user.save(update_fields=["is_active"])
    after = {"id": user.id, "username": user.username, "email": user.email, "plan": user.plan, "is_active": getattr(user, "is_active", True)}

    _audit_log(
        request,
        action="users.set_active",
        entity_type="user",
        entity_id=str(user_id),
        before=before,
        after=after,
        reason=reason,
    )

    return Response({'success': True, 'message': 'Estado actualizado', 'user': after})

@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def create_user_admin(request):
    try:
        if not getattr(request, "user", None) or not request.user.is_superuser:
            return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)
        data = request.data
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        plan = data.get('plan', 'Gratis')
        reason = (data.get('reason') or '').strip() if isinstance(data, dict) else ''

        if not username or not email or not password:
            return Response({'error': 'Todos los campos son obligatorios'}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({'error': 'El usuario ya existe'}, status=status.HTTP_400_BAD_REQUEST)

        if plan not in ("Gratis", "Premium"):
            plan = "Gratis"

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            plan=plan
        )

        _audit_log(
            request,
            action="users.create",
            entity_type="user",
            entity_id=str(user.id),
            before=None,
            after={"id": user.id, "username": user.username, "email": user.email, "plan": user.plan},
            reason=reason,
        )
        
        serializer = UserSerializer(user)
        return Response({
            'success': True, 
            'message': 'Usuario creado exitosamente',
            'user': serializer.data
        }, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(["POST"])
@permission_classes([AllowAny])
def internal_bootstrap_superuser(request):
    """Bootstrap controlado del primer superuser.

    Autorización: header X-Internal-Token debe coincidir con INTERNAL_ADMIN_BOOTSTRAP_TOKEN.
    Recomendación: setear el token solo para el bootstrap y luego removerlo.
    """

    expected = (os.getenv("INTERNAL_ADMIN_BOOTSTRAP_TOKEN", "") or "").strip()
    if not expected:
        return Response({"ok": False, "error": "bootstrap_not_configured"}, status=503)

    provided = (
        request.headers.get("X-Internal-Token")
        or request.META.get("HTTP_X_INTERNAL_TOKEN")
        or ""
    ).strip()
    if not provided or not secrets.compare_digest(provided, expected):
        return Response({"ok": False, "error": "unauthorized"}, status=401)

    # Safety: only allow if there are no superusers yet.
    if User.objects.filter(is_superuser=True).exists():
        return Response({"ok": False, "error": "already_bootstrapped"}, status=403)

    payload = request.data if isinstance(request.data, dict) else {}
    username = (payload.get("username") or "").strip()
    email = (payload.get("email") or "").strip()
    password = payload.get("password") or ""
    if not username or not password:
        return Response({"ok": False, "error": "username_password_required"}, status=400)

    if not email:
        email = f"{username}@gotogym.store"

    user, created = User.objects.get_or_create(username=username, defaults={"email": email})
    user.email = email
    user.is_staff = True
    user.is_superuser = True
    user.set_password(password)
    user.save()

    return Response({"ok": True, "created": created, "username": user.username})


@api_view(["GET"])
@permission_classes([AllowAny])
def internal_ocr_health(request):
    """Healthcheck interno para OCR (chat/documentos).

    Autorización: header X-Internal-Token debe coincidir con INTERNAL_OCR_HEALTH_TOKEN.
    Uso: GET /api/internal/ocr_health/?smoke=1
    """

    expected = (os.getenv("INTERNAL_OCR_HEALTH_TOKEN", "") or "").strip()
    if not expected:
        return Response({"ok": False, "error": "health_token_not_configured"}, status=503)

    provided = (
        request.headers.get("X-Internal-Token")
        or request.META.get("HTTP_X_INTERNAL_TOKEN")
        or ""
    ).strip()
    if not provided or not secrets.compare_digest(provided, expected):
        return Response({"ok": False, "error": "unauthorized"}, status=401)

    import shutil
    import subprocess

    def _which(cmd: str) -> str:
        try:
            return shutil.which(cmd) or ""
        except Exception:
            return ""

    def _first_line(output: str, limit: int = 200) -> str:
        if not output:
            return ""
        for line in (output.splitlines() or []):
            if (line or "").strip():
                return (line or "")[:limit]
        return ""

    def _run_version(cmd: str, args: list[str] | None = None):
        try:
            final_args = args if args is not None else ["--version"]
            proc = subprocess.run([cmd, *final_args], capture_output=True, text=True, timeout=3)
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            return proc.returncode == 0, _first_line(out)
        except Exception as ex:
            return False, f"failed:{ex}"[:200]

    tesseract_path = _which("tesseract")
    pdftoppm_path = _which("pdftoppm")
    t_ok, t_ver = (
        _run_version("tesseract", ["--version"]) if tesseract_path else (False, "not_found")
    )
    # `pdftoppm` (poppler-utils) usa `-v` y escribe a stderr; `--version` puede fallar.
    p_ok, p_ver = (
        _run_version("pdftoppm", ["-v"]) if pdftoppm_path else (False, "not_found")
    )

    langs = []
    if tesseract_path:
        try:
            proc = subprocess.run(
                ["tesseract", "--list-langs"],
                capture_output=True,
                text=True,
                timeout=4,
            )
            out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            # Normalmente: "List of available languages (2):" + langs
            langs = [ln for ln in lines if not ln.lower().startswith("list of available")]
            langs = langs[:30]
        except Exception:
            langs = []

    smoke = _as_bool(
        (request.query_params.get("smoke") if hasattr(request, "query_params") else request.GET.get("smoke"))
        or "false"
    )
    smoke_result = {
        "attempted": bool(smoke),
        "ok": False,
        "text": "",
        "error": "",
    }

    if smoke:
        if not (pytesseract and Image is not None and tesseract_path):
            smoke_result["error"] = "smoke_prereq_missing"
        else:
            try:
                from PIL import ImageDraw

                img = Image.new("RGB", (420, 140), color="white")
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), "TEST OCR 123", fill="black")
                text = pytesseract.image_to_string(img)
                smoke_result["text"] = (text or "").strip()[:200]
                smoke_result["ok"] = bool(smoke_result["text"])
                if not smoke_result["ok"]:
                    smoke_result["error"] = "smoke_empty"
            except Exception as ex:
                smoke_result["error"] = f"smoke_failed:{ex}"[:200]

    return Response(
        {
            "ok": True,
            "python": {
                "pytesseract": bool(pytesseract),
                "pdf2image": bool(convert_from_path),
                "pil": bool(Image is not None),
            },
            "binaries": {
                "tesseract": {
                    "found": bool(tesseract_path),
                    "path": tesseract_path,
                    "can_run": bool(t_ok),
                    "version": t_ver,
                },
                "pdftoppm": {
                    "found": bool(pdftoppm_path),
                    "path": pdftoppm_path,
                    "can_run": bool(p_ok),
                    "version": p_ver,
                },
            },
            "tesseract_langs": langs,
            "env": {
                "CHAT_ATTACHMENT_OCR": (os.getenv("CHAT_ATTACHMENT_OCR", "") or "").strip(),
                "CHAT_ATTACHMENT_TEXT_MAX_CHARS": (os.getenv("CHAT_ATTACHMENT_TEXT_MAX_CHARS", "") or "").strip(),
                "CHAT_ATTACHMENT_MAX_BYTES": (os.getenv("CHAT_ATTACHMENT_MAX_BYTES", "") or "").strip(),
                "INSTALL_OCR_SYSTEM_DEPS": (os.getenv("INSTALL_OCR_SYSTEM_DEPS", "") or "").strip(),
                "INSTALL_PDF_OCR_SYSTEM_DEPS": (os.getenv("INSTALL_PDF_OCR_SYSTEM_DEPS", "") or "").strip(),
                "CHAT_ATTACHMENT_INSTALL_OCR_DEPS": (os.getenv("CHAT_ATTACHMENT_INSTALL_OCR_DEPS", "") or "").strip(),
                "CHAT_ATTACHMENT_INSTALL_PDF_OCR_SYSTEM_DEPS": (os.getenv("CHAT_ATTACHMENT_INSTALL_PDF_OCR_SYSTEM_DEPS", "") or "").strip(),
                "CHAT_ATTACHMENT_INSTALL_PDF_OCR_DEPS": (os.getenv("CHAT_ATTACHMENT_INSTALL_PDF_OCR_DEPS", "") or "").strip(),
            },
            "smoke": smoke_result,
        }
    )


@api_view(["GET"])
@permission_classes([AllowAny])
def internal_vision_health(request):
    """Healthcheck interno para Vision (Azure OpenAI) usado por el chat.

    Autorización: header X-Internal-Token debe coincidir con INTERNAL_VISION_HEALTH_TOKEN
    (o, si no existe, con INTERNAL_OCR_HEALTH_TOKEN).

    Uso: GET /api/internal/vision_health/?smoke=1
    """

    expected = (
        (os.getenv("INTERNAL_VISION_HEALTH_TOKEN", "") or "").strip()
        or (os.getenv("INTERNAL_OCR_HEALTH_TOKEN", "") or "").strip()
    )
    if not expected:
        return Response({"ok": False, "error": "health_token_not_configured"}, status=503)

    provided = (
        request.headers.get("X-Internal-Token")
        or request.META.get("HTTP_X_INTERNAL_TOKEN")
        or ""
    ).strip()
    if not provided or not secrets.compare_digest(provided, expected):
        return Response({"ok": False, "error": "unauthorized"}, status=401)

    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT", "") or "").strip()
    deployment = (
        (os.getenv("AZURE_OPENAI_VISION_DEPLOYMENT", "") or "").strip()
        or (os.getenv("AZURE_OPENAI_DEPLOYMENT", "") or "").strip()
    )
    api_version = (os.getenv("AZURE_OPENAI_API_VERSION", "") or "").strip() or "2024-06-01"
    enabled = _as_bool(os.getenv("CHAT_ATTACHMENT_VISION", "true"))
    configured = bool(_azure_openai_vision_config())
    diagnostic = "" if configured else _azure_openai_vision_diagnostic()

    smoke = _as_bool(
        (request.query_params.get("smoke") if hasattr(request, "query_params") else request.GET.get("smoke"))
        or "false"
    )

    smoke_result = {
        "attempted": bool(smoke),
        "ok": False,
        "diagnostic": "",
        "preview": "",
        "time_ms": 0,
    }

    if smoke:
        if not configured:
            smoke_result["diagnostic"] = diagnostic or "vision_not_configured"
        else:
            try:
                import time

                png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+lmf8AAAAASUVORK5CYII="
                img_bytes = base64.b64decode(png_b64)
                start = time.time()
                desc, diag = _describe_image_with_azure_openai(
                    "internal://vision_smoke",
                    image_bytes=img_bytes,
                    content_type="image/png",
                )
                smoke_result["time_ms"] = int((time.time() - start) * 1000)
                smoke_result["ok"] = bool((desc or "").strip())
                smoke_result["diagnostic"] = (diag or "")[:350]
                smoke_result["preview"] = (desc or "").strip().replace("\n", " ")[:200]
                if not smoke_result["ok"] and not smoke_result["diagnostic"]:
                    smoke_result["diagnostic"] = "smoke_empty"
            except Exception as ex:
                smoke_result["diagnostic"] = f"smoke_failed:{ex}"[:200]

    return Response(
        {
            "ok": True,
            "configured": configured,
            "diagnostic": diagnostic,
            "config": {
                "CHAT_ATTACHMENT_VISION": str(enabled).lower(),
                "AZURE_OPENAI_ENDPOINT_set": bool(endpoint),
                "AZURE_OPENAI_API_KEY_set": bool((os.getenv("AZURE_OPENAI_API_KEY", "") or "").strip()),
                "AZURE_OPENAI_VISION_DEPLOYMENT": deployment or "",
                "AZURE_OPENAI_API_VERSION": api_version,
            },
            "limits": {
                "CHAT_VISION_MAX_BYTES": (os.getenv("CHAT_VISION_MAX_BYTES", "") or "").strip(),
            },
            "smoke": smoke_result,
        }
    )


@api_view(["POST", "OPTIONS"])
@permission_classes([AllowAny])
def internal_ocr_extract(request):
    """Extrae texto (PDF/imagen) de un archivo subido para debug.

    Autorización: header X-Internal-Token debe coincidir con INTERNAL_OCR_HEALTH_TOKEN.
    """

    if request.method == "OPTIONS":
        response = Response()
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Internal-Token"
        return response

    expected = (os.getenv("INTERNAL_OCR_HEALTH_TOKEN", "") or "").strip()
    if not expected:
        return Response({"ok": False, "error": "health_token_not_configured"}, status=503)

    provided = (
        request.headers.get("X-Internal-Token")
        or request.META.get("HTTP_X_INTERNAL_TOKEN")
        or ""
    ).strip()
    if not provided or not secrets.compare_digest(provided, expected):
        return Response({"ok": False, "error": "unauthorized"}, status=401)

    file_obj = request.FILES.get("file")
    if not file_obj:
        return Response({"ok": False, "error": "file_required"}, status=400)

    original_name = file_obj.name or "adjunto"
    content_type = (getattr(file_obj, "content_type", "") or "application/octet-stream").strip()
    file_bytes = file_obj.read() or b""
    extracted_text, diagnostic = _extract_text_from_file_bytes(original_name, file_bytes)

    return Response(
        {
            "ok": True,
            "file": {
                "name": original_name,
                "content_type": content_type,
                "bytes": len(file_bytes),
            },
            "extracted_text": extracted_text or "",
            "diagnostic": diagnostic or "",
            "ocr": _ocr_params(),
        }
    )

@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def predict_happiness(request):
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    try:
        data = request.data
        # Extraer username para guardar resultado
        username = data.get('username')
        scores = data.get('scores')  # Debe ser un dict con las 16 variables

        if not scores:
            return Response({'error': 'Faltan los scores'},
                            status=status.HTTP_400_BAD_REQUEST)

        # Predecir
        # predict_if_from_scores devuelve (pred_ridge, pred_stack)
        # Usamos pred_stack si existe, sino ridge
        try:
            final_score = _compute_final_score(scores)
        except Exception as ml_error:
            print("Error ML:", ml_error)
            return Response({'error': f"Error en modelo ML: {str(ml_error)}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Guardar en usuario si se provee
        if username:
            try:
                user = User.objects.get(username=username)
                user.happiness_index = final_score
                user.scores = scores  # Guardar los scores crudos
                user.save()

                # Log History  ✅ COMPLEMENTO: guardar también scores
                HappinessRecord.objects.create(
                    user=user,
                    value=final_score,
                    scores=scores
                )

                # Update Streak
                update_user_streak(user)

            except User.DoesNotExist:
                pass  # Si es anónimo, solo devolvemos el valor

        return Response({
            'success': True,
            'happiness_index': final_score,
            'happiness_percentage': round((final_score / 10.0) * 100, 1),

            # (Opcional) ✅ útil para depurar / n8n: devolver las 16 variables
            'happiness_scores': scores
        })

    except Exception as e:
        print("Error general prediction:", str(e))
        return Response({'error': str(e)},
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def get_if_question(request):
    username = request.query_params.get('username')
    slot = request.query_params.get('slot', '').strip() or None
    exclude = request.query_params.get('exclude', '')
    if not username:
        return Response({'error': 'Username required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

    _ensure_if_questions()
    week_id = _week_id()

    exclude_ids = {s.strip() for s in exclude.split(',') if s.strip()}
    answered_ids = set(
        IFAnswer.objects.filter(user=user, week_id=week_id)
        .values_list('question__key', flat=True)
    )
    seen = answered_ids | exclude_ids

    question = (
        IFQuestion.objects.filter(active=True)
        .exclude(key__in=seen)
        .order_by('order')
        .first()
    )

    total = IFQuestion.objects.filter(active=True).count()
    answered = IFAnswer.objects.filter(user=user, week_id=week_id).values('question').distinct().count()

    if not question:
        return Response({
            'completed': True,
            'week_id': week_id,
            'progress': {'answered': answered, 'total': total},
        })

    return Response({
        'completed': False,
        'week_id': week_id,
        'slot': slot,
        'question': {
            'id': question.key,
            'label': question.label,
            'order': question.order,
        },
        'progress': {'answered': answered, 'total': total},
    })


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def submit_if_answer(request):
    data = request.data
    score = data.get('score')
    value = score if score is not None else data.get('value')
    slot = data.get('slot')
    answered_at = data.get('answered_at')
    source = data.get('source', 'app')

    if value is None or not slot:
        return Response({'error': 'Faltan datos'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        value = int(value)
    except Exception:
        return Response({'error': 'Valor inválido'}, status=status.HTTP_400_BAD_REQUEST)

    if value < 0 or value > 10:
        return Response({'error': 'Valor fuera de rango'}, status=status.HTTP_400_BAD_REQUEST)

    if slot not in dict(IFAnswer.SLOT_CHOICES):
        return Response({'error': 'Slot inválido'}, status=status.HTTP_400_BAD_REQUEST)

    user = request.user
    _ensure_if_questions()

    if answered_at:
        try:
            parsed = datetime.fromisoformat(str(answered_at).replace('Z', '+00:00'))
            answered_date = parsed.date()
        except Exception:
            answered_date = timezone.localdate()
    else:
        answered_date = timezone.localdate()

    if IFAnswer.objects.filter(user=user, answered_date=answered_date, slot=slot).exists():
        return Response(
            {'error': 'Ya existe una respuesta para este horario'},
            status=status.HTTP_409_CONFLICT
        )

    week_id = _week_id(answered_date)
    answered_ids = set(
        IFAnswer.objects.filter(user=user, week_id=week_id)
        .values_list('question__key', flat=True)
    )
    question = (
        IFQuestion.objects.filter(active=True)
        .exclude(key__in=answered_ids)
        .order_by('order')
        .first()
    )

    if not question:
        return Response(
            {'error': 'Semana completada', 'completed': True},
            status=status.HTTP_409_CONFLICT
        )

    IFAnswer.objects.create(
        user=user,
        question=question,
        week_id=week_id,
        value=value,
        slot=slot,
        source=source,
        answered_date=answered_date,
    )

    # Gamificación: cada check-in cuenta como 1 acción válida del día.
    try:
        from api.gamification_service import update_user_streak
        update_user_streak(user, activity_date=answered_date, source=f"if_answer:{source}")
    except Exception:
        # Nunca bloquear el flujo del IF por gamificación.
        pass

    total = IFQuestion.objects.filter(active=True).count()
    answered = IFAnswer.objects.filter(user=user, week_id=week_id).values('question').distinct().count()

    payload = {
        'success': True,
        'week_id': week_id,
        'progress': {'answered': answered, 'total': total},
        'message': 'Respuesta guardada',
        'current_streak': int(getattr(user, 'current_streak', 0) or 0),
    }

    if answered >= total:
        answers = (
            IFAnswer.objects.filter(user=user, week_id=week_id)
            .select_related('question')
        )
        scores = {a.question.key: a.value for a in answers}
        final_score = _compute_final_score(scores)

        user.happiness_index = final_score
        user.scores = scores
        user.save()

        HappinessRecord.objects.create(
            user=user,
            value=final_score,
            scores=scores
        )
        update_user_streak(user)

        payload.update({
            'happiness_index': final_score,
            'happiness_percentage': round((final_score / 10.0) * 100, 1),
            'message': 'Semana completada'
        })

    return Response(payload)


@api_view(['POST'])
def mercadopago_webhook(request):
    access_token = os.getenv("MERCADOPAGO_ACCESS_TOKEN")
    if not access_token:
        return Response({'error': 'Webhook no configurado'}, status=500)

    payload = request.data or {}
    data_id = None

    if isinstance(payload, dict):
        data_id = payload.get('data', {}).get('id') or payload.get('id')

    if not data_id:
        return Response({'status': 'ignored'})

    try:
        r = requests.get(
            f"https://api.mercadopago.com/preapproval/{data_id}",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        if r.status_code != 200:
            return Response({'error': 'No se pudo consultar suscripción'}, status=400)
        sub = r.json()
    except Exception as e:
        return Response({'error': str(e)}, status=400)

    external_reference = sub.get('external_reference')
    user = None
    if external_reference:
        user = User.objects.filter(username=external_reference).first()
    if not user:
        user = User.objects.filter(subscription_id=data_id).first()
    if not user:
        return Response({'status': 'user_not_found'})

    status_str = sub.get('status', '')
    last_payment_status = sub.get('last_payment_status') or status_str
    next_payment_date = sub.get('next_payment_date')
    cancelled = sub.get('cancelled') or sub.get('cancelled_at') is not None

    if next_payment_date:
        try:
            current_period_end = datetime.fromisoformat(next_payment_date.replace('Z', '+00:00'))
        except Exception:
            current_period_end = None
    else:
        current_period_end = None

    user.subscription_provider = "mercadopago"
    user.subscription_id = data_id
    user.current_period_end = current_period_end
    user.cancel_at_period_end = bool(cancelled)
    user.last_payment_status = last_payment_status

    if status_str in ["authorized", "active"]:
        user.plan = "Premium"
        user.billing_status = "active"
        user.trial_active = False
    elif status_str in ["cancelled", "paused", "expired"]:
        user.billing_status = "canceled"
        if not user.trial_active:
            user.plan = "Gratis"
    else:
        user.billing_status = "pending"

    user.save()

    return Response({'status': 'ok'})


@api_view(['POST', 'PUT', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def update_profile_settings(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, PUT, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    try:
        user = _resolve_request_user(request)
        if not user:
            return Response({'success': False, 'error': 'Autenticacion requerida'}, status=401)

        photo_only = str(request.data.get('photo_only', '')).strip().lower() in ('1', 'true', 'yes', 'on')
        if photo_only and 'profile_picture' not in request.FILES:
            return Response({'success': False, 'error': 'No se recibió archivo de foto'}, status=400)
        
        # Handle text fields
        if not photo_only and 'email' in request.data:
            email = request.data['email']
            if email:
                exists = User.objects.filter(email=email).exclude(id=user.id).exists()
                if exists:
                    return Response({'success': False, 'error': 'El email ya esta registrado'}, status=400)
                user.email = email
        if not photo_only and 'profession' in request.data: user.profession = request.data['profession']
        if not photo_only and 'full_name' in request.data: user.full_name = request.data['full_name']
        if not photo_only and 'favorite_exercise_time' in request.data: user.favorite_exercise_time = request.data['favorite_exercise_time']
        if not photo_only and 'favorite_sport' in request.data: user.favorite_sport = request.data['favorite_sport']

        # Exp-003 (perfil metabólico): sexo biológico
        if not photo_only and 'sex' in request.data:
            sx = str(request.data.get('sex') or '').strip().lower()
            user.sex = sx if sx in ('male', 'female') else None

        # Exp-002 (meta + actividad)
        if not photo_only and 'goal_type' in request.data:
            gt = str(request.data.get('goal_type') or '').strip().lower()
            user.goal_type = gt if gt in ('deficit', 'maintenance', 'gain') else None
        if not photo_only and 'activity_level' in request.data:
            al = str(request.data.get('activity_level') or '').strip().lower()
            user.activity_level = al if al in ('low', 'moderate', 'high') else None
        if not photo_only and 'daily_target_kcal_override' in request.data:
            raw = str(request.data.get('daily_target_kcal_override') or '').strip()
            if raw == '':
                user.daily_target_kcal_override = None
            else:
                try:
                    v = float(raw)
                    user.daily_target_kcal_override = v if v > 0 else None
                except Exception:
                    pass
        if not photo_only and 'age' in request.data:
            try:
                user.age = int(request.data['age']) if str(request.data['age']).strip() != '' else None
            except Exception:
                pass
        if not photo_only and 'weight' in request.data:
            try:
                user.weight = float(request.data['weight']) if str(request.data['weight']).strip() != '' else None
            except Exception:
                pass
        if not photo_only and 'height' in request.data:
            try:
                raw = str(request.data['height']).strip()
                if raw == '':
                    user.height = None
                else:
                    h = float(raw)
                    # UX: en la UI el label es "Altura (m)", pero muchos usuarios ingresan cm (ej. 175).
                    # Normalización: si es >3, asumimos cm y convertimos a metros.
                    if h > 3.0:
                        h = h / 100.0
                    # Guardrail (m): rango humano típico
                    if 0.5 <= h <= 2.6:
                        user.height = float(h)
            except Exception:
                pass
        # Add other fields as needed

        password = request.data.get('password') if not photo_only else None
        password_confirm = request.data.get('password_confirm') if not photo_only else None
        if (not photo_only) and (password or password_confirm):
            if password != password_confirm:
                return Response({'error': 'Las contraseñas no coinciden'}, status=400)
            if len(password) < 8 or not re.search(r'[A-Z]', password) or not re.search(r'[0-9]', password):
                return Response({'error': 'La contraseña no cumple los requisitos'}, status=400)
            user.set_password(password)
        
        # Handle Image
        if 'profile_picture' in request.FILES:
            upload = request.FILES['profile_picture']
            content_type = (upload.content_type or '').lower()
            if not content_type.startswith('image/'):
                return Response({'success': False, 'error': 'Formato de imagen no permitido'}, status=400)
            max_size_bytes = 8 * 1024 * 1024
            if upload.size and upload.size > max_size_bytes:
                return Response({'success': False, 'error': 'La imagen supera el maximo permitido (8MB)'}, status=400)

            base, ext = os.path.splitext(upload.name)
            ext = ext.lower() if ext else '.jpg'
            upload.name = f"profile_{user.id}_{uuid.uuid4().hex}{ext}"
            user.profile_picture = upload
            
        user.save()
        
        serializer = UserSerializer(user, context={"request": request})
        user_data = serializer.data
        profile_picture = _profile_picture_db_value(user)
        profile_picture_url = _canonical_profile_picture_url(request, user)
        target_info = _storage_target_info()
        user_data['profile_picture'] = profile_picture
        user_data['profile_picture_url'] = profile_picture_url
        user_data.update(target_info)
        return Response({
            'success': True,
            'message': 'Perfil actualizado correctamente',
            'user': user_data,
            'profile_picture': profile_picture,
            'profile_picture_url': profile_picture_url,
            **target_info,
        })
    except User.DoesNotExist:
        return Response({'success': False, 'error': 'User not found'}, status=404)
    except Exception as e:
        return Response({'success': False, 'error': str(e)}, status=500)

@api_view(['POST', 'OPTIONS'])
def update_single_score(request):
    """
    Actualiza 1 sola variable (ej: s_sleep=8), 
    rellena el resto con lo que ya tenía el usuario (o 5 por defecto),
    recalcula el IF y devuelve el nuevo %.
    """
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    try:
        data = request.data
        username = data.get('username')
        variable = data.get('variable') # ej: 's_sleep'
        value = data.get('value')       # ej: 8

        if not username or not variable or value is None:
            return Response({'error': 'Faltan datos'}, status=status.HTTP_400_BAD_REQUEST)

        # Buscar usuario
        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)

        # Obtener scores previos o iniciar default
        current_scores = user.scores if user.scores else {}
        
        # Validar consistencia (si faltan keys, rellenar con 5)
        # Lista de vars esperadas por el modelo (podríamos importarlas, pero hardcode de seguridad)
        ALL_VARS = [
            "s_steps","s_sleep","s_stress_inv","s_intensity","s_emotional",
            "s_social","s_hrv","s_bio_age","s_sleep_quality","s_circadian",
            "s_focus","s_mood_sust","s_flow","s_purpose","s_hobbies","s_prosocial"
        ]
        
        # Rellenar faltantes con 5
        for v in ALL_VARS:
            if v not in current_scores:
                current_scores[v] = 5

        # Actualizar la variable solicitada
        current_scores[variable] = int(value)

        # Recalcular IF
        try:
            val_ridge, val_stack = predict_if_from_scores(current_scores)
            final_score = val_stack if val_stack is not None else val_ridge
        except Exception as ml_error:
            return Response({'error': f"Error ML: {ml_error}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Guardar todo
        user.scores = current_scores
        user.happiness_index = final_score
        user.save()

        # Log History
        HappinessRecord.objects.create(user=user, value=final_score)

        # Update Streak
        update_user_streak(user)

        percentage = round((final_score / 10.0) * 100, 1)

        return Response({
            'success': True,
            'message': f'{variable} actualizado a {value}',
            'happiness_index': final_score,
            'happiness_percentage': percentage
        })

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def get_happiness_history(request):
    username = request.query_params.get('username')
    if not username:
        return Response({'error': 'Username required'}, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        user = User.objects.get(username=username)
        # Get last 20 records chronological
        history = HappinessRecord.objects.filter(user=user).order_by('-date')[:20]
        # Reverse to show valid timeline (oldest -> newest)
        # Format date as Day/Month Hour:Minute
        data = [{'value': h.value, 'date': h.date.strftime('%d/%m %H:%M')} for h in reversed(history)]
        
        return Response(data)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)

 # streak/badges logic lives in gamification_service.update_user_streak

@api_view(['PUT', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def update_user_admin(request, user_id):
    try:
        if not getattr(request, "user", None) or not request.user.is_superuser:
            return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)

        data = request.data if isinstance(request.data, dict) else {}
        reason = (data.get('reason') or '').strip()
        before = {"id": user.id, "username": user.username, "email": user.email, "plan": user.plan, "is_active": getattr(user, "is_active", True)}

        if 'username' in data and data['username']:
            user.username = str(data['username']).strip()
        if 'email' in data and data['email']:
            user.email = str(data['email']).strip()
        requested_plan = None
        if 'plan' in data and data['plan']:
            plan = str(data['plan']).strip()
            if plan in ("Gratis", "Premium"):
                requested_plan = plan

        if requested_plan and requested_plan != before.get('plan'):
            if not reason:
                return Response({'error': 'reason requerido para cambio de plan'}, status=status.HTTP_400_BAD_REQUEST)
            user.plan = requested_plan

        user.save()
        after = {"id": user.id, "username": user.username, "email": user.email, "plan": user.plan, "is_active": getattr(user, "is_active", True)}

        _audit_log(
            request,
            action="users.update_admin",
            entity_type="user",
            entity_id=str(user_id),
            before=before,
            after=after,
            reason=reason,
        )

        return Response({'success': True, 'message': 'Usuario actualizado'})
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def get_global_history(request):
    try:
        if not getattr(request, "user", None) or not request.user.is_superuser:
            return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)
        # Group by day and calc average
        # Limit to last 30 days roughly
        thirty_days_ago = date.today() - timedelta(days=30)
        
        history = HappinessRecord.objects.filter(date__gte=thirty_days_ago)\
            .annotate(day=TruncDate('date'))\
            .values('day')\
            .annotate(avg_value=Avg('value'))\
            .order_by('day')
            
        data = [{'date': h['day'].strftime('%d/%m'), 'value': round(h['avg_value'], 1)} for h in history]
        return Response(data)
    except Exception as e:
        print(e)
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def upload_medical_record(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response
    try:
        username = (request.data.get('username') or '').strip()
        file_obj = request.FILES.get('file')
        doc_type = (request.data.get('doc_type') or request.POST.get('doc_type') or '').strip()

        if not getattr(request, "user", None) or not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, status=401)

        if username and username != request.user.username:
            return Response({'error': 'Forbidden'}, status=403)

        if not file_obj:
            return Response({'error': 'File required'}, status=400)

        if doc_type not in ('nutrition_plan', 'training_plan', 'medical_history'):
            return Response({'error': 'doc_type inválido'}, status=400)

        user = request.user
        username = user.username
            
        original_name = file_obj.name or "documento"
        extension = os.path.splitext(original_name)[1].lower()
        safe_base = get_valid_filename(os.path.splitext(original_name)[0]) or "documento"
        safe_name = f"{safe_base}_{uuid.uuid4().hex[:8]}{extension}"

        file_bytes = file_obj.read()
        content_type = (file_obj.content_type or 'application/octet-stream').strip() or 'application/octet-stream'
        safe_username = username.replace("/", "_")
        blob_path = f"{safe_username}/{safe_name}"

        with tempfile.NamedTemporaryFile(delete=False, suffix=extension or "") as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name

        if _is_azure_blob_enabled():
            container_name = _doc_container_map().get(doc_type, 'historiaclinica')
            file_url = _upload_file_to_blob(container_name, blob_path, file_bytes, content_type)
            file_path = temp_path
        else:
            file_path, file_url = _save_local_document(doc_type, username, safe_name, file_bytes)
        
        # EXTRACT TEXT (PDF + OCR para imágenes)
        extracted_text = ""
        try:
            lower_name = original_name.lower()
            if lower_name.endswith('.pdf'):
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                for page in reader.pages:
                    extracted_text += (page.extract_text() or "") + "\n"
                # OCR fallback si el PDF es escaneado o viene vacío
                if not extracted_text.strip():
                    if pytesseract and convert_from_path:
                        pages = convert_from_path(file_path, dpi=200, first_page=1, last_page=10)
                        ocr_chunks = []
                        for img in pages:
                            try:
                                ocr_chunks.append(pytesseract.image_to_string(img))
                            except Exception:
                                pass
                        extracted_text = "\n".join([c for c in ocr_chunks if c.strip()])
                    elif pytesseract:
                        # Intento directo con tesseract si soporta PDF
                        try:
                            extracted_text = pytesseract.image_to_string(file_path)
                        except Exception:
                            extracted_text = "[OCR no disponible: instala pdf2image y poppler]"
            elif lower_name.endswith(('.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif')):
                if pytesseract and Image is not None:
                    img = Image.open(file_path)
                    extracted_text = pytesseract.image_to_string(img)
                else:
                    extracted_text = "[OCR no disponible: instala pytesseract]"
        except Exception as ex:
            print(f"Extraction Warning: {ex}")
            if not extracted_text:
                extracted_text = "[No se pudo extraer texto automáticamente]"

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        # Guardar/actualizar documento en perfil
        extracted_text = _clean_extracted_text(extracted_text)
        doc_key = doc_type or 'medical_history'
        existing = UserDocument.objects.filter(user=user, doc_type=doc_key).first()
        previous_file_url = existing.file_url if existing else ""

        UserDocument.objects.update_or_create(
            user=user,
            doc_type=doc_key,
            defaults={
                "file_name": original_name,
                "file_url": file_url,
                "extracted_text": extracted_text,
            }
        )

        if previous_file_url and previous_file_url != file_url:
            _delete_blob_if_exists(previous_file_url)

        signed_url = _build_signed_blob_url(file_url)
        include_text = _as_bool(request.data.get('include_text') or request.POST.get('include_text'))

        response_payload = {
            'success': True,
            'file_url': signed_url,
            'file_url_raw': file_url,
            'doc_type': doc_key
        }
        if include_text:
            response_payload['extracted_text'] = extracted_text
        return Response(response_payload)
    except Exception as e:
        print(f"Upload Error: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['POST', 'OPTIONS'])
@permission_classes([AllowAny])
def upload_chat_attachment(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    try:
        # IMPORTANTE:
        # No usamos autenticación/permisos DRF aquí porque si el request es multipart grande
        # y el framework responde 401 antes de consumir el body, App Service/ARR puede devolver 502.
        # Forzamos el parse (request.data / request.FILES) y luego validamos JWT manualmente.
        username = (request.data.get('username') or '').strip()
        file_obj = request.FILES.get('file')
        include_text = _as_bool(request.data.get('include_text') or request.POST.get('include_text'))

        auth_result = None
        try:
            auth_result = JWTAuthentication().authenticate(request)
        except Exception:
            auth_result = None

        if not auth_result:
            return Response({'error': 'Authentication required'}, status=401)

        user, _validated_token = auth_result

        if username and username != user.username:
            return Response({'error': 'Forbidden'}, status=403)

        if not file_obj:
            return Response({'error': 'File required'}, status=400)

        max_bytes = int(os.getenv('CHAT_ATTACHMENT_MAX_BYTES', str(15 * 1024 * 1024)) or (15 * 1024 * 1024))
        if getattr(file_obj, 'size', None) and file_obj.size > max_bytes:
            return Response({'error': 'Archivo demasiado grande'}, status=413)

        original_name = file_obj.name or "adjunto"
        extension = os.path.splitext(original_name)[1].lower()
        allowed_ext = (
            '.pdf',
            '.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif'
        )
        if extension not in allowed_ext:
            return Response({'error': 'Tipo de archivo no permitido'}, status=400)

        safe_base = get_valid_filename(os.path.splitext(original_name)[0]) or "adjunto"
        safe_name = f"{safe_base}_{uuid.uuid4().hex[:8]}{extension}"

        file_bytes = file_obj.read()
        if len(file_bytes) > max_bytes:
            return Response({'error': 'Archivo demasiado grande'}, status=413)
        content_type = (file_obj.content_type or 'application/octet-stream').strip() or 'application/octet-stream'
        safe_username = user.username.replace("/", "_")
        blob_path = f"{safe_username}/{safe_name}"

        extracted_text = None
        extracted_text_diagnostic = ""
        if include_text:
            extracted_text, diagnostic = _extract_text_from_file_bytes(original_name, file_bytes)
            if not extracted_text.strip():
                extracted_text = ""
                extracted_text_diagnostic = (
                    "No se pudo extraer texto del adjunto automáticamente. "
                    f"diag={diagnostic}. "
                    "Para OCR de imágenes instala tesseract. Para OCR de PDF escaneado se requiere poppler + tesseract."
                )

        if _is_azure_blob_enabled():
            container_name = _chat_attachment_container()
            file_url = _upload_file_to_blob(container_name, blob_path, file_bytes, content_type)
        else:
            return Response({'error': 'Storage no configurado'}, status=503)

        signed_url = _build_signed_blob_url(file_url)
        payload = {'success': True, 'file_url': signed_url, 'file_url_raw': file_url}
        if include_text:
            payload['extracted_text'] = extracted_text or ''
            if extracted_text_diagnostic and not (extracted_text or '').strip():
                payload['extracted_text_diagnostic'] = extracted_text_diagnostic
        return Response(payload)
    except Exception as e:
        print(f"Upload chat attachment error: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def speech_to_text(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    audio = request.FILES.get('audio')
    if not audio:
        return Response({'error': 'Audio requerido'}, status=400)

    speech_key = os.getenv('AZURE_SPEECH_KEY', '').strip()
    speech_region = os.getenv('AZURE_SPEECH_REGION', '').strip()
    language = (request.data.get('language') or 'es-ES').strip()

    if not speech_key or not speech_region:
        return Response({'error': 'STT no configurado'}, status=503)

    try:
        endpoint = (
            f"https://{speech_region}.stt.speech.microsoft.com/"
            "speech/recognition/conversation/cognitiveservices/v1"
        )
        headers = {
            'Ocp-Apim-Subscription-Key': speech_key,
            'Content-Type': audio.content_type or 'audio/wav',
        }
        params = {
            'language': language,
        }
        audio_bytes = audio.read()
        if len(audio_bytes) > 10 * 1024 * 1024:
            return Response({'error': 'Audio demasiado grande'}, status=413)

        resp = requests.post(
            endpoint,
            params=params,
            headers=headers,
            data=audio_bytes,
            timeout=30,
        )
        try:
            payload = resp.json()
        except Exception:
            payload = {}

        if resp.status_code != 200:
            return Response(
                {
                    'error': 'STT error',
                    'detail': payload or resp.text,
                },
                status=502,
            )

        text = payload.get('DisplayText') or ''
        if not text and payload.get('NBest'):
            text = payload['NBest'][0].get('Display', '')
        if not text:
            return Response({'error': 'Sin texto reconocido'}, status=422)

        return Response({'text': text})
    except requests.RequestException as exc:
        return Response({'error': 'STT request failed', 'detail': str(exc)}, status=502)

@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def user_documents(request):
    username = request.query_params.get('username')
    doc_type = request.query_params.get('doc_type')
    include_text = _as_bool(request.query_params.get('include_text'))
    if not username:
        return Response({'error': 'Username required'}, status=400)

    user, auth_error = _require_authenticated_user(request, username)
    if auth_error:
        return auth_error

    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=404)

    qs = UserDocument.objects.filter(user=user)
    if doc_type:
        qs = qs.filter(doc_type=doc_type)

    doc = qs.order_by('-updated_at').first()
    if not doc:
        return Response({'success': True, 'document': None})

    signed_url = _build_signed_blob_url(doc.file_url)

    document_payload = {
        'doc_type': doc.doc_type,
        'file_name': doc.file_name,
        'file_url': signed_url,
        'file_url_raw': doc.file_url,
        'updated_at': doc.updated_at.isoformat(),
    }
    if include_text:
        document_payload['extracted_text'] = doc.extracted_text

    return Response({
        'success': True,
        'document': document_payload,
    })

@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def user_documents_delete(request):
    username = request.data.get('username')
    doc_type = request.data.get('doc_type')
    if not username or not doc_type:
        return Response({'error': 'Username and doc_type required'}, status=400)

    user, auth_error = _require_authenticated_user(request, username)
    if auth_error:
        return auth_error

    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        return Response({'error': 'User not found'}, status=404)

    qs = UserDocument.objects.filter(user=user, doc_type=doc_type)
    if not qs.exists():
        return Response({'success': True, 'message': 'No document found'})

    docs = list(qs)
    for doc in docs:
        _delete_blob_if_exists(doc.file_url)
    qs.delete()
    return Response({'success': True})

@api_view(['POST'])
def chat_n8n(request):
    """
    Proxy que recibe mensajes del frontend y los envía al Webhook de n8n.
    """
    try:
        # 1. Obtener datos del frontend
        message = request.data.get('message')
        session_id = request.data.get('sessionId') or ''
        attachment_url = request.data.get('attachment')
        attachment_text = request.data.get('attachment_text')  # Nuevo: Texto extraído
        attachment_text_diagnostic = request.data.get('attachment_text_diagnostic') or ""

        # QAF (porciones/calorías) - confirmación por botones + contexto
        confirmed_portions = request.data.get('confirmed_portions')
        goal_kcal_meal = request.data.get('goal_kcal_meal')
        qaf_context = request.data.get('qaf_context')

        auth_header = request.headers.get('Authorization', '')

        if not message and not attachment_url:
            return Response({'error': 'Mensaje o adjunto vacío'}, status=400)

        # 2. Configuración de n8n
        n8n_url = getattr(settings, "N8N_WEBHOOK_URL", "").strip() or "http://172.200.202.47/webhook/general-agent-gotogym-v2"

        # Resolver usuario temprano (para fallback de adjuntos y sessionId seguro)
        user = _resolve_request_user(request)
        if user:
            session_id = _safe_session_id(user)
        elif not session_id or session_id == 'invitado':
            session_id = f"guest_{uuid.uuid4().hex}"

        # Sync por evento (contextual): si el usuario expresa fatiga/estrés, encolar una sync rápida.
        try:
            if user and message:
                msg_low = str(message).lower()
                if re.search(r"\b(cansad|agotad|deprim|estres|ansios|sin energia|fatig)\b", msg_low):
                    enqueue_sync_request(user, provider="", reason="chat_signal", priority=3)
        except Exception:
            pass

        # Normalizar: si viene placeholder, no lo tratamos como texto real.
        if _is_attachment_text_placeholder(attachment_text or ""):
            attachment_text = ""

        # Fallback de adjunto: si hay URL pero no texto, descargar y extraer (solo attachments del usuario)
        try:
            if user and attachment_url and not (attachment_text or "").strip():
                container_name, blob_name = _extract_blob_ref_from_url(str(attachment_url))
                if container_name == _chat_attachment_container() and blob_name:
                    safe_username = user.username.replace("/", "_")
                    if blob_name.startswith(f"{safe_username}/"):
                        max_bytes = int(os.getenv('CHAT_ATTACHMENT_MAX_BYTES', str(15 * 1024 * 1024)) or (15 * 1024 * 1024))
                        resp = requests.get(str(attachment_url), timeout=20)
                        if resp.status_code == 200 and resp.content and len(resp.content) <= max_bytes:
                            filename = os.path.basename(blob_name)
                            extracted, diagnostic = _extract_text_from_file_bytes(filename, resp.content)
                            attachment_text = extracted or ""
                            if diagnostic and not (attachment_text or "").strip():
                                attachment_text_diagnostic = f"fallback_extraction_failed: {diagnostic}"
        except Exception as ex:
            print(f"Attachment fallback extraction warning: {ex}")

        # Vision para imágenes (comida/porciones): para fotos suele ser mejor que OCR.
        # Nota: por restricciones típicas (n8n/LLM no puede hacer fetch), preferimos mandar bytes como data URL.
        vision_parsed = None
        qaf_result = None
        qaf_text_for_output_override = ""

        # Exp-003 (perfil metabólico): quick-actions + cálculo semanal
        quick_actions_out = []
        metabolic_result = None
        metabolic_text_for_output_override = ""

        # Exp-004 (meal planner): resultado + texto
        meal_plan_result = None
        meal_plan_text_for_output_override = ""

        # Exp-005 (predictor de tendencia corporal): resultado + texto
        body_trend_result = None
        body_trend_text_for_output_override = ""

        # Exp-006 (postura): resultado + texto + quick-actions
        posture_result = None
        posture_text_for_output_override = ""
        posture_quick_actions_out = []

        # Exp-007 (lifestyle): resultado + texto + quick-actions
        lifestyle_result = None
        lifestyle_text_for_output_override = ""
        lifestyle_quick_actions_out: list[dict[str, Any]] = []

        # Exp-008 (motivación): resultado + texto + quick-actions
        motivation_result = None
        motivation_text_for_output_override = ""
        motivation_quick_actions_out: list[dict[str, Any]] = []

        # Exp-009 (progresión): resultado + texto + quick-actions
        progression_result = None
        progression_text_for_output_override = ""
        progression_quick_actions_out: list[dict[str, Any]] = []

        # Guardrail UX: si el usuario está en un flujo de Postura, no mezclar check-ins/metabólico.
        try:
            pr0 = request.data.get('posture_request') if isinstance(request.data, dict) else None
            msg_low0 = str(message or '').lower()
            suppress_weekly_checkins = bool(isinstance(pr0, dict) or re.search(r"\b(postura|posture)\b", msg_low0))
        except Exception:
            suppress_weekly_checkins = False

        def _week_weights_from_state(state, week_id: str):
            try:
                ww = state.get('weekly_weights') if isinstance(state.get('weekly_weights'), dict) else {}
                row = ww.get(week_id)
                if isinstance(row, dict):
                    v = row.get('avg_weight_kg')
                else:
                    v = row
                return float(v) if v is not None else None
            except Exception:
                return None

        def _set_week_avg_weight(state, week_id: str, avg_weight_kg: float, source: str):
            out = dict(state or {})
            ww = out.get('weekly_weights') if isinstance(out.get('weekly_weights'), dict) else {}
            ww2 = dict(ww)
            ww2[week_id] = {
                'avg_weight_kg': float(avg_weight_kg),
                'source': str(source or 'chat'),
                'recorded_at': timezone.now().isoformat(),
            }
            out['weekly_weights'] = ww2
            return out

        # 0) Aplicar payloads de quick-actions (sin cambios de UI): actualizar perfil o registrar peso semanal
        try:
            if user and isinstance(request.data, dict):
                profile_updates = request.data.get('profile_updates')
                if isinstance(profile_updates, dict) and 'sex' in profile_updates:
                    sx = str(profile_updates.get('sex') or '').strip().lower()
                    if sx in ('male', 'female'):
                        user.sex = sx
                        user.save(update_fields=['sex'])
        except Exception:
            pass

        # 0.1) Peso promedio semanal desde el chat
        try:
            if user and isinstance(request.data, dict) and request.data.get('weekly_weight_avg_kg') is not None:
                try:
                    avg_w = float(request.data.get('weekly_weight_avg_kg'))
                except Exception:
                    avg_w = None
                if avg_w and avg_w > 0:
                    week_id_now = _week_id()
                    state = getattr(user, 'coach_weekly_state', {}) or {}
                    user.coach_weekly_state = _set_week_avg_weight(state, week_id_now, avg_w, source='chat')
                    user.coach_weekly_updated_at = timezone.now()
                    try:
                        user.weight = float(avg_w)
                    except Exception:
                        pass
                    user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at', 'weight'])
        except Exception:
            pass

        # 0.2) Snooze: no volver a preguntar esta semana
        try:
            if user and isinstance(request.data, dict) and str(request.data.get('weekly_checkin_snooze') or '').strip().lower() in ('1', 'true', 'yes', 'on'):
                cs = getattr(user, 'coach_state', {}) or {}
                cs2 = dict(cs)
                cs2['weekly_checkin_snoozed_week_id'] = _week_id()
                cs2['weekly_checkin_prompted_week_id'] = _week_id()
                user.coach_state = cs2
                user.coach_state_updated_at = timezone.now()
                user.save(update_fields=['coach_state', 'coach_state_updated_at'])
        except Exception:
            pass

        # 0.3) Exp-004: generar menú por payload o por intención en el mensaje
        try:
            if user:
                # Acciones directas desde botones (quick-actions)
                if isinstance(request.data, dict) and request.data.get('meal_plan_apply') is True:
                    try:
                        week_id_now = _week_id()
                        weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
                        ws2 = dict(weekly_state)
                        ws2['meal_plan_active_week_id'] = week_id_now
                        ws2['meal_plan_active_at'] = timezone.now().isoformat()
                        user.coach_weekly_state = ws2
                        user.coach_weekly_updated_at = timezone.now()
                        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                            "[MENÚ] Estado: aplicado para esta semana."
                        )
                    except Exception:
                        pass

                view_mode = request.data.get('meal_plan_view') if isinstance(request.data, dict) else None
                if isinstance(view_mode, str) and view_mode.strip() == 'shopping_list':
                    try:
                        week_id_now = _week_id()
                        weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
                        mp = weekly_state.get('meal_plan') if isinstance(weekly_state.get('meal_plan'), dict) else {}
                        stored = mp.get(week_id_now)
                        current_result = None
                        if isinstance(stored, dict) and 'result' in stored and isinstance(stored.get('result'), dict):
                            current_result = stored.get('result')
                        elif isinstance(stored, dict) and 'plan' in stored:
                            current_result = stored
                        if isinstance(current_result, dict):
                            from .qaf_meal_planner.engine import render_shopping_list_text
                            shop_text = render_shopping_list_text(current_result)
                            if shop_text:
                                meal_plan_text_for_output_override = shop_text
                                attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[LISTA DE COMPRAS]\n{shop_text}".strip()
                    except Exception:
                        pass

                if isinstance(request.data, dict) and isinstance(request.data.get('meal_plan_mutate'), dict):
                    try:
                        m = request.data.get('meal_plan_mutate')
                        day_index = int(m.get('day_index'))
                        slot = str(m.get('slot') or '').strip().lower()
                        direction = str(m.get('direction') or 'normal').strip().lower()
                        if direction not in ('simple', 'normal', 'high'):
                            direction = 'normal'
                        if slot in ('desayuno', 'almuerzo', 'cena', 'snack'):
                            week_id_now = _week_id()
                            weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
                            mp = weekly_state.get('meal_plan') if isinstance(weekly_state.get('meal_plan'), dict) else {}
                            stored = mp.get(week_id_now)
                            current_result = None
                            if isinstance(stored, dict) and 'result' in stored and isinstance(stored.get('result'), dict):
                                current_result = stored.get('result')
                            elif isinstance(stored, dict) and 'plan' in stored:
                                current_result = stored
                            if isinstance(current_result, dict):
                                from .qaf_meal_planner.engine import mutate_plan_slot, render_professional_summary
                                seed = (hash(f"{user.id}:{week_id_now}:{direction}:{day_index}:{slot}") & 0xFFFFFFFF)
                                mutated = mutate_plan_slot(
                                    result=current_result,
                                    day_index=day_index,
                                    slot=slot,
                                    direction=direction,
                                    seed=int(seed),
                                    locale=((request.data.get('locale') or '').strip() or 'es-CO'),
                                    exclude_item_ids=[],
                                )
                                meal_plan_result = mutated
                                menu_text = render_professional_summary(mutated)
                                if menu_text:
                                    meal_plan_text_for_output_override = menu_text
                                    attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[MENÚ ACTUALIZADO]\n{menu_text}".strip()
                                try:
                                    ws2 = dict(weekly_state)
                                    mp2 = dict(mp)
                                    mp2[week_id_now] = {'result': mutated, 'updated_at': timezone.now().isoformat(), 'week_id': week_id_now}
                                    ws2['meal_plan'] = mp2
                                    user.coach_weekly_state = ws2
                                    user.coach_weekly_updated_at = timezone.now()
                                    user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                                except Exception:
                                    pass
                    except Exception:
                        pass

                meal_req = request.data.get('meal_plan_request') if isinstance(request.data, dict) else None
                want_menu = False
                if isinstance(meal_req, dict):
                    want_menu = True
                else:
                    msg_low = str(message or '').lower()
                    if re.search(r"\b(men[uú]|meal\s*plan|plan\s+de\s+comidas|menu\s+semanal|men[uú]\s+semanal)\b", msg_low):
                        want_menu = True

                if want_menu:
                    variety = None
                    meals_per_day = None
                    kcal_day = None
                    exclude_item_ids = []
                    locale = (request.data.get('locale') or '').strip() if isinstance(request.data, dict) else ''
                    locale = locale or 'es-CO'

                    if isinstance(meal_req, dict):
                        variety = str(meal_req.get('variety') or '').strip().lower() or None
                        try:
                            meals_per_day = int(meal_req.get('meals_per_day')) if meal_req.get('meals_per_day') is not None else None
                        except Exception:
                            meals_per_day = None
                        try:
                            kcal_day = float(meal_req.get('kcal_day')) if meal_req.get('kcal_day') is not None else None
                        except Exception:
                            kcal_day = None
                        if isinstance(meal_req.get('exclude_item_ids'), list):
                            exclude_item_ids = [str(x) for x in meal_req.get('exclude_item_ids') if str(x).strip()]
                    if len(exclude_item_ids) > 50:
                        exclude_item_ids = exclude_item_ids[:50]

                    if variety not in ('simple', 'normal', 'high'):
                        variety = 'normal'
                    if meals_per_day not in (3, 4):
                        meals_per_day = 3

                    if kcal_day is None:
                        if getattr(user, 'daily_target_kcal_override', None):
                            try:
                                kcal_day = float(user.daily_target_kcal_override)
                            except Exception:
                                kcal_day = None
                    if kcal_day is None:
                        try:
                            weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
                            kcal_day = float(((weekly_state.get('metabolic_last') or {}).get('kcal_day')) or 0.0) or None
                        except Exception:
                            kcal_day = None
                    if kcal_day is None:
                        kcal_day = 2000.0

                    week_id_now = _week_id()
                    exclude_item_ids = [str(x).strip() for x in exclude_item_ids if str(x).strip()][:50]
                    exclude_sig = ",".join(sorted(exclude_item_ids))
                    signature = f"v0|{week_id_now}|{variety}|{int(round(float(kcal_day)))}|{int(meals_per_day)}|{exclude_sig}"
                    seed = (hash(f"{user.id}:{signature}") & 0xFFFFFFFF)

                    from .qaf_meal_planner.engine import (
                        build_quick_actions_for_menu,
                        generate_week_plan,
                        render_professional_summary as render_menu_summary,
                    )

                    # Cache por variantes (coach_weekly_state)
                    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
                    cached = None
                    try:
                        variants = weekly_state.get('meal_plan_variants') if isinstance(weekly_state.get('meal_plan_variants'), dict) else {}
                        wk = variants.get(week_id_now) if isinstance(variants.get(week_id_now), dict) else {}
                        cached = wk.get(signature)
                    except Exception:
                        cached = None

                    if isinstance(cached, dict) and cached.get('result') and isinstance(cached.get('result'), dict):
                        meal_plan_result = cached.get('result')
                    else:
                        meal_plan_result = generate_week_plan(
                            kcal_day=float(kcal_day),
                            meals_per_day=int(meals_per_day),
                            variety_level=variety,
                            exclude_item_ids=exclude_item_ids,
                            seed=int(seed),
                            locale=locale,
                        )

                    # Persistir siempre para continuidad de UX
                    try:
                        ws2 = dict(weekly_state)
                        mp = ws2.get('meal_plan') if isinstance(ws2.get('meal_plan'), dict) else {}
                        mp2 = dict(mp)
                        mp2[week_id_now] = {
                            'result': meal_plan_result,
                            'updated_at': timezone.now().isoformat(),
                            'week_id': week_id_now,
                            'signature': signature,
                        }
                        ws2['meal_plan'] = mp2

                        variants = ws2.get('meal_plan_variants') if isinstance(ws2.get('meal_plan_variants'), dict) else {}
                        wk = variants.get(week_id_now) if isinstance(variants.get(week_id_now), dict) else {}
                        wk2 = dict(wk)
                        wk2[signature] = {
                            'result': meal_plan_result,
                            'updated_at': timezone.now().isoformat(),
                            'signature': signature,
                        }
                        if len(wk2) > 3:
                            keys = sorted(wk2.keys())
                            for k in keys[:-3]:
                                wk2.pop(k, None)
                        variants2 = dict(variants)
                        variants2[week_id_now] = wk2
                        ws2['meal_plan_variants'] = variants2

                        user.coach_weekly_state = ws2
                        user.coach_weekly_updated_at = timezone.now()
                        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                    except Exception:
                        pass

                    menu_text = render_menu_summary(meal_plan_result)
                    if menu_text:
                        meal_plan_text_for_output_override = menu_text
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[MENÚ SEMANAL PROPUESTO]\n{menu_text}".strip()

                    # Quick-actions para regeneración (wow: 1 tap)
                    try:
                        quick_actions_out.extend(build_quick_actions_for_menu(variety_level=variety))
                    except Exception:
                        pass
        except Exception as ex:
            print(f"QAF meal planner warning: {ex}")

        # 0.4) Exp-005: Predictor de tendencias corporales (6 semanas)
        try:
            if user:
                bt_req = request.data.get('body_trend_request') if isinstance(request.data, dict) else None
                want_trend = False
                if isinstance(bt_req, dict):
                    want_trend = True
                else:
                    msg_low = str(message or '').lower()
                    if re.search(r"\b(proyecci[oó]n|tendencia|si\s+contin[uú]o|peso\s+en\s+6|6\s+semanas|escenario)\b", msg_low):
                        want_trend = True

                # Capturar kcal promedio si el usuario lo escribió (MVP)
                kcal_in_from_text = None
                try:
                    if message:
                        m = re.search(r"\b(\d{3,4})\s*(kcal|cal)\b", str(message).lower())
                        if m:
                            kcal_in_from_text = float(m.group(1))
                except Exception:
                    kcal_in_from_text = None

                if want_trend:
                    week_id_now = _week_id()
                    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

                    # pesos semanales
                    cur_w = None
                    prev_w = None
                    try:
                        ww = weekly_state.get('weekly_weights') if isinstance(weekly_state.get('weekly_weights'), dict) else {}
                        row = ww.get(week_id_now)
                        cur_w = float(row.get('avg_weight_kg')) if isinstance(row, dict) else float(row)
                        prev_keys = [k for k in ww.keys() if isinstance(k, str) and k != week_id_now]
                        if prev_keys:
                            prev_key = sorted(prev_keys)[-1]
                            row2 = ww.get(prev_key)
                            prev_w = float(row2.get('avg_weight_kg')) if isinstance(row2, dict) else float(row2)
                    except Exception:
                        cur_w = getattr(user, 'weight', None)
                        prev_w = None

                    # kcal ingreso promedio: request > texto > cache
                    kcal_in_avg = None
                    if isinstance(bt_req, dict) and bt_req.get('kcal_in_avg_day') is not None:
                        try:
                            kcal_in_avg = float(bt_req.get('kcal_in_avg_day'))
                        except Exception:
                            kcal_in_avg = None
                    if kcal_in_avg is None and kcal_in_from_text is not None:
                        kcal_in_avg = kcal_in_from_text
                    try:
                        kbw = weekly_state.get('kcal_avg_by_week') if isinstance(weekly_state.get('kcal_avg_by_week'), dict) else {}
                        if kcal_in_avg is None and week_id_now in kbw:
                            row = kbw.get(week_id_now)
                            kcal_in_avg = float(row.get('kcal_in_avg_day')) if isinstance(row, dict) else float(row)
                    except Exception:
                        pass

                    # tdee + recomendación desde metabolic_last
                    tdee = None
                    reco = None
                    try:
                        ml = weekly_state.get('metabolic_last') if isinstance(weekly_state.get('metabolic_last'), dict) else {}
                        tdee = float(ml.get('tdee_effective_kcal_day')) if ml.get('tdee_effective_kcal_day') is not None else None
                        reco = float(ml.get('kcal_day')) if ml.get('kcal_day') is not None else None
                    except Exception:
                        tdee = None
                        reco = None
                    # Guardrail: descartar valores absurdos (p.ej. bug legacy de unidades)
                    try:
                        if tdee is not None and float(tdee) > 10000.0:
                            tdee = None
                    except Exception:
                        tdee = None
                    try:
                        if reco is not None and float(reco) > 10000.0:
                            reco = None
                    except Exception:
                        reco = None
                    if tdee is None:
                        tdee = float(reco) if reco is not None else None

                    profile = {
                        'tdee_kcal_day': tdee,
                        'recommended_kcal_day': (float(getattr(user, 'daily_target_kcal_override', None)) if getattr(user, 'daily_target_kcal_override', None) else reco),
                    }
                    observations = {
                        'weight_current_week_avg_kg': cur_w,
                        'weight_previous_week_avg_kg': prev_w,
                        'kcal_in_avg_day': kcal_in_avg,
                    }

                    from .qaf_body_trend.engine import evaluate_body_trend, render_professional_summary, build_quick_actions_for_trend

                    body_trend_result = evaluate_body_trend(profile, observations, horizon_weeks=6).payload

                    # Si el usuario pidió un escenario específico, priorizarlo en el texto (sin cambiar JSON)
                    scenario = None
                    if isinstance(bt_req, dict):
                        scenario = str(bt_req.get('scenario') or '').strip().lower() or None

                    bt_text = render_professional_summary(body_trend_result)
                    if bt_text:
                        body_trend_text_for_output_override = bt_text
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[PROYECCIÓN CORPORAL (6 SEMANAS)]\n{bt_text}".strip()

                    # Persistir kcal promedio si se obtuvo
                    if kcal_in_avg is not None and kcal_in_avg > 0:
                        try:
                            ws2 = dict(weekly_state)
                            kbw = ws2.get('kcal_avg_by_week') if isinstance(ws2.get('kcal_avg_by_week'), dict) else {}
                            kbw2 = dict(kbw)
                            kbw2[week_id_now] = {'kcal_in_avg_day': float(kcal_in_avg), 'updated_at': timezone.now().isoformat()}
                            ws2['kcal_avg_by_week'] = kbw2
                            ws2['body_trend_last'] = {'result': body_trend_result, 'updated_at': timezone.now().isoformat()}
                            user.coach_weekly_state = ws2
                            user.coach_weekly_updated_at = timezone.now()
                            user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                        except Exception:
                            pass

                    # Quick-actions
                    try:
                        quick_actions_out.extend(build_quick_actions_for_trend(has_intake=(kcal_in_avg is not None)))
                    except Exception:
                        pass
        except Exception as ex:
            print(f"QAF body trend warning: {ex}")

        # 0.5) Exp-006: Postura (requiere keypoints ya calculados)
        try:
            if user:
                pr = request.data.get('posture_request') if isinstance(request.data, dict) else None
                want_posture = False
                if isinstance(pr, dict):
                    want_posture = True
                else:
                    msg_low = str(message or '').lower()
                    if re.search(r"\b(postura|posture|hombros\s+adelantados|cabeza\s+adelantada|joroba)\b", msg_low):
                        want_posture = True

                # Router por imagen: si Vision clasificó como entrenamiento, activar guía de captura postural.
                try:
                    if (not want_posture) and isinstance(vision_parsed, dict):
                        vr = str(vision_parsed.get('route') or '').strip().lower()
                        if vr in ('training', 'entrenamiento'):
                            want_posture = True
                except Exception:
                    pass

                if want_posture:
                    from .qaf_posture.engine import evaluate_posture, render_professional_summary

                    # Seguridad mínima: no inferimos desde 'visión descriptiva'. Solo usamos keypoints.
                    poses = None
                    user_ctx = {}
                    locale = 'es-CO'
                    if isinstance(pr, dict):
                        poses = pr.get('poses') if isinstance(pr.get('poses'), dict) else None
                        user_ctx = pr.get('user_context') if isinstance(pr.get('user_context'), dict) else {}
                        locale = (pr.get('locale') or '').strip() or 'es-CO'

                    def _posture_payload_ok(p: dict) -> bool:
                        if not isinstance(p, dict):
                            return False
                        for v in ('front', 'side'):
                            pose = p.get(v)
                            if not isinstance(pose, dict):
                                return False
                            kps = pose.get('keypoints')
                            if not isinstance(kps, list) or not kps:
                                return False
                            if len(kps) > 80:
                                return False
                            for kp in kps[:40]:
                                if not isinstance(kp, dict):
                                    return False
                                if not str(kp.get('name') or '').strip():
                                    return False
                                if kp.get('x') is None or kp.get('y') is None:
                                    return False
                        return True

                    if isinstance(poses, dict) and _posture_payload_ok(poses):
                        posture_result = evaluate_posture({'poses': poses, 'user_context': user_ctx, 'locale': locale}).payload
                        ptext = render_professional_summary(posture_result)
                        if ptext:
                            posture_text_for_output_override = ptext
                            attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[POSTURA]\n{ptext}".strip()

                        # Si el motor no concluye, ofrecer retomar captura.
                        try:
                            if isinstance(posture_result, dict) and posture_result.get('decision') == 'needs_confirmation':
                                posture_quick_actions_out.extend([
                                    {'label': 'Repetir frontal', 'type': 'open_camera'},
                                    {'label': 'Repetir lateral', 'type': 'open_camera'},
                                    {'label': 'Adjuntar fotos', 'type': 'open_attach'},
                                ])
                        except Exception:
                            pass
                    else:
                        # UX: guiar captura con botones sin introducir pantallas nuevas.
                        posture_quick_actions_out.extend([
                            {
                                'label': 'Tomar foto frontal',
                                'type': 'open_camera',
                            },
                            {
                                'label': 'Tomar foto lateral',
                                'type': 'open_camera',
                            },
                            {
                                'label': 'Adjuntar fotos',
                                'type': 'open_attach',
                            },
                        ])
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                            "[POSTURA / CAPTURA]\n"
                            "Necesito 2 fotos: frontal y lateral (cuerpo completo, buena luz, cámara a la altura del pecho, 2–3m).\n"
                            "Cuando tengas los keypoints (pose estimation), podré calcular señales y darte una rutina específica."
                        )
        except Exception as ex:
            print(f"QAF posture warning: {ex}")

        # 0.6) Exp-007: Estado de hoy (Lifestyle Intelligence)
        try:
            if user:
                lr = request.data.get('lifestyle_request') if isinstance(request.data, dict) else None
                habit_done = request.data.get('lifestyle_habit_done') if isinstance(request.data, dict) else None

                want_lifestyle = False
                if isinstance(lr, dict) or isinstance(habit_done, dict):
                    want_lifestyle = True
                else:
                    msg_low = str(message or '').lower()
                    if re.search(r"\b(estado\s+de\s+hoy|como\s+voy\s+hoy|mi\s+energia\s+hoy|dhss|lifestyle)\b", msg_low):
                        want_lifestyle = True

                # Registrar hábito como hecho (sin UI nueva)
                try:
                    if isinstance(habit_done, dict):
                        hid = str(habit_done.get('id') or '').strip()
                        if hid:
                            cs = getattr(user, 'coach_state', {}) or {}
                            cs2 = dict(cs)
                            done = cs2.get('lifestyle_done') if isinstance(cs2.get('lifestyle_done'), dict) else {}
                            dkey = timezone.localdate().isoformat()
                            day_list = done.get(dkey) if isinstance(done.get(dkey), list) else []
                            day_list2 = [str(x) for x in day_list if str(x).strip()]
                            if hid not in day_list2:
                                day_list2.append(hid)
                            done2 = dict(done)
                            done2[dkey] = day_list2[:10]
                            cs2['lifestyle_done'] = done2
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                except Exception:
                    pass

                if want_lifestyle:
                    from datetime import timedelta
                    from .qaf_lifestyle.engine import evaluate_lifestyle, render_professional_summary

                    days_i = 14
                    if isinstance(lr, dict) and lr.get('days') is not None:
                        try:
                            days_i = int(lr.get('days'))
                        except Exception:
                            days_i = 14
                    days_i = max(3, min(30, int(days_i)))

                    # Self-report puede venir por confirmaciones rápidas
                    self_report = {}
                    if isinstance(lr, dict) and isinstance(lr.get('self_report'), dict):
                        self_report = dict(lr.get('self_report'))

                    # daily_metrics desde FitnessSync (último por día)
                    start_dt = timezone.now() - timedelta(days=days_i)
                    qs = DevicesFitnessSync.objects.filter(user=user, created_at__gte=start_dt).only('created_at', 'metrics').order_by('created_at')
                    by_day = {}
                    for s in qs:
                        d = timezone.localdate(s.created_at).isoformat()
                        by_day[d] = (s.metrics or {})

                    daily_metrics = []
                    for d, m in sorted(by_day.items()):
                        row = {'date': d}
                        if isinstance(m, dict):
                            for k in ('steps', 'sleep_minutes', 'calories', 'resting_heart_rate_bpm', 'avg_heart_rate_bpm', 'distance_m', 'distance_km'):
                                if k in m:
                                    row[k] = m.get(k)
                        daily_metrics.append(row)

                    # memory desde coach_state
                    cs = getattr(user, 'coach_state', {}) or {}
                    mem = cs.get('lifestyle_memory') if isinstance(cs.get('lifestyle_memory'), dict) else {}

                    lifestyle_result = evaluate_lifestyle({'daily_metrics': daily_metrics, 'self_report': self_report, 'memory': mem}).payload
                    ltext = render_professional_summary(lifestyle_result)
                    if ltext:
                        lifestyle_text_for_output_override = ltext
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[ESTADO DE HOY]\n{ltext}".strip()

                    # Persistir memoria mínima (last_ids)
                    try:
                        micro = lifestyle_result.get('microhabits') if isinstance(lifestyle_result.get('microhabits'), list) else []
                        ids = [str(x.get('id')) for x in micro if isinstance(x, dict) and x.get('id')]
                        mem2 = dict(mem)
                        mem2['last_ids'] = ids[:3]
                        cs2 = dict(cs)
                        cs2['lifestyle_memory'] = mem2
                        cs2['lifestyle_last'] = {'result': lifestyle_result, 'updated_at': timezone.now().isoformat()}
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    # Quick actions de confirmación mínima (sin usar follow_up_questions del frontend)
                    try:
                        if isinstance(lifestyle_result, dict) and lifestyle_result.get('decision') == 'needs_confirmation':
                            missing = ((lifestyle_result.get('confidence') or {}).get('missing') or [])
                            missing = [str(x) for x in missing]

                            def _mk_scale_actions(prefix_label: str, key: str):
                                out = []
                                for i in range(1, 6):
                                    out.append({
                                        'label': f"{prefix_label} {i}/5",
                                        'type': 'message',
                                        'text': f"{prefix_label} {i}/5",
                                        'payload': {
                                            'lifestyle_request': {
                                                'days': days_i,
                                                'self_report': {**self_report, key: i},
                                            }
                                        },
                                    })
                                return out

                            if 'sleep' in missing:
                                lifestyle_quick_actions_out.extend(_mk_scale_actions('Sueño', 'sleep_quality_1_5'))
                            if 'steps' in missing:
                                lifestyle_quick_actions_out.extend(_mk_scale_actions('Movimiento', 'movement_1_5'))
                            # si no hay stress, permitir confirmarlo también
                            sig = lifestyle_result.get('signals') if isinstance(lifestyle_result.get('signals'), dict) else {}
                            if (sig.get('stress_inv') or {}).get('score01') is None:
                                lifestyle_quick_actions_out.extend(_mk_scale_actions('Estrés', 'stress_1_5'))

                            lifestyle_quick_actions_out = lifestyle_quick_actions_out[:6]
                        else:
                            # Si ya está aceptado, ofrecer marcar micro-hábitos como hechos (máx 3)
                            micro = lifestyle_result.get('microhabits') if isinstance(lifestyle_result.get('microhabits'), list) else []
                            for mh in micro[:3]:
                                if not isinstance(mh, dict):
                                    continue
                                hid = str(mh.get('id') or '').strip()
                                lab = str(mh.get('label') or '').strip()
                                if not hid or not lab:
                                    continue
                                # Etiqueta corta
                                short = (lab[:32] + '…') if len(lab) > 33 else lab
                                lifestyle_quick_actions_out.append({
                                    'label': f"✅ {short}",
                                    'type': 'message',
                                    'text': f"✅ {short}",
                                    'payload': {'lifestyle_habit_done': {'id': hid}},
                                })
                            lifestyle_quick_actions_out = lifestyle_quick_actions_out[:6]
                    except Exception:
                        pass
        except Exception as ex:
            print(f"QAF lifestyle warning: {ex}")

        # 0.7) Exp-008: Motivación (perfil + estado + reto) usando chat como sensor
        try:
            if user:
                mr = request.data.get('motivation_request') if isinstance(request.data, dict) else None
                ma = request.data.get('motivation_action') if isinstance(request.data, dict) else None

                want_motivation = False
                if isinstance(mr, dict) or isinstance(ma, dict):
                    want_motivation = True
                else:
                    msg_low = str(message or '').lower()
                    if re.search(r"\b(motivaci[oó]n|me\s+cuesta|no\s+quiero|no\s+pude|me\s+dio\s+pereza|estoy\s+agotad|estoy\s+cansad)\b", msg_low):
                        want_motivation = True

                cs = getattr(user, 'coach_state', {}) or {}
                mem = cs.get('motivation_memory') if isinstance(cs.get('motivation_memory'), dict) else {}
                prefs = cs.get('motivation_preferences') if isinstance(cs.get('motivation_preferences'), dict) else {}

                # Actualizar preferencias desde botones
                if isinstance(mr, dict) and isinstance(mr.get('preferences'), dict):
                    prefs = {**prefs, **mr.get('preferences')}

                # Registrar acciones (lo hago / modo fácil) como memoria simple
                if isinstance(ma, dict):
                    try:
                        cs2 = dict(cs)
                        acts = cs2.get('motivation_actions') if isinstance(cs2.get('motivation_actions'), list) else []
                        acts2 = [a for a in acts if isinstance(a, dict)][:20]
                        acts2.append({'at': timezone.now().isoformat(), 'action': ma})
                        cs2['motivation_actions'] = acts2[-20:]
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                        cs = cs2
                    except Exception:
                        pass

                    # Efectos UX: reconocer y (si aplica) acreditar racha por auto-reporte.
                    try:
                        if ma.get('accept') is True:
                            from api.gamification_service import update_user_streak
                            update_user_streak(user, source='motivation_accept')
                        if str(ma.get('mode') or '').strip().lower() == 'renacer_7d':
                            cs2 = getattr(user, 'coach_state', {}) or {}
                            cs3 = dict(cs2)
                            # ventana 7 días desde hoy (fecha local)
                            until_d = (timezone.localdate() + timezone.timedelta(days=7)).isoformat()
                            mem0 = cs3.get('motivation_memory') if isinstance(cs3.get('motivation_memory'), dict) else {}
                            mem1 = dict(mem0)
                            mem1['renacer_until'] = until_d
                            cs3['motivation_memory'] = mem1
                            cs3['motivation_mode'] = {'key': 'renacer_7d', 'until': until_d, 'started_at': timezone.now().isoformat()}
                            user.coach_state = cs3
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                            cs = cs3
                    except Exception:
                        pass

                # Inferir days_inactive por última interacción guardada
                try:
                    last_seen = mem.get('last_seen_at')
                    if isinstance(last_seen, str) and last_seen:
                        last_dt = timezone.datetime.fromisoformat(last_seen.replace('Z', '+00:00'))
                        delta = timezone.now() - last_dt
                        mem = dict(mem)
                        mem['days_inactive'] = int(delta.total_seconds() // (24 * 3600))
                except Exception:
                    pass

                if want_motivation:
                    # Si viene acción explícita, priorizar reconocimiento breve.
                    try:
                        if isinstance(ma, dict) and ma.get('accept') is True:
                            streak_now = int(getattr(user, 'current_streak', 0) or 0)
                            motivation_text_for_output_override = (
                                "Perfecto. Eso cuenta como constancia de hoy.\n"
                                f"Racha actual: {streak_now} días.\n"
                                "¿Quieres que subamos un poco el reto mañana o mantenemos estabilidad?"
                            )
                    except Exception:
                        pass
                    try:
                        if isinstance(ma, dict) and str(ma.get('mode') or '').strip().lower() == 'renacer_7d':
                            motivation_text_for_output_override = (
                                "Listo. Activé Modo Renacer por 7 días.\n"
                                "Objetivo: hábitos mínimos, sin presión.\n"
                                "Hoy solo vamos por el siguiente paso pequeño."
                            )
                    except Exception:
                        pass

                    lifestyle_last = None
                    try:
                        lifestyle_last = (cs.get('lifestyle_last') or {}).get('result')
                    except Exception:
                        lifestyle_last = None

                    gam = {
                        'streak': int(getattr(user, 'current_streak', 0) or 0),
                        'badges': getattr(user, 'badges', []) or [],
                    }

                    from .qaf_motivation.engine import evaluate_motivation, render_professional_summary

                    motivation_result = evaluate_motivation({
                        'message': str(message or ''),
                        'memory': mem,
                        'preferences': prefs,
                        'gamification': gam,
                        'lifestyle': lifestyle_last or {},
                    }).payload

                    mtext = render_professional_summary(motivation_result)
                    if mtext and not motivation_text_for_output_override:
                        motivation_text_for_output_override = mtext
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[MOTIVACIÓN]\n{mtext}".strip()

                    # Persistir memoria (vector + last_seen)
                    try:
                        vec = ((motivation_result.get('profile') or {}).get('vector')) if isinstance(motivation_result.get('profile'), dict) else None
                        mem2 = dict(mem)
                        if isinstance(vec, dict):
                            mem2['vector'] = vec
                        mem2['last_seen_at'] = timezone.now().isoformat()
                        cs2 = dict(cs)
                        cs2['motivation_memory'] = mem2
                        cs2['motivation_preferences'] = prefs
                        cs2['motivation_last'] = {'result': motivation_result, 'updated_at': timezone.now().isoformat()}
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                        cs = cs2
                    except Exception:
                        pass

                    # Quick actions: confirmación mínima (pressure) y CTAs
                    try:
                        if isinstance(motivation_result, dict) and motivation_result.get('decision') == 'needs_confirmation':
                            motivation_quick_actions_out.extend([
                                {
                                    'label': 'Suave',
                                    'type': 'message',
                                    'text': 'Suave',
                                    'payload': {'motivation_request': {'preferences': {'pressure': 'suave'}}},
                                },
                                {
                                    'label': 'Medio',
                                    'type': 'message',
                                    'text': 'Medio',
                                    'payload': {'motivation_request': {'preferences': {'pressure': 'medio'}}},
                                },
                                {
                                    'label': 'Firme',
                                    'type': 'message',
                                    'text': 'Firme',
                                    'payload': {'motivation_request': {'preferences': {'pressure': 'firme'}}},
                                },
                            ])
                        else:
                            cid = None
                            try:
                                cid = (motivation_result.get('challenge') or {}).get('id')
                            except Exception:
                                cid = None
                            motivation_quick_actions_out.append({
                                'label': '✅ Lo hago',
                                'type': 'message',
                                'text': '✅ Lo hago',
                                'payload': {'motivation_action': {'accept': True, 'challenge_id': cid}},
                            })
                            motivation_quick_actions_out.append({
                                'label': '🟡 Modo fácil 7 días',
                                'type': 'message',
                                'text': '🟡 Modo fácil 7 días',
                                'payload': {'motivation_action': {'mode': 'renacer_7d'}},
                            })
                        motivation_quick_actions_out = motivation_quick_actions_out[:6]
                    except Exception:
                        pass

                    # UX: si el usuario está interactuando con botones/payloads de motivación,
                    # respondemos directo para no depender de n8n (más consistente y rápido).
                    try:
                        if (isinstance(mr, dict) or isinstance(ma, dict)) and motivation_text_for_output_override:
                            qa_existing = quick_actions_out if isinstance(quick_actions_out, list) else []
                            qa_existing2 = [x for x in qa_existing if isinstance(x, dict)]
                            out_actions = (qa_existing2 + motivation_quick_actions_out)[:6] if motivation_quick_actions_out else qa_existing2[:6]
                            return Response({
                                'output': motivation_text_for_output_override,
                                'quick_actions': out_actions,
                                'qaf_motivation': motivation_result,
                            })
                    except Exception:
                        pass
        except Exception as ex:
            print(f"QAF motivation warning: {ex}")

        # 0.8) Exp-009: Progresión (fuerza + cardio)
        try:
            if user:
                pr = request.data.get('progression_request') if isinstance(request.data, dict) else None
                pa = request.data.get('progression_action') if isinstance(request.data, dict) else None

                want_prog = False
                if isinstance(pr, dict) or isinstance(pa, dict):
                    want_prog = True
                else:
                    msg_low = str(message or '').lower()
                    if re.search(r"\b(progres|estanc|plateau|subir\s+peso|subir\s+carga|mas\s+reps|cardio|correr|bici|eliptica)\b", msg_low):
                        want_prog = True

                if want_prog:
                    from .qaf_progression.engine import evaluate_progression, render_professional_summary, parse_strength_line

                    cs = getattr(user, 'coach_state', {}) or {}
                    mem = cs.get('progression_history') if isinstance(cs.get('progression_history'), dict) else {}

                    # Base session payload
                    session = {}
                    strength = None
                    cardio = None
                    if isinstance(pr, dict):
                        session = pr.get('session') if isinstance(pr.get('session'), dict) else {}
                        strength = pr.get('strength') if isinstance(pr.get('strength'), dict) else None
                        cardio = pr.get('cardio') if isinstance(pr.get('cardio'), dict) else None

                    # Parse desde texto si no vino fuerza
                    if not strength and message:
                        strength = parse_strength_line(str(message))

                    # Señales: reusar lifestyle + mood
                    lifestyle_last = (cs.get('lifestyle_last') or {}).get('result') if isinstance(cs.get('lifestyle_last'), dict) else None
                    mood = None
                    try:
                        mood = ((cs.get('motivation_last') or {}).get('result') or {}).get('state', {}).get('mood')
                    except Exception:
                        mood = None

                    signals = pr.get('signals') if isinstance(pr, dict) and isinstance(pr.get('signals'), dict) else {}
                    try:
                        if isinstance(lifestyle_last, dict):
                            sig = lifestyle_last.get('signals') if isinstance(lifestyle_last.get('signals'), dict) else {}
                            if signals.get('sleep_minutes') is None:
                                signals['sleep_minutes'] = (sig.get('sleep') or {}).get('value')
                            if signals.get('steps') is None:
                                signals['steps'] = (sig.get('steps') or {}).get('value')
                            if signals.get('resting_heart_rate_bpm') is None:
                                signals['resting_heart_rate_bpm'] = (sig.get('stress_inv') or {}).get('value')
                            if signals.get('lifestyle_band') is None:
                                signals['lifestyle_band'] = (lifestyle_last.get('dhss') or {}).get('band')
                    except Exception:
                        pass
                    if signals.get('mood') is None and mood:
                        signals['mood'] = mood

                    # Aplicar acciones de botones (paging / valores)
                    if isinstance(pa, dict):
                        # merge session updates
                        if isinstance(pa.get('session'), dict):
                            session = {**session, **pa.get('session')}
                        if isinstance(pa.get('cardio'), dict):
                            cardio = {**(cardio or {}), **pa.get('cardio')}
                        if isinstance(pa.get('strength'), dict):
                            strength = {**(strength or {}), **pa.get('strength')}

                    progression_result = evaluate_progression({
                        'session': session,
                        'strength': strength,
                        'cardio': cardio,
                        'history': mem,
                        'signals': signals,
                    }).payload

                    ptext = render_professional_summary(progression_result)
                    if ptext:
                        progression_text_for_output_override = ptext
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[PROGRESIÓN]\n{ptext}".strip()

                    # Quick actions guiadas para missing (sin UI nueva)
                    try:
                        missing = ((progression_result.get('confidence') or {}).get('missing') or [])
                        missing = [str(x) for x in missing]

                        def _rpe_page(high: bool):
                            out = []
                            rng = range(6, 11) if high else range(1, 6)
                            for i in rng:
                                out.append({
                                    'label': f"RPE {i}",
                                    'type': 'message',
                                    'text': f"RPE {i}",
                                    'payload': {'progression_action': {'session': {'rpe_1_10': i}}},
                                })
                            # toggle
                            out.append({
                                'label': 'RPE 6–10' if not high else 'RPE 1–5',
                                'type': 'message',
                                'text': 'RPE 6–10' if not high else 'RPE 1–5',
                                'payload': {'progression_action': {'ui': {'rpe_page': ('high' if not high else 'low')}}},
                            })
                            return out

                        rpe_page = None
                        try:
                            rpe_page = str(((pa or {}).get('ui') or {}).get('rpe_page') or '').strip().lower() if isinstance(pa, dict) else ''
                        except Exception:
                            rpe_page = ''

                        # UX: si falta modalidad, pedirla primero; si no, pedir RPE y luego % de cumplimiento.
                        if 'modality' in missing:
                            progression_quick_actions_out.append({
                                'label': 'Fuerza',
                                'type': 'message',
                                'text': 'Fuerza',
                                'payload': {'progression_action': {'strength': {'name': 'pendiente', 'sets': 0, 'reps': 0}}},
                            })
                            progression_quick_actions_out.append({
                                'label': 'Cardio',
                                'type': 'message',
                                'text': 'Cardio',
                                'payload': {'progression_action': {'cardio': {'minutes': 20}}},
                            })
                        elif 'rpe_1_10' in missing:
                            progression_quick_actions_out.extend(_rpe_page(high=(rpe_page == 'high'))[:6])
                        elif 'completion_pct' in missing and len(progression_quick_actions_out) < 6:
                            # completion presets
                            opts = [
                                (1.0, '100%'),
                                (0.8, '80%'),
                                (0.6, '60%'),
                                (0.4, '40%'),
                            ]
                            for v, lab in opts:
                                if len(progression_quick_actions_out) >= 6:
                                    break
                                progression_quick_actions_out.append({
                                    'label': f"Cumplí {lab}",
                                    'type': 'message',
                                    'text': f"Cumplí {lab}",
                                    'payload': {'progression_action': {'session': {'completion_pct': v}}},
                                })
                    except Exception:
                        pass

                    # Persistir history mínimo si accepted y hay datos
                    try:
                        if isinstance(progression_result, dict) and progression_result.get('decision') == 'accepted':
                            mem2 = dict(mem)
                            if isinstance(strength, dict) and strength.get('name') and strength.get('sets') and strength.get('reps'):
                                name = str(strength.get('name') or '').strip().lower()
                                sets = int(strength.get('sets') or 0)
                                reps = int(strength.get('reps') or 0)
                                load = strength.get('load_kg')
                                try:
                                    load_f = float(load) if load is not None else None
                                except Exception:
                                    load_f = None
                                ton = None
                                est1 = None
                                if load_f is not None and load_f > 0:
                                    ton = float(sets) * float(reps) * float(load_f)
                                    est1 = float(load_f) * (1.0 + (float(reps) / 30.0))
                                key = f"strength:{name}"
                                rows = mem2.get(key) if isinstance(mem2.get(key), list) else []
                                rows2 = [r for r in rows if isinstance(r, dict)][-2:]
                                rows2.append({'date': timezone.localdate().isoformat(), 'sets': sets, 'reps': reps, 'load_kg': load_f, 'tonnage': ton, 'est_1rm': est1, 'rpe': session.get('rpe_1_10')})
                                mem2[key] = rows2[-3:]
                            if isinstance(cardio, dict) and cardio.get('minutes') is not None:
                                rows = mem2.get('cardio:default') if isinstance(mem2.get('cardio:default'), list) else []
                                rows2 = [r for r in rows if isinstance(r, dict)][-2:]
                                rows2.append({'date': timezone.localdate().isoformat(), 'minutes': cardio.get('minutes'), 'avg_hr': cardio.get('avg_hr'), 'rpe': session.get('rpe_1_10')})
                                mem2['cardio:default'] = rows2[-3:]
                            cs2 = dict(cs)
                            cs2['progression_history'] = mem2
                            cs2['progression_last'] = {'result': progression_result, 'updated_at': timezone.now().isoformat()}
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    # Responder directo si viene de botones/payloads
                    try:
                        if (isinstance(pr, dict) or isinstance(pa, dict)) and progression_text_for_output_override:
                            existing = quick_actions_out if isinstance(quick_actions_out, list) else []
                            existing2 = [x for x in existing if isinstance(x, dict)]
                            out_actions = (existing2 + progression_quick_actions_out)[:6] if progression_quick_actions_out else existing2[:6]
                            return Response({
                                'output': progression_text_for_output_override,
                                'quick_actions': out_actions,
                                'qaf_progression': progression_result,
                            })
                    except Exception:
                        pass
        except Exception as ex:
            print(f"QAF progression warning: {ex}")

        # 0) Si llega qaf_context (ej. click en botones), usamos eso para estimar sin Vision.
        try:
            if isinstance(qaf_context, dict) and isinstance(qaf_context.get('vision'), dict):
                vision_parsed = qaf_context.get('vision')
        except Exception:
            vision_parsed = None

        def _parse_vision_json(text: str):
            raw = (text or "").strip()
            if not raw:
                return None

            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
                raw = re.sub(r"\s*```\s*$", "", raw)
                raw = raw.strip()

            try:
                return json.loads(raw)
            except Exception:
                pass

            try:
                i = raw.find("{")
                j = raw.rfind("}")
                if i >= 0 and j > i:
                    return json.loads(raw[i : j + 1])
            except Exception:
                return None
            return None
        try:
            if attachment_url:
                filename_guess = os.path.basename(urlparse(str(attachment_url)).path or "")
                vision_enabled = _as_bool(os.getenv("CHAT_ATTACHMENT_VISION", "true"))
                if _is_image_filename(filename_guess) and vision_enabled:
                    max_vision_bytes = int(os.getenv("CHAT_VISION_MAX_BYTES", str(4 * 1024 * 1024)) or (4 * 1024 * 1024))

                    image_bytes = None
                    image_content_type = None

                    had_text = bool((attachment_text or "").strip()) and not _is_attachment_text_placeholder(attachment_text)

                    # Intento seguro: solo descargamos bytes si es un blob de attachments del propio usuario.
                    try:
                        if user:
                            container_name, blob_name = _extract_blob_ref_from_url(str(attachment_url))
                            if container_name == _chat_attachment_container() and blob_name:
                                safe_username = user.username.replace("/", "_")
                                if blob_name.startswith(f"{safe_username}/"):
                                    resp = requests.get(str(attachment_url), timeout=20)
                                    if resp.status_code == 200 and resp.content and len(resp.content) <= max_vision_bytes:
                                        image_bytes = resp.content
                                        image_content_type = (resp.headers.get("Content-Type") or "image/png")
                                    elif resp.status_code != 200 and not attachment_text_diagnostic:
                                        attachment_text_diagnostic = f"vision_download_http_{resp.status_code}"[:350]
                                    elif resp.status_code == 200 and resp.content and len(resp.content) > max_vision_bytes and not attachment_text_diagnostic:
                                        attachment_text_diagnostic = "vision_download_too_large"[:350]
                    except Exception as ex:
                        if not attachment_text_diagnostic:
                            attachment_text_diagnostic = f"vision_download_failed: {ex}"[:350]

                    # 1) Preferimos bytes (data URL)
                    vision_desc, vision_diag = _describe_image_with_azure_openai(
                        str(attachment_url),
                        image_bytes=image_bytes,
                        content_type=image_content_type,
                    )

                    # 2) Fallback: si el intento con bytes/data URL falló, probamos con URL directa.
                    # (Si no había bytes, el primer intento ya fue con URL, así que no repetimos.)
                    if (not (vision_desc or "").strip()) and image_bytes:
                        vision_desc2, vision_diag2 = _describe_image_with_azure_openai(str(attachment_url))
                        vision_desc = vision_desc2 or vision_desc
                        vision_diag = vision_diag2 or vision_diag

                    if (vision_desc or "").strip():
                        pretty = vision_desc.strip()
                        try:
                            parsed = _parse_vision_json(pretty)
                            if isinstance(parsed, dict):
                                if isinstance(parsed.get("items"), str):
                                    items_raw = str(parsed.get("items") or "")
                                    parsed["items"] = [x.strip() for x in re.split(r"[,;\n]+", items_raw) if x.strip()]

                                vision_parsed = parsed
                                items = parsed.get("items")
                                if isinstance(items, list):
                                    items_str = ", ".join([str(x) for x in items if str(x).strip()])
                                else:
                                    items_str = ""
                                portion = str(parsed.get("portion_estimate") or "").strip()
                                notes = str(parsed.get("notes") or "").strip()
                                is_food = parsed.get("is_food")
                                route = str(parsed.get("route") or "").strip().lower()
                                route_conf = parsed.get("route_confidence")
                                summary = []
                                if route:
                                    summary.append(f"route: {route}")
                                if route_conf is not None:
                                    try:
                                        summary.append(f"route_confidence: {round(float(route_conf), 2)}")
                                    except Exception:
                                        pass
                                summary.append(f"is_food: {bool(is_food)}")
                                if items_str:
                                    summary.append(f"items: {items_str}")
                                if portion:
                                    summary.append(f"portion_estimate: {portion}")
                                if notes:
                                    summary.append(f"notes: {notes}")
                                pretty = "\n".join(summary)
                        except Exception:
                            pass

                        attachment_text = f"[DESCRIPCIÓN DE IMAGEN]\n{pretty}"
                    elif (not had_text) and vision_diag and not attachment_text_diagnostic:
                        attachment_text_diagnostic = f"vision_failed: {vision_diag}"[:350]
        except Exception as ex:
            print(f"Vision image description warning: {ex}")

        # Router: si la imagen cae en Salud, pedimos intención mínima (módulo) y evitamos inferencias.
        try:
            if attachment_url and isinstance(vision_parsed, dict):
                vr = str(vision_parsed.get('route') or '').strip().lower()
                if vr in ('health', 'salud'):
                    attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                        "[SALUD / IMAGEN]\n"
                        "Detecté una imagen tipo salud (primer plano de piel/músculo/rostro).\n"
                        "¿Qué quieres hacer con esta foto?\n"
                        "- Comparar/medir músculo\n"
                        "- Belleza/piel\n"
                        "Responde con una opción."
                    )
                    qa = quick_actions_out if isinstance(quick_actions_out, list) else []
                    qa.extend([
                        {'label': 'Comparar músculo', 'type': 'message', 'text': 'Comparar músculo'},
                        {'label': 'Belleza / piel', 'type': 'message', 'text': 'Belleza / piel'},
                    ])
                    quick_actions_out = qa[:6]
        except Exception:
            pass

        # Router: si la imagen cae en Entrenamiento, guiar captura postural (sin intentar calorías).
        try:
            if attachment_url and isinstance(vision_parsed, dict):
                vr = str(vision_parsed.get('route') or '').strip().lower()
                if vr in ('training', 'entrenamiento'):
                    # Guardrail UX: no mezclar check-ins/metabólico si el usuario está en modo entrenamiento por imagen.
                    try:
                        suppress_weekly_checkins = True
                    except Exception:
                        pass

                    attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                        "[ENTRENAMIENTO / IMAGEN]\n"
                        "Puedo ayudarte con técnica/postura, pero necesito 2 fotos: frontal y lateral (cuerpo completo, buena luz, cámara a la altura del pecho, 2–3m).\n"
                        "Si solo es una selfie o una foto casual, también puedo responder como Quantum Coach."
                    )
                    qa = quick_actions_out if isinstance(quick_actions_out, list) else []
                    qa.extend([
                        {'label': 'Tomar foto frontal', 'type': 'open_camera'},
                        {'label': 'Tomar foto lateral', 'type': 'open_camera'},
                        {'label': 'Adjuntar fotos', 'type': 'open_attach'},
                    ])
                    quick_actions_out = qa[:6]
        except Exception:
            pass

        # 1) QAF (post-proceso): solo si Vision confirmó ruta Nutrición/Comida.
        try:
            food_ok = False
            try:
                if isinstance(vision_parsed, dict):
                    vr = str(vision_parsed.get('route') or '').strip().lower()
                    is_food = vision_parsed.get('is_food')
                    food_ok = (is_food is True) or (vr in ('nutrition', 'nutricion', 'food'))
            except Exception:
                food_ok = False

            if food_ok and isinstance(vision_parsed, dict) and isinstance(vision_parsed.get('items'), list):
                from .qaf_calories.engine import (
                    load_aliases,
                    load_calorie_db,
                    load_items_meta,
                    load_micros_db,
                    load_nutrition_db,
                    normalize_item,
                    qaf_estimate_v2,
                    render_professional_summary,
                )

                cat = _qaf_catalog_paths()
                aliases = load_aliases(cat['aliases'])
                calorie_db = load_calorie_db(cat['calorie_db'])
                nutrition_db = load_nutrition_db(cat['nutrition_db'])
                micros_db = load_micros_db(cat['micros_db'])
                items_meta = load_items_meta(cat['items_meta'])

                locale = (request.data.get('locale') or '').strip() or 'es-CO'

                memory_hint: dict[str, float] = {}
                try:
                    if user:
                        vision_items = vision_parsed.get('items') or []
                        item_ids = []
                        for raw in vision_items:
                            iid = normalize_item(str(raw), aliases=aliases)
                            if iid and iid not in item_ids:
                                item_ids.append(iid)
                        if item_ids:
                            rows = QAFSoftMemoryPortion.objects.filter(user=user, item_id__in=item_ids)
                            for r in rows:
                                if r.grams_last and r.grams_last > 0:
                                    memory_hint[str(r.item_id)] = float(r.grams_last)
                except Exception:
                    memory_hint = {}

                qaf_result = qaf_estimate_v2(
                    vision_parsed,
                    calorie_db=calorie_db,
                    nutrition_db=nutrition_db,
                    micros_db=micros_db,
                    aliases=aliases,
                    items_meta=items_meta,
                    locale=locale,
                    memory_hint_by_item=memory_hint,
                    confirmed_portions=confirmed_portions if isinstance(confirmed_portions, list) else None,
                    goal_kcal_meal=goal_kcal_meal,
                )

                qaf_text = render_professional_summary(qaf_result)
                if qaf_text:
                    qaf_text_for_output_override = qaf_text
                    attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[CALORÍAS ESTIMADAS]\n{qaf_text}".strip()

                # 1.1) Exp-002: coherencia comida ↔ meta + alertas (si hay usuario)
                try:
                    if user and isinstance(qaf_result, dict):
                        from .qaf_goal_coherence.engine import evaluate_meal, render_professional_summary as render_goal_summary

                        meal_payload = {
                            "total_calories": qaf_result.get("total_calories"),
                            "uncertainty_score": ((qaf_result.get("uncertainty") or {}).get("uncertainty_score")),
                            "needs_confirmation": bool(qaf_result.get("needs_confirmation")),
                            "meal_slot": (request.data.get("meal_slot") or "unknown"),
                        }

                        user_context_payload = {
                            "weight_kg": getattr(user, "weight", None),
                            "age": getattr(user, "age", None),
                            "height_cm": (float(getattr(user, "height", None)) * 100.0) if getattr(user, "height", None) else None,
                            "goal_type": getattr(user, "goal_type", None),
                            "goal_text": None,
                            "activity_level": getattr(user, "activity_level", None),
                            "daily_target_kcal_override": getattr(user, "daily_target_kcal_override", None),
                        }

                        goal_eval = evaluate_meal(user_context_payload, meal_payload)
                        goal_text = render_goal_summary(goal_eval)
                        if goal_text:
                            attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[COHERENCIA CON META]\n{goal_text}".strip()
                except Exception as ex:
                    print(f"QAF goal coherence warning: {ex}")

                # Escritura memoria suave: solo confirmaciones explícitas
                try:
                    if user and isinstance(confirmed_portions, list):
                        for cp in confirmed_portions:
                            if not isinstance(cp, dict):
                                continue
                            iid = str(cp.get('item_id') or '').strip()
                            try:
                                g = float(cp.get('grams'))
                            except Exception:
                                continue
                            if not iid or g <= 0:
                                continue
                            obj, created = QAFSoftMemoryPortion.objects.get_or_create(
                                user=user,
                                item_id=iid,
                                defaults={"grams_last": float(g), "count_confirmed": 1},
                            )
                            if not created:
                                obj.grams_last = float(g)
                                obj.count_confirmed = int(obj.count_confirmed or 0) + 1
                                obj.save(update_fields=["grams_last", "count_confirmed", "updated_at"])
                except Exception:
                    pass

        except Exception as ex:
            print(f"QAF calories warning: {ex}")


        # Exp-003: Perfil metabólico dinámico (si hay usuario)
        try:
            if user and not suppress_weekly_checkins:
                week_id_now = _week_id()
                weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
                coach_state = getattr(user, 'coach_state', {}) or {}

                snoozed_week = str(coach_state.get('weekly_checkin_snoozed_week_id') or '')
                prompted_week = str(coach_state.get('weekly_checkin_prompted_week_id') or '')

                cur_avg = _week_weights_from_state(weekly_state, week_id_now)

                prev_avg = None
                try:
                    ww = weekly_state.get('weekly_weights') if isinstance(weekly_state.get('weekly_weights'), dict) else {}
                    prev_keys = [k for k in ww.keys() if isinstance(k, str) and k != week_id_now]
                    if prev_keys:
                        prev_key = sorted(prev_keys)[-1]
                        prev_avg = _week_weights_from_state(weekly_state, prev_key)
                except Exception:
                    prev_avg = None

                should_prompt = (cur_avg is None) and (prompted_week != week_id_now) and (snoozed_week != week_id_now)
                if should_prompt:
                    cs2 = dict(coach_state)
                    cs2['weekly_checkin_prompted_week_id'] = week_id_now
                    user.coach_state = cs2
                    user.coach_state_updated_at = timezone.now()
                    user.save(update_fields=['coach_state', 'coach_state_updated_at'])

                    suggested = None
                    try:
                        if getattr(user, 'weight', None):
                            suggested = round(float(user.weight), 1)
                    except Exception:
                        suggested = None

                    if not (getattr(user, 'sex', None) in ('male', 'female')):
                        quick_actions_out.append({
                            'label': 'Masculino',
                            'type': 'message',
                            'text': 'Masculino',
                            'payload': {'profile_updates': {'sex': 'male'}},
                        })
                        quick_actions_out.append({
                            'label': 'Femenino',
                            'type': 'message',
                            'text': 'Femenino',
                            'payload': {'profile_updates': {'sex': 'female'}},
                        })
                    if suggested is not None:
                        quick_actions_out.append({
                            'label': f'Registrar {suggested} kg',
                            'type': 'message',
                            'text': f'Registrar {suggested} kg',
                            'payload': {'weekly_weight_avg_kg': float(suggested)},
                        })
                    quick_actions_out.append({
                        'label': 'Luego',
                        'type': 'message',
                        'text': 'Luego',
                        'payload': {'weekly_checkin_snooze': True},
                    })

                    attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                        "[PERFIL METABÓLICO / CHECK-IN]\n"
                        "Necesito tu peso promedio semanal (kg) para recalibrar tu perfil metabólico.\n"
                        "Si falta sexo biológico, pedirlo con una sola pregunta."
                    )

                # Si ya tenemos dato semanal, calculamos
                cur_avg2 = _week_weights_from_state(weekly_state, week_id_now)
                if cur_avg2 is not None:
                    from .qaf_metabolic_profile.engine import evaluate_weekly_metabolic_profile, render_professional_summary

                    prof = {
                        'sex': getattr(user, 'sex', None),
                        'age': getattr(user, 'age', None),
                        'height_cm': _normalize_height_cm_from_user_value(getattr(user, 'height', None)),
                        'weight_kg': float(cur_avg2),
                        'goal_type': getattr(user, 'goal_type', None),
                        'activity_level': getattr(user, 'activity_level', None),
                        'daily_target_kcal_override': getattr(user, 'daily_target_kcal_override', None),
                    }

                    last_alpha = None
                    try:
                        last_alpha = (weekly_state.get('metabolic_last') or {}).get('adaptation_alpha')
                    except Exception:
                        last_alpha = None

                    weights_payload = {
                        'current_week': [float(cur_avg2)],
                        'previous_week': [float(prev_avg)] if (prev_avg is not None and prev_avg > 0) else [],
                    }

                    mr = evaluate_weekly_metabolic_profile(prof, weights_payload, last_alpha=last_alpha)
                    metabolic_result = mr.payload

                    # Persistir último cálculo para UX y continuidad
                    try:
                        ws2 = dict(weekly_state)
                        ws2['metabolic_last'] = {
                            'kcal_day': ((mr.payload.get('recommendation') or {}).get('kcal_day')),
                            'weekly_adjustment_kcal_day': ((mr.payload.get('recommendation') or {}).get('weekly_adjustment_kcal_day')),
                            'adaptation_alpha': ((mr.payload.get('metabolic') or {}).get('adaptation_alpha')),
                            'tdee_effective_kcal_day': ((mr.payload.get('metabolic') or {}).get('tdee_effective_kcal_day')),
                            'tdee_base_kcal_day': ((mr.payload.get('metabolic') or {}).get('tdee_base_kcal_day')),
                            'confidence': ((mr.payload.get('confidence') or {}).get('score')),
                            'updated_at': timezone.now().isoformat(),
                            'week_id': week_id_now,
                        }
                        user.coach_weekly_state = ws2
                        user.coach_weekly_updated_at = timezone.now()
                        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                        weekly_state = ws2
                    except Exception:
                        pass

                    mtext = render_professional_summary(mr.payload)
                    if mtext:
                        metabolic_text_for_output_override = mtext
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[PERFIL METABÓLICO DINÁMICO]\n{mtext}".strip()
        except Exception as ex:
            print(f"QAF metabolic profile warning: {ex}")


        # Construir prompt enriquecido (solo con texto real si se pudo)
        final_input = message or "Analisis de archivo adjunto"
        if (attachment_text or "").strip() and not _is_attachment_text_placeholder(attachment_text):
            final_input += f"\n\n--- DOCUMENTO ADJUNTO ---\n{attachment_text}\n-----------------------"

        professional_rule = (
            os.getenv("CHAT_PROFESSIONAL_RULE", "")
            .strip()
            or "Toma decisiones con el foco en ser lo más profesional posible: prudente, claro, verificable y seguro. No hagas diagnóstico médico. Si falta información, pide confirmación mínima antes de concluir."
        )
        if professional_rule:
            final_input = f"INSTRUCCION DEL SISTEMA: {professional_rule}\n\n{final_input}"

        fitness_payload = None
        profile_payload = None
        if_snapshot = None
        integrations_payload = None
        documents_payload = None

        username_for_payload = None
        if user:
            username_for_payload = user.username
        else:
            username_for_payload = (request.data.get('username') or '').strip() or None

        def _dt_iso(value):
            return value.isoformat() if value else None

        if user:
            try:
                # Últimos 5 IF (histórico)
                recent_records_qs = (
                    HappinessRecord.objects.filter(user=user)
                    .order_by("-date")[:5]
                )
                recent_records = [
                    {
                        "value": r.value,
                        "scores": r.scores,
                        "date": _dt_iso(r.date),
                    }
                    for r in recent_records_qs
                ]

                documents_qs = (
                    UserDocument.objects.filter(user=user)
                    .order_by("-updated_at")
                )
                max_doc_text = int(os.getenv("N8N_DOC_TEXT_MAX_CHARS", "5000") or 5000)
                documents_payload = [
                    {
                        "doc_type": d.doc_type,
                        "file_name": d.file_name,
                        "updated_at": _dt_iso(d.updated_at),
                        "extracted_text": _clean_extracted_text(d.extracted_text or "")[:max_doc_text],
                    }
                    for d in documents_qs
                ]
                documents_types = [d["doc_type"] for d in documents_payload]

                # IF snapshot (último registro + respuestas de la semana)
                latest_record = recent_records_qs[0] if recent_records_qs else None
                canonical_scores = latest_record.scores if latest_record else (user.scores or {})

                # Evitar duplicar el último IF en el histórico
                if latest_record and recent_records:
                    recent_records = recent_records[1:]

                profile_payload = {
                    "username": user.username,
                    "plan": user.plan,
                    "full_name": getattr(user, "full_name", None),
                    "age": user.age,
                    "weight": user.weight,
                    "height": user.height,
                    "goal_type": getattr(user, "goal_type", None),
                    "activity_level": getattr(user, "activity_level", None),
                    "daily_target_kcal_override": getattr(user, "daily_target_kcal_override", None),
                    "profession": getattr(user, "profession", None),
                    "favorite_exercise_time": getattr(user, "favorite_exercise_time", None),
                    "favorite_sport": getattr(user, "favorite_sport", None),
                    "age_range": _range_bucket(user.age, 5),
                    "weight_range": _range_bucket(user.weight, 5, lower=30, upper=200),
                    "height_range": _range_bucket(user.height, 5, lower=120, upper=230),
                    "happiness_index": user.happiness_index,
                    "scores_baseline": user.scores or {},
                    "current_streak": user.current_streak,
                    "badges": user.badges,
                    "if_history": recent_records,
                    "has_documents": bool(documents_payload),
                    "documents_count": len(documents_payload),
                    "documents_types": documents_types,
                }

                week_id = _week_id()
                answers_qs = (
                    IFAnswer.objects.filter(user=user, week_id=week_id)
                    .select_related("question")
                    .order_by("answered_at")
                )
                answers_payload = [
                    {
                        "question_id": a.question.key,
                        "question_label": a.question.label,
                        "value": a.value,
                        "slot": a.slot,
                        "answered_at": _dt_iso(a.answered_at),
                        "answered_date": _dt_iso(a.answered_date),
                        "source": a.source,
                    }
                    for a in answers_qs
                ]

                if_snapshot = {
                    "week_id": week_id,
                    "scores": canonical_scores,
                    "latest_record": {
                        "value": latest_record.value if latest_record else None,
                        "scores": latest_record.scores if latest_record else {},
                        "date": _dt_iso(latest_record.date) if latest_record else None,
                    },
                    "answers": answers_payload,
                    "answers_status": "empty" if not answers_payload else "ok",
                }

                # Integraciones / dispositivos
                device_qs = (
                    DeviceConnection.objects.filter(user=user)
                    .order_by("-updated_at")
                )
                devices_payload = [
                    {
                        "provider": d.provider,
                        "status": d.status,
                        "last_sync_at": _dt_iso(d.last_sync_at),
                        "updated_at": _dt_iso(d.updated_at),
                    }
                    for d in device_qs
                ]
                connected_providers = [
                    d["provider"] for d in devices_payload if d["status"] == "connected"
                ]

                # Último sync por proveedor + último sync global
                fitness_by_provider = {}
                latest_sync = None
                recent_syncs = (
                    DevicesFitnessSync.objects.filter(user=user)
                    .order_by("-created_at")[:50]
                )
                for sync in recent_syncs:
                    sync_payload = {
                        "provider": sync.provider,
                        "start_time": _dt_iso(sync.start_time),
                        "end_time": _dt_iso(sync.end_time),
                        "metrics": sync.metrics,
                        "created_at": _dt_iso(sync.created_at),
                    }
                    if latest_sync is None:
                        latest_sync = sync_payload
                    if sync.provider not in fitness_by_provider:
                        fitness_by_provider[sync.provider] = sync_payload

                integrations_payload = {
                    "devices": devices_payload,
                    "connected_providers": connected_providers,
                }

                fitness_payload = {
                    "providers": fitness_by_provider,
                    "latest": latest_sync,
                }
            except User.DoesNotExist:
                fitness_payload = None

        attachment_url_for_n8n = attachment_url
        # Solo omitimos el URL si hay texto real (evita perder el adjunto cuando hay solo diagnóstico).
        if (attachment_text or "").strip() and not _is_attachment_text_placeholder(attachment_text):
            attachment_url_for_n8n = ""

        payload = {
            "chatInput": final_input,
            "system_rules": {
                "professional_focus": professional_rule,
            },
            "message": message,
            "sessionId": session_id,
            "username": username_for_payload,
            "auth_header": auth_header,
            "attachment": attachment_url_for_n8n,
            "attachment_text": attachment_text,
            "attachment_text_diagnostic": attachment_text_diagnostic,
            "qaf": qaf_result,
            "qaf_context": {"vision": vision_parsed} if isinstance(vision_parsed, dict) else None,
            "qaf_metabolic": metabolic_result,
            "qaf_meal_plan": meal_plan_result,
            "qaf_body_trend": body_trend_result,
            "qaf_posture": posture_result,
            "qaf_motivation": motivation_result,
            "qaf_progression": progression_result,
            "fitness": fitness_payload,
            "profile": profile_payload,
            "if_snapshot": if_snapshot,
            "integrations": integrations_payload,
            "documents": documents_payload,
        }

        # Enriquecer reglas para n8n (mejor tono / framing) si se calculó motivación.
        try:
            if isinstance(motivation_result, dict):
                prof = motivation_result.get('profile') if isinstance(motivation_result.get('profile'), dict) else {}
                state = motivation_result.get('state') if isinstance(motivation_result.get('state'), dict) else {}
                tone = motivation_result.get('tone') if isinstance(motivation_result.get('tone'), dict) else {}
                chall = motivation_result.get('challenge') if isinstance(motivation_result.get('challenge'), dict) else {}
                payload.setdefault('system_rules', {})
                payload['system_rules']['motivation_context'] = {
                    'profile_top': prof.get('top'),
                    'mood': state.get('mood'),
                    'tone_style': tone.get('style'),
                    'challenge_label': chall.get('label'),
                    'decision': motivation_result.get('decision'),
                }
        except Exception:
            pass

        # 3. Enviar a n8n
        # Timeout corto por si n8n tarda
        response = requests.post(n8n_url, json=payload, timeout=60)
        
        def _extract_text_from_iframe(html: str) -> str:
            if not html:
                return ""
            lowered = html.lstrip().lower()
            if not lowered.startswith("<iframe"):
                return html
            try:
                import html as _html
                # Extraer srcdoc="..." o srcdoc='...'
                m = re.search(r"\bsrcdoc=(\"|')(.*?)(\1)", html, flags=re.IGNORECASE | re.DOTALL)
                if not m:
                    return html
                srcdoc = _html.unescape(m.group(2))
                # Quitar tags básicos (defensivo)
                srcdoc = re.sub(r"<\s*br\s*/?>", "\n", srcdoc, flags=re.IGNORECASE)
                srcdoc = re.sub(r"<[^>]+>", "", srcdoc)
                return srcdoc.strip() or html
            except Exception:
                return html

        # 4. Procesar respuesta
        if response.status_code == 200:
            try:
                # Si n8n devuelve JSON, lo pasamos directo
                data = response.json()
            except:
                # Si devuelve texto plano, lo envolvemos
                data = {'output': response.text}

            # Normalizar output si llega como iframe HTML (mejor compatibilidad mobile)
            try:
                if isinstance(data, dict) and isinstance(data.get('output'), str):
                    data['output'] = _extract_text_from_iframe(data['output'])
            except Exception:
                pass

            # Enriquecer respuesta hacia frontend (sin requerir cambios en n8n)
            try:
                if isinstance(data, dict) and qaf_result:
                    data.setdefault('qaf', qaf_result)
                    data.setdefault('follow_up_questions', qaf_result.get('follow_up_questions') or [])
                    data.setdefault('qaf_context', {"vision": vision_parsed} if isinstance(vision_parsed, dict) else None)

                    out_text = data.get('output')
                    if isinstance(out_text, str) and qaf_text_for_output_override:
                        low = out_text.lower()
                        has_kcal = ("kcal" in low) or ("calor" in low)
                        if (not out_text.strip()) or ("problema tecnico" in low) or ("problema técnico" in low):
                            data['output'] = qaf_text_for_output_override
                        elif not has_kcal:
                            data['output'] = (qaf_text_for_output_override.strip() + "\n\n" + out_text.strip()).strip()
            except Exception:
                pass

            # Exp-003: quick-actions + resultado metabólico
            try:
                if isinstance(data, dict):
                    if quick_actions_out:
                        data.setdefault('quick_actions', quick_actions_out)
                    if metabolic_result:
                        data.setdefault('qaf_metabolic', metabolic_result)

                    out_text = data.get('output')
                    if isinstance(out_text, str) and metabolic_text_for_output_override:
                        low = out_text.lower()
                        has_kcal = ("kcal" in low) or ("calor" in low)
                        if (not out_text.strip()) or ("problema tecnico" in low) or ("problema técnico" in low):
                            data['output'] = metabolic_text_for_output_override
                        elif not has_kcal:
                            data['output'] = (metabolic_text_for_output_override.strip() + "\n\n" + out_text.strip()).strip()
            except Exception:
                pass

            # Exp-006: postura (quick-actions + resultado)
            try:
                if isinstance(data, dict):
                    if posture_quick_actions_out:
                        existing = data.get('quick_actions') if isinstance(data.get('quick_actions'), list) else []
                        data['quick_actions'] = (existing + posture_quick_actions_out)[:6]
                    if posture_result:
                        data.setdefault('qaf_posture', posture_result)

                    out_text = data.get('output')
                    if isinstance(out_text, str) and posture_text_for_output_override:
                        low = out_text.lower()
                        has_posture = ('postura' in low) or ('hombro' in low) or ('cabeza' in low)
                        if (not out_text.strip()) or ('problema tecnico' in low) or ('problema técnico' in low):
                            data['output'] = posture_text_for_output_override
                        elif not has_posture:
                            data['output'] = (posture_text_for_output_override.strip() + "\n\n" + out_text.strip()).strip()
            except Exception:
                pass

            # Exp-007: lifestyle (quick-actions + resultado)
            try:
                if isinstance(data, dict):
                    if lifestyle_quick_actions_out:
                        existing = data.get('quick_actions') if isinstance(data.get('quick_actions'), list) else []
                        data['quick_actions'] = (existing + lifestyle_quick_actions_out)[:6]
                    if lifestyle_result:
                        data.setdefault('qaf_lifestyle', lifestyle_result)

                    out_text = data.get('output')
                    if isinstance(out_text, str) and lifestyle_text_for_output_override:
                        low = out_text.lower()
                        has_state = ('estado de hoy' in low) or ('dhss' in low) or ('micro' in low)
                        if (not out_text.strip()) or ('problema tecnico' in low) or ('problema técnico' in low):
                            data['output'] = lifestyle_text_for_output_override
                        elif not has_state:
                            data['output'] = (lifestyle_text_for_output_override.strip() + "\n\n" + out_text.strip()).strip()
            except Exception:
                pass

            # Exp-008: motivación (quick-actions + resultado)
            try:
                if isinstance(data, dict):
                    if motivation_quick_actions_out:
                        existing = data.get('quick_actions') if isinstance(data.get('quick_actions'), list) else []
                        data['quick_actions'] = (existing + motivation_quick_actions_out)[:6]
                    if motivation_result:
                        data.setdefault('qaf_motivation', motivation_result)

                    out_text = data.get('output')
                    if isinstance(out_text, str) and motivation_text_for_output_override:
                        low = out_text.lower()
                        has_mot = ('motiv' in low) or ('reto:' in low) or ('perfil dominante' in low)
                        if (not out_text.strip()) or ('problema tecnico' in low) or ('problema técnico' in low):
                            data['output'] = motivation_text_for_output_override
                        elif not has_mot:
                            data['output'] = (motivation_text_for_output_override.strip() + "\n\n" + out_text.strip()).strip()
            except Exception:
                pass

            # Exp-009: progresión (quick-actions + resultado)
            try:
                if isinstance(data, dict):
                    if progression_quick_actions_out:
                        existing = data.get('quick_actions') if isinstance(data.get('quick_actions'), list) else []
                        data['quick_actions'] = (existing + progression_quick_actions_out)[:6]
                    if progression_result:
                        data.setdefault('qaf_progression', progression_result)

                    out_text = data.get('output')
                    if isinstance(out_text, str) and progression_text_for_output_override:
                        low = out_text.lower()
                        has_prog = ('readiness' in low) or ('micro-objetivo' in low) or ('progres' in low)
                        if (not out_text.strip()) or ('problema tecnico' in low) or ('problema técnico' in low):
                            data['output'] = progression_text_for_output_override
                        elif not has_prog:
                            data['output'] = (progression_text_for_output_override.strip() + "\n\n" + out_text.strip()).strip()
            except Exception:
                pass

            # Exp-004: resultado de menú + posible override de texto
            try:
                if isinstance(data, dict):
                    if meal_plan_result:
                        data.setdefault('qaf_meal_plan', meal_plan_result)

                    out_text = data.get('output')
                    if isinstance(out_text, str) and meal_plan_text_for_output_override:
                        low = out_text.lower()
                        has_menu = ("menú" in low) or ("menu" in low) or ("plan de comidas" in low)
                        if (not out_text.strip()) or ("problema tecnico" in low) or ("problema técnico" in low):
                            data['output'] = meal_plan_text_for_output_override
                        elif not has_menu:
                            data['output'] = (meal_plan_text_for_output_override.strip() + "\n\n" + out_text.strip()).strip()
            except Exception:
                pass

            # Exp-005: tendencia corporal + posible override de texto
            try:
                if isinstance(data, dict):
                    if body_trend_result:
                        data.setdefault('qaf_body_trend', body_trend_result)

                    out_text = data.get('output')
                    if isinstance(out_text, str) and body_trend_text_for_output_override:
                        low = out_text.lower()
                        has_trend = ("proye" in low) or ("tendenc" in low) or ("6 semanas" in low)
                        if (not out_text.strip()) or ("problema tecnico" in low) or ("problema técnico" in low):
                            data['output'] = body_trend_text_for_output_override
                        elif not has_trend:
                            data['output'] = (body_trend_text_for_output_override.strip() + "\n\n" + out_text.strip()).strip()
            except Exception:
                pass
            return Response(data)
        else:
            # Intentar obtener mensaje de error de n8n
            try:
                err_body = response.json()
                err_msg = err_body.get('message') or response.text
            except:
                err_msg = response.text
                
            print(f"Error n8n body: {err_msg}")
            return Response({'error': f"n8n Error ({response.status_code}): {err_msg}"}, status=502)

    except Exception as e:
        print(f"Error chat_n8n: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_meal_coherence(request):
    """Evalúa coherencia de una comida vs meta (Exp-002).

    Entrada (JSON):
      { "meal": { total_calories, uncertainty_score, needs_confirmation, meal_slot }, "goal_text"?: str }

    Usa el usuario autenticado como fuente de `goal_type`, `activity_level` y `weight`.
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    meal = payload.get('meal') if isinstance(payload.get('meal'), dict) else {}
    goal_text = payload.get('goal_text') if isinstance(payload.get('goal_text'), str) else None

    from .qaf_goal_coherence.engine import evaluate_meal

    user_ctx = {
        "weight_kg": getattr(user, "weight", None),
        "age": getattr(user, "age", None),
        "height_cm": (float(getattr(user, "height", None)) * 100.0) if getattr(user, "height", None) else None,
        "goal_type": getattr(user, "goal_type", None),
        "goal_text": goal_text,
        "activity_level": getattr(user, "activity_level", None),
        "daily_target_kcal_override": getattr(user, "daily_target_kcal_override", None),
    }

    result = evaluate_meal(user_ctx, meal)
    return Response(result)


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_metabolic_profile(request):
    """Calcula y opcionalmente persiste el perfil metabólico semanal (Exp-003).

    Entrada (JSON):
      { "weekly_weight_avg_kg"?: number, "persist"?: bool }

    Usa el usuario autenticado como fuente de sexo/edad/altura/meta/actividad.
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    persist = str(payload.get('persist') or '').strip().lower() in ('1', 'true', 'yes', 'on')

    week_id_now = _week_id()
    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

    avg_w = payload.get('weekly_weight_avg_kg')
    if avg_w is not None:
        try:
            avg_w = float(avg_w)
        except Exception:
            avg_w = None

    if avg_w and avg_w > 0 and persist:
        ww = weekly_state.get('weekly_weights') if isinstance(weekly_state.get('weekly_weights'), dict) else {}
        ww2 = dict(ww)
        ww2[week_id_now] = {
            'avg_weight_kg': float(avg_w),
            'source': 'api',
            'recorded_at': timezone.now().isoformat(),
        }
        weekly_state = dict(weekly_state)
        weekly_state['weekly_weights'] = ww2
        user.coach_weekly_state = weekly_state
        user.coach_weekly_updated_at = timezone.now()
        try:
            user.weight = float(avg_w)
        except Exception:
            pass
        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at', 'weight'])

    cur = None
    try:
        row = (weekly_state.get('weekly_weights') or {}).get(week_id_now)
        cur = float(row.get('avg_weight_kg')) if isinstance(row, dict) else float(row)
    except Exception:
        cur = None

    prev = None
    try:
        ww = weekly_state.get('weekly_weights') if isinstance(weekly_state.get('weekly_weights'), dict) else {}
        prev_keys = [k for k in ww.keys() if isinstance(k, str) and k != week_id_now]
        if prev_keys:
            prev_key = sorted(prev_keys)[-1]
            row = ww.get(prev_key)
            prev = float(row.get('avg_weight_kg')) if isinstance(row, dict) else float(row)
    except Exception:
        prev = None

    from .qaf_metabolic_profile.engine import evaluate_weekly_metabolic_profile

    prof = {
        'sex': getattr(user, 'sex', None),
        'age': getattr(user, 'age', None),
        'height_cm': _normalize_height_cm_from_user_value(getattr(user, 'height', None)),
        'weight_kg': float(cur or getattr(user, 'weight', 0) or 0),
        'goal_type': getattr(user, 'goal_type', None),
        'activity_level': getattr(user, 'activity_level', None),
        'daily_target_kcal_override': getattr(user, 'daily_target_kcal_override', None),
    }

    last_alpha = None
    try:
        last_alpha = (weekly_state.get('metabolic_last') or {}).get('adaptation_alpha')
    except Exception:
        last_alpha = None

    weights_payload = {
        'current_week': [float(cur)] if (cur is not None and cur > 0) else [],
        'previous_week': [float(prev)] if (prev is not None and prev > 0) else [],
    }

    res = evaluate_weekly_metabolic_profile(prof, weights_payload, last_alpha=last_alpha)

    if persist:
        try:
            ws2 = dict(weekly_state)
            ws2['metabolic_last'] = {
                'kcal_day': ((res.payload.get('recommendation') or {}).get('kcal_day')),
                'weekly_adjustment_kcal_day': ((res.payload.get('recommendation') or {}).get('weekly_adjustment_kcal_day')),
                'adaptation_alpha': ((res.payload.get('metabolic') or {}).get('adaptation_alpha')),
                'tdee_effective_kcal_day': ((res.payload.get('metabolic') or {}).get('tdee_effective_kcal_day')),
                'tdee_base_kcal_day': ((res.payload.get('metabolic') or {}).get('tdee_base_kcal_day')),
                'confidence': ((res.payload.get('confidence') or {}).get('score')),
                'updated_at': timezone.now().isoformat(),
                'week_id': week_id_now,
            }
            user.coach_weekly_state = ws2
            user.coach_weekly_updated_at = timezone.now()
            user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
        except Exception:
            pass

    return Response({'success': True, 'result': res.payload})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_meal_plan(request):
    """Genera y opcionalmente persiste un menú semanal (Exp-004).

    Entrada (JSON):
      {
        "kcal_day"?: number,
        "meals_per_day"?: 3|4,
        "variety"?: "simple"|"normal"|"high",
        "exclude_item_ids"?: ["..."]
        "persist"?: bool
      }

    Por defecto persiste en `user.coach_weekly_state.meal_plan[week_id]`.
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    persist = str(payload.get('persist') or '').strip().lower() in ('1', 'true', 'yes', 'on')
    if payload.get('persist') is None:
        persist = True

    locale = (payload.get('locale') or '').strip() or 'es-CO'

    try:
        kcal_day = float(payload.get('kcal_day')) if payload.get('kcal_day') is not None else None
    except Exception:
        kcal_day = None

    try:
        meals_per_day = int(payload.get('meals_per_day')) if payload.get('meals_per_day') is not None else None
    except Exception:
        meals_per_day = None

    variety = str(payload.get('variety') or '').strip().lower() or 'normal'
    if variety not in ('simple', 'normal', 'high'):
        variety = 'normal'

    exclude_item_ids = payload.get('exclude_item_ids') if isinstance(payload.get('exclude_item_ids'), list) else []

    # kcal target: override > metabolic_last > fallback
    if kcal_day is None:
        if getattr(user, 'daily_target_kcal_override', None):
            try:
                kcal_day = float(user.daily_target_kcal_override)
            except Exception:
                kcal_day = None

    if kcal_day is None:
        try:
            weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
            kcal_day = float(((weekly_state.get('metabolic_last') or {}).get('kcal_day')) or 0.0) or None
        except Exception:
            kcal_day = None

    if kcal_day is None:
        kcal_day = 2000.0

    if meals_per_day not in (3, 4):
        meals_per_day = 3

    week_id_now = _week_id()
    exclude_norm = [str(x).strip() for x in exclude_item_ids if str(x).strip()]
    exclude_norm = exclude_norm[:50]
    exclude_sig = ",".join(sorted(exclude_norm))
    signature = f"v0|{week_id_now}|{variety}|{int(round(float(kcal_day)))}|{int(meals_per_day)}|{exclude_sig}"
    seed = (hash(f"{user.id}:{signature}") & 0xFFFFFFFF)

    from .qaf_meal_planner.engine import (
        build_quick_actions_for_menu,
        generate_week_plan,
        render_professional_summary,
    )

    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
    # Cache en coach_weekly_state para UX rápida (no depende de memoria del proceso)
    cached = None
    try:
        variants = weekly_state.get('meal_plan_variants') if isinstance(weekly_state.get('meal_plan_variants'), dict) else {}
        wk = variants.get(week_id_now) if isinstance(variants.get(week_id_now), dict) else {}
        cached = wk.get(signature)
    except Exception:
        cached = None

    if isinstance(cached, dict) and cached.get('result') and isinstance(cached.get('result'), dict):
        result = cached.get('result')
    else:
        result = generate_week_plan(
            kcal_day=float(kcal_day),
            meals_per_day=int(meals_per_day),
            variety_level=variety,
            exclude_item_ids=exclude_norm,
            seed=int(seed),
            locale=locale,
        )

    quick_actions = build_quick_actions_for_menu(variety_level=variety)
    text = render_professional_summary(result)

    if persist:
        try:
            ws2 = dict(weekly_state)
            mp = ws2.get('meal_plan') if isinstance(ws2.get('meal_plan'), dict) else {}
            mp2 = dict(mp)
            mp2[week_id_now] = {
                'result': result,
                'updated_at': timezone.now().isoformat(),
                'week_id': week_id_now,
                'signature': signature,
            }
            ws2['meal_plan'] = mp2

            variants = ws2.get('meal_plan_variants') if isinstance(ws2.get('meal_plan_variants'), dict) else {}
            wk = variants.get(week_id_now) if isinstance(variants.get(week_id_now), dict) else {}
            wk2 = dict(wk)
            wk2[signature] = {
                'result': result,
                'updated_at': timezone.now().isoformat(),
                'signature': signature,
            }
            # Limitar a 3 variantes para no inflar JSON
            if len(wk2) > 3:
                keys = sorted(wk2.keys())
                for k in keys[:-3]:
                    wk2.pop(k, None)
            variants2 = dict(variants)
            variants2[week_id_now] = wk2
            ws2['meal_plan_variants'] = variants2

            user.coach_weekly_state = ws2
            user.coach_weekly_updated_at = timezone.now()
            user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
        except Exception:
            pass

    return Response({'success': True, 'result': result, 'text': text, 'quick_actions': quick_actions})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_meal_plan_apply(request):
    """Marca el menú semanal como 'activo' (Exp-004) para evitar cambios sin confirmación."""

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    week_id = str(payload.get('week_id') or _week_id())

    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
    ws2 = dict(weekly_state)
    ws2['meal_plan_active_week_id'] = week_id
    ws2['meal_plan_active_at'] = timezone.now().isoformat()
    user.coach_weekly_state = ws2
    user.coach_weekly_updated_at = timezone.now()
    user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
    return Response({'success': True, 'week_id': week_id})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_meal_plan_mutate(request):
    """Mutación local del menú: cambia un slot sin regenerar toda la semana."""

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    week_id = str(payload.get('week_id') or _week_id())
    day_index = payload.get('day_index')
    slot = str(payload.get('slot') or '').strip().lower()
    direction = str(payload.get('direction') or 'normal').strip().lower()
    if direction not in ('simple', 'normal', 'high'):
        direction = 'normal'

    try:
        day_index_i = int(day_index)
    except Exception:
        return Response({'error': 'day_index requerido'}, status=status.HTTP_400_BAD_REQUEST)

    if slot not in ('desayuno', 'almuerzo', 'cena', 'snack'):
        return Response({'error': 'slot inválido'}, status=status.HTTP_400_BAD_REQUEST)

    locale = (payload.get('locale') or '').strip() or 'es-CO'
    exclude_item_ids = payload.get('exclude_item_ids') if isinstance(payload.get('exclude_item_ids'), list) else []
    if len(exclude_item_ids) > 50:
        exclude_item_ids = exclude_item_ids[:50]

    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
    mp = weekly_state.get('meal_plan') if isinstance(weekly_state.get('meal_plan'), dict) else {}
    stored = mp.get(week_id)
    if not stored:
        return Response({'error': 'No hay menú guardado para esta semana'}, status=status.HTTP_404_NOT_FOUND)

    # Compat: stored puede ser {result: ...} o el result directo
    if isinstance(stored, dict) and 'result' in stored and isinstance(stored.get('result'), dict):
        current_result = stored.get('result')
    elif isinstance(stored, dict) and 'plan' in stored:
        current_result = stored
    else:
        return Response({'error': 'Menú guardado inválido'}, status=status.HTTP_400_BAD_REQUEST)

    seed = (hash(f"{user.id}:{week_id}:{direction}:{day_index_i}:{slot}") & 0xFFFFFFFF)

    from .qaf_meal_planner.engine import mutate_plan_slot, render_professional_summary, build_quick_actions_for_menu

    mutated = mutate_plan_slot(
        result=current_result,
        day_index=day_index_i,
        slot=slot,
        direction=direction,
        seed=int(seed),
        locale=locale,
        exclude_item_ids=[str(x) for x in exclude_item_ids if str(x).strip()],
    )

    # Persistir
    try:
        ws2 = dict(weekly_state)
        mp2 = dict(mp)
        mp2[week_id] = {
            'result': mutated,
            'updated_at': timezone.now().isoformat(),
            'week_id': week_id,
        }
        ws2['meal_plan'] = mp2
        user.coach_weekly_state = ws2
        user.coach_weekly_updated_at = timezone.now()
        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
    except Exception:
        pass

    text = render_professional_summary(mutated)
    quick_actions = build_quick_actions_for_menu(variety_level=str(((mutated.get('inputs') or {}).get('variety')) or 'normal'))
    return Response({'success': True, 'result': mutated, 'text': text, 'quick_actions': quick_actions})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_body_trend(request):
    """Proyección de tendencia corporal a 6 semanas (Exp-005)."""

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    week_id_now = _week_id()
    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

    # pesos semanales
    cur_w = None
    prev_w = None
    try:
        ww = weekly_state.get('weekly_weights') if isinstance(weekly_state.get('weekly_weights'), dict) else {}
        row = ww.get(week_id_now)
        cur_w = float(row.get('avg_weight_kg')) if isinstance(row, dict) else float(row)
        prev_keys = [k for k in ww.keys() if isinstance(k, str) and k != week_id_now]
        if prev_keys:
            prev_key = sorted(prev_keys)[-1]
            row2 = ww.get(prev_key)
            prev_w = float(row2.get('avg_weight_kg')) if isinstance(row2, dict) else float(row2)
    except Exception:
        cur_w = getattr(user, 'weight', None)
        prev_w = None

    # kcal ingreso promedio: payload > cache semanal
    kcal_in_avg = None
    if payload.get('kcal_in_avg_day') is not None:
        try:
            kcal_in_avg = float(payload.get('kcal_in_avg_day'))
        except Exception:
            kcal_in_avg = None

    try:
        kcal_week = weekly_state.get('kcal_avg_by_week') if isinstance(weekly_state.get('kcal_avg_by_week'), dict) else {}
        if kcal_in_avg is None and week_id_now in kcal_week:
            row = kcal_week.get(week_id_now)
            kcal_in_avg = float(row.get('kcal_in_avg_day')) if isinstance(row, dict) else float(row)
    except Exception:
        pass

    # recommended y tdee desde metabolic_last
    tdee = None
    reco = None
    try:
        ml = weekly_state.get('metabolic_last') if isinstance(weekly_state.get('metabolic_last'), dict) else {}
        tdee = float(ml.get('tdee_effective_kcal_day')) if ml.get('tdee_effective_kcal_day') is not None else None
        reco = float(ml.get('kcal_day')) if ml.get('kcal_day') is not None else None
    except Exception:
        tdee = None
        reco = None

    # Guardrail: descartar valores absurdos (p.ej. bug legacy de unidades)
    try:
        if tdee is not None and float(tdee) > 10000.0:
            tdee = None
    except Exception:
        tdee = None
    try:
        if reco is not None and float(reco) > 10000.0:
            reco = None
    except Exception:
        reco = None

    # fallback tdee: aproximación por activity (si no hay metabolic_last)
    if tdee is None:
        try:
            from .qaf_metabolic_profile.engine import evaluate_weekly_metabolic_profile

            prof = {
                'sex': getattr(user, 'sex', None),
                'age': getattr(user, 'age', None),
                'height_cm': _normalize_height_cm_from_user_value(getattr(user, 'height', None)),
                'weight_kg': float(cur_w or getattr(user, 'weight', 0) or 0),
                'goal_type': getattr(user, 'goal_type', None),
                'activity_level': getattr(user, 'activity_level', None),
                'daily_target_kcal_override': getattr(user, 'daily_target_kcal_override', None),
            }
            weights_payload = {
                'current_week': [float(cur_w)] if (cur_w is not None and float(cur_w) > 0) else [],
                'previous_week': [float(prev_w)] if (prev_w is not None and float(prev_w) > 0) else [],
            }
            mr = evaluate_weekly_metabolic_profile(prof, weights_payload, last_alpha=None)
            tdee = float(((mr.payload.get('metabolic') or {}).get('tdee_effective_kcal_day')) or 0.0) or None
            if reco is None:
                reco = (mr.payload.get('recommendation') or {}).get('kcal_day')
        except Exception:
            tdee = float(reco) if reco is not None else None

    profile = {
        'tdee_kcal_day': tdee,
        'recommended_kcal_day': (float(getattr(user, 'daily_target_kcal_override', None)) if getattr(user, 'daily_target_kcal_override', None) else reco),
    }
    observations = {
        'weight_current_week_avg_kg': cur_w,
        'weight_previous_week_avg_kg': prev_w,
        'kcal_in_avg_day': kcal_in_avg,
    }

    from .qaf_body_trend.engine import evaluate_body_trend, render_professional_summary, build_quick_actions_for_trend

    res = evaluate_body_trend(profile, observations, horizon_weeks=6)
    text = render_professional_summary(res.payload)
    quick_actions = build_quick_actions_for_trend(has_intake=(kcal_in_avg is not None))

    # persistir kcal avg si viene y es válido
    if kcal_in_avg is not None and kcal_in_avg > 0:
        try:
            ws2 = dict(weekly_state)
            kbw = ws2.get('kcal_avg_by_week') if isinstance(ws2.get('kcal_avg_by_week'), dict) else {}
            kbw2 = dict(kbw)
            kbw2[week_id_now] = {
                'kcal_in_avg_day': float(kcal_in_avg),
                'updated_at': timezone.now().isoformat(),
            }
            ws2['kcal_avg_by_week'] = kbw2
            ws2['body_trend_last'] = {
                'result': res.payload,
                'updated_at': timezone.now().isoformat(),
            }
            user.coach_weekly_state = ws2
            user.coach_weekly_updated_at = timezone.now()
            user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
        except Exception:
            pass

    return Response({'success': True, 'result': res.payload, 'text': text, 'quick_actions': quick_actions})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_posture(request):
    """Exp-006: análisis postural basado en keypoints (pose estimation) + rutina correctiva.

    Importante: este endpoint NO hace pose estimation. Recibe `poses.*.keypoints` ya calculados.
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    user_ctx = payload.get('user_context') if isinstance(payload.get('user_context'), dict) else {}
    locale = (payload.get('locale') or '').strip() or 'es-CO'

    from .qaf_posture.engine import evaluate_posture, render_professional_summary

    res = evaluate_posture({
        'poses': payload.get('poses') if isinstance(payload.get('poses'), dict) else {},
        'user_context': user_ctx,
        'locale': locale,
    }).payload

    text = render_professional_summary(res)
    return Response({'success': True, 'result': res, 'text': text})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_lifestyle(request):
    """Exp-007: Lifestyle Intelligence Engine (DHSS + patrones + micro-hábitos)."""

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    days = payload.get('days')
    try:
        days_i = int(days) if days is not None else 14
    except Exception:
        days_i = 14
    days_i = max(3, min(30, days_i))

    self_report = payload.get('self_report') if isinstance(payload.get('self_report'), dict) else {}

    # Construir daily_metrics desde FitnessSync (último por día)
    try:
        from datetime import timedelta
        from django.utils import timezone

        start_dt = timezone.now() - timedelta(days=days_i)
        qs = DevicesFitnessSync.objects.filter(user=user, created_at__gte=start_dt).only('created_at', 'metrics').order_by('created_at')
        by_day = {}
        for s in qs:
            d = timezone.localdate(s.created_at).isoformat()
            by_day[d] = (s.metrics or {})

        daily_metrics = []
        for d, m in sorted(by_day.items()):
            row = {'date': d}
            if isinstance(m, dict):
                for k in ('steps', 'sleep_minutes', 'calories', 'resting_heart_rate_bpm', 'avg_heart_rate_bpm', 'distance_m', 'distance_km'):
                    if k in m:
                        row[k] = m.get(k)
            daily_metrics.append(row)
    except Exception:
        daily_metrics = []

    # memory desde coach_state
    cs = getattr(user, 'coach_state', {}) or {}
    mem = cs.get('lifestyle_memory') if isinstance(cs.get('lifestyle_memory'), dict) else {}

    from .qaf_lifestyle.engine import evaluate_lifestyle, render_professional_summary
    res = evaluate_lifestyle({'daily_metrics': daily_metrics, 'self_report': self_report, 'memory': mem}).payload
    text = render_professional_summary(res)

    # Persistir memoria mínima (last_ids)
    try:
        micro = res.get('microhabits') if isinstance(res.get('microhabits'), list) else []
        ids = [str(x.get('id')) for x in micro if isinstance(x, dict) and x.get('id')]
        mem2 = dict(mem)
        mem2['last_ids'] = ids[:3]
        cs2 = dict(cs)
        cs2['lifestyle_memory'] = mem2
        cs2['lifestyle_last'] = {
            'result': res,
            'updated_at': timezone.now().isoformat(),
        }
        user.coach_state = cs2
        user.coach_state_updated_at = timezone.now()
        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
    except Exception:
        pass

    return Response({'success': True, 'result': res, 'text': text, 'daily_metrics': daily_metrics[-7:]})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_motivation(request):
    """Exp-008: Motivación psicológica personalizada (perfil vectorial + estado + reto)."""

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    message = str(payload.get('message') or payload.get('text') or '').strip() or 'Necesito motivación.'

    cs = getattr(user, 'coach_state', {}) or {}
    mem = cs.get('motivation_memory') if isinstance(cs.get('motivation_memory'), dict) else {}
    prefs = cs.get('motivation_preferences') if isinstance(cs.get('motivation_preferences'), dict) else {}
    if isinstance(payload.get('preferences'), dict):
        prefs = {**prefs, **payload.get('preferences')}

    gam = {
        'streak': int(getattr(user, 'current_streak', 0) or 0),
        'badges': getattr(user, 'badges', []) or [],
    }

    lifestyle = None
    try:
        lifestyle = (cs.get('lifestyle_last') or {}).get('result')
    except Exception:
        lifestyle = None

    from .qaf_motivation.engine import evaluate_motivation, render_professional_summary
    res = evaluate_motivation({'message': message, 'memory': mem, 'preferences': prefs, 'gamification': gam, 'lifestyle': lifestyle or {}}).payload
    text = render_professional_summary(res)

    try:
        cs2 = dict(cs)
        vec = ((res.get('profile') or {}).get('vector')) if isinstance(res.get('profile'), dict) else None
        mem2 = dict(mem)
        if isinstance(vec, dict):
            mem2['vector'] = vec
        mem2['last_seen_at'] = timezone.now().isoformat()
        cs2['motivation_memory'] = mem2
        cs2['motivation_preferences'] = prefs
        cs2['motivation_last'] = {'result': res, 'updated_at': timezone.now().isoformat()}
        user.coach_state = cs2
        user.coach_state_updated_at = timezone.now()
        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
    except Exception:
        pass

    return Response({'success': True, 'result': res, 'text': text})


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_progression(request):
    """Exp-009: Progresión inteligente (fuerza + cardio).

    Entrada: payload con `session`, y opcional `strength`/`cardio`.
    Si no vienen, intenta parsear desde `message`.
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    message = str(payload.get('message') or payload.get('text') or '').strip()

    from .qaf_progression.engine import evaluate_progression, render_professional_summary, parse_strength_line

    session = payload.get('session') if isinstance(payload.get('session'), dict) else {}
    strength = payload.get('strength') if isinstance(payload.get('strength'), dict) else None
    cardio = payload.get('cardio') if isinstance(payload.get('cardio'), dict) else None

    # Parse rápido si no hay estructura
    if not strength and message:
        strength = parse_strength_line(message)

    # Señales del día: usar último lifestyle + mood
    cs = getattr(user, 'coach_state', {}) or {}
    lifestyle_last = (cs.get('lifestyle_last') or {}).get('result') if isinstance(cs.get('lifestyle_last'), dict) else None
    mood = None
    try:
        mood = ((cs.get('motivation_last') or {}).get('result') or {}).get('state', {}).get('mood')
    except Exception:
        mood = None

    signals = payload.get('signals') if isinstance(payload.get('signals'), dict) else {}
    # Atajo: completar señales desde lifestyle si faltan
    try:
        if isinstance(lifestyle_last, dict):
            sig = lifestyle_last.get('signals') if isinstance(lifestyle_last.get('signals'), dict) else {}
            if signals.get('sleep_minutes') is None:
                signals['sleep_minutes'] = (sig.get('sleep') or {}).get('value')
            if signals.get('steps') is None:
                signals['steps'] = (sig.get('steps') or {}).get('value')
            if signals.get('resting_heart_rate_bpm') is None:
                signals['resting_heart_rate_bpm'] = (sig.get('stress_inv') or {}).get('value')
            if signals.get('lifestyle_band') is None:
                signals['lifestyle_band'] = (lifestyle_last.get('dhss') or {}).get('band')
    except Exception:
        pass
    if signals.get('mood') is None and mood:
        signals['mood'] = mood

    # History
    mem = cs.get('progression_history') if isinstance(cs.get('progression_history'), dict) else {}

    res = evaluate_progression({'session': session, 'strength': strength, 'cardio': cardio, 'history': mem, 'signals': signals}).payload
    text = render_professional_summary(res)

    # Persistir history mínimo
    try:
        mem2 = dict(mem)
        if isinstance(strength, dict) and strength.get('name'):
            name = str(strength.get('name') or '').strip().lower()
            sets = int(strength.get('sets') or 0)
            reps = int(strength.get('reps') or 0)
            load = strength.get('load_kg')
            try:
                load_f = float(load) if load is not None else None
            except Exception:
                load_f = None
            ton = None
            est1 = None
            if load_f is not None and load_f > 0 and sets > 0 and reps > 0:
                ton = float(sets) * float(reps) * float(load_f)
                est1 = float(load_f) * (1.0 + (float(reps) / 30.0))
            key = f"strength:{name}"
            rows = mem2.get(key) if isinstance(mem2.get(key), list) else []
            rows2 = [r for r in rows if isinstance(r, dict)][-2:]
            rows2.append({'date': timezone.localdate().isoformat(), 'sets': sets, 'reps': reps, 'load_kg': load_f, 'tonnage': ton, 'est_1rm': est1, 'rpe': session.get('rpe_1_10')})
            mem2[key] = rows2[-3:]
        if isinstance(cardio, dict) and cardio.get('minutes') is not None:
            rows = mem2.get('cardio:default') if isinstance(mem2.get('cardio:default'), list) else []
            rows2 = [r for r in rows if isinstance(r, dict)][-2:]
            rows2.append({'date': timezone.localdate().isoformat(), 'minutes': cardio.get('minutes'), 'avg_hr': cardio.get('avg_hr'), 'rpe': session.get('rpe_1_10')})
            mem2['cardio:default'] = rows2[-3:]
        cs2 = dict(cs)
        cs2['progression_history'] = mem2
        cs2['progression_last'] = {'result': res, 'updated_at': timezone.now().isoformat()}
        user.coach_state = cs2
        user.coach_state_updated_at = timezone.now()
        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
    except Exception:
        pass

    return Response({'success': True, 'result': res, 'text': text})


