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
import time
from typing import Any

from devices.models import DeviceConnection, FitnessSync as DevicesFitnessSync
from devices.scheduler_service import enqueue_sync_request
from devices.sync_service import sync_device

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
from .gamification_service import update_user_streak, build_gamification_status, claim_daily_wow_reward, award_wow_event
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
from urllib.parse import quote, unquote, urlparse, urlencode
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


def _bench_event_get(request):
    """Obtiene el dict de benchmark creado por `BenchmarkChatMiddleware` (si está activo)."""
    try:
        django_req = getattr(request, "_request", request)
        return getattr(django_req, "_bench_event", None)
    except Exception:
        return None


def _bench_event_note(request, **kwargs):
    """Actualiza campos del evento benchmark de forma defensiva (no rompe la request)."""
    bench = _bench_event_get(request)
    if not isinstance(bench, dict):
        return
    try:
        for k, v in kwargs.items():
            if v is not None:
                bench[k] = v
    except Exception:
        pass


def _infer_chat_flow(message: str, data: Any) -> str:
    """Clasifica el request para métricas comparativas por servicio.

    Valores esperados para benchmark: 011/012/013/cognition/protocol/other.
    Aquí usamos nombres estables en forma `exp-XXX_*` para poder segmentar.
    """
    try:
        if isinstance(data, dict):
            if data.get("skin_habits_request") is True or data.get("skin_cancel") is True:
                return "exp-011_skin_health"
            if data.get("couture_garments_request") is True:
                return "exp-012_shape_presence"
            # Arquitectura Corporal (Postura & Proporción)
            if data.get("pp_cancel") is True or data.get("pp_start") is True:
                return "exp-013_body_architecture"
            if data.get("qaf_cognition") is True:
                return "cognition"
            if data.get("protocol_cognitive_body") is True:
                return "protocol"
    except Exception:
        pass

    msg = (str(message or "").strip().lower())
    if not msg:
        return "other"
    if ("vitalidad" in msg and ("piel" in msg or "peil" in msg)) or ("skin health" in msg) or ("skincare" in msg):
        return "exp-011_skin_health"
    if ("alta costura" in msg) or ("couture" in msg):
        return "exp-012_shape_presence"
    if ("arquitectura corporal" in msg) or ("postura" in msg and "propor" in msg):
        return "exp-013_body_architecture"
    if "cognici" in msg:
        return "cognition"
    if "protocolo" in msg and "corporal" in msg:
        return "protocol"
    return "other"


def _bench_n8n_post(request, url: str, payload: dict, timeout: int):
    """Wrapper para medir n8n y marcar `n8n_called` + tokens si vienen en la respuesta."""
    _bench_event_note(request, n8n_called=True)
    start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=timeout)
    ms = int(round((time.perf_counter() - start) * 1000.0))

    # Acumular latencia n8n (por si hay más de 1 llamada)
    bench = _bench_event_get(request)
    if isinstance(bench, dict):
        try:
            prev = int(bench.get("latency_ms_n8n") or 0)
        except Exception:
            prev = 0
        bench["latency_ms_n8n"] = prev + ms

        # Tokens opcionales (si n8n devuelve usage)
        try:
            data = None
            if getattr(resp, "headers", None) and (resp.headers.get("Content-Type") or "").lower().startswith("application/json"):
                data = resp.json()
            if isinstance(data, dict):
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
                if isinstance(usage, dict):
                    bench["tokens_in"] = usage.get("prompt_tokens") or bench.get("tokens_in")
                    bench["tokens_out"] = usage.get("completion_tokens") or bench.get("tokens_out")
                # Alternativos
                if bench.get("tokens_in") is None and data.get("tokens_in") is not None:
                    bench["tokens_in"] = data.get("tokens_in")
                if bench.get("tokens_out") is None and data.get("tokens_out") is not None:
                    bench["tokens_out"] = data.get("tokens_out")
        except Exception:
            pass

    return resp


def _hito5_env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")


def _hito5_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        val = float(raw)
    except Exception:
        return float(default)
    if val < 0.0:
        return 0.0
    if val > 100.0:
        return 100.0
    return val


def _hito5_release_context(user, session_id: str) -> dict:
    enabled = _hito5_env_bool("HITO5_RELEASE_ENABLED", False)
    rollback_force = _hito5_env_bool("HITO5_ROLLBACK_FORCE", False)
    canary_percent = _hito5_env_float("HITO5_CANARY_PERCENT", 10.0)
    salt = os.getenv("HITO5_CANARY_SALT", "gotogym-hito5")

    if not enabled:
        return {
            "enabled": False,
            "rollback_force": rollback_force,
            "canary_percent": canary_percent,
            "assigned": True,
            "variant": "legacy",
            "bucket": None,
            "reason": "release_disabled",
        }

    if rollback_force:
        return {
            "enabled": True,
            "rollback_force": True,
            "canary_percent": canary_percent,
            "assigned": False,
            "variant": "rollback",
            "bucket": None,
            "reason": "rollback_forced",
        }

    if not user:
        return {
            "enabled": True,
            "rollback_force": False,
            "canary_percent": canary_percent,
            "assigned": False,
            "variant": "control",
            "bucket": None,
            "reason": "unauthenticated",
        }

    key = f"{getattr(user, 'id', '')}:{session_id}:{salt}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    assigned = float(bucket) < float(canary_percent)
    return {
        "enabled": True,
        "rollback_force": False,
        "canary_percent": canary_percent,
        "assigned": bool(assigned),
        "variant": "canary" if assigned else "control",
        "bucket": int(bucket),
        "reason": "bucket_assignment",
    }


def _hito5_should_inject_qaf(release_ctx: dict) -> bool:
    if not isinstance(release_ctx, dict):
        return True
    if not bool(release_ctx.get("enabled")):
        return True
    if bool(release_ctx.get("rollback_force")):
        return False
    return bool(release_ctx.get("assigned"))


def _hito5_should_use_fused_runtime(release_ctx: dict) -> bool:
    fused_enabled = _hito5_env_bool("HITO5_FUSED_MODE_ENABLED", False)
    if not fused_enabled:
        return False

    if not isinstance(release_ctx, dict):
        return True

    if bool(release_ctx.get("rollback_force")):
        return False

    canary_only = _hito5_env_bool("HITO5_FUSED_MODE_CANARY_ONLY", True)
    if not canary_only:
        return True

    return bool(release_ctx.get("enabled")) and bool(release_ctx.get("assigned"))


def _hito5_compose_fused_output(
    *,
    message: str,
    qaf_cognition: dict | None,
    protocol_cognitive_body: dict | None,
    preferred_outputs: list[str] | None = None,
) -> str:
    candidates = [str(x).strip() for x in (preferred_outputs or []) if str(x or "").strip()]
    if candidates:
        return candidates[0]

    decision = qaf_cognition.get("decision") if isinstance(qaf_cognition, dict) and isinstance(qaf_cognition.get("decision"), dict) else {}
    mode = str(decision.get("mode") or "").strip().lower()
    dtyp = str(decision.get("type") or "").strip().lower()

    actions = decision.get("next_3_actions") if isinstance(decision.get("next_3_actions"), list) else []
    action_titles = [
        str(row.get("title") or "").strip()
        for row in actions
        if isinstance(row, dict) and str(row.get("title") or "").strip()
    ][:3]

    follow = decision.get("follow_up_questions") if isinstance(decision.get("follow_up_questions"), list) else []
    follow_prompt = ""
    for row in follow:
        if isinstance(row, dict) and str(row.get("prompt") or "").strip():
            follow_prompt = str(row.get("prompt") or "").strip()
            break

    header = "Activé el modo cognitivo QAF"
    if mode:
        header = f"Activé el modo QAF: {mode}"

    lines = [header + "."]
    if action_titles:
        lines.append("")
        for idx, title in enumerate(action_titles, start=1):
            lines.append(f"{idx}) {title}.")

    if dtyp in ("ask_clarifying", "needs_confirmation") and follow_prompt:
        lines.append("")
        lines.append(follow_prompt)

    if isinstance(protocol_cognitive_body, dict):
        summary = protocol_cognitive_body.get("summary") if isinstance(protocol_cognitive_body.get("summary"), dict) else {}
        week_goal = str(summary.get("week_goal") or "").strip()
        if week_goal:
            lines.append("")
            lines.append(f"Meta semanal sugerida: {week_goal}.")

    result = "\n".join(lines).strip()
    if result:
        return result

    msg = str(message or "").strip()
    if msg:
        return (
            "Recibí tu mensaje y activé una respuesta local segura sin n8n. "
            "Puedo orientarte con pasos concretos si me confirmas si quieres priorizar nutrición, entrenamiento o salud."
        )

    return "Modo local activo. Comparte tu objetivo de hoy y te doy 3 acciones concretas."


def _hito5_apply_response_guardrail(output_text: str, qaf_cognition: dict | None) -> tuple[str, bool]:
    if not isinstance(qaf_cognition, dict):
        return output_text, False

    decision = qaf_cognition.get("decision") if isinstance(qaf_cognition.get("decision"), dict) else {}
    flags = qaf_cognition.get("flags") if isinstance(qaf_cognition.get("flags"), dict) else {}
    needs_human = bool(flags.get("human_validation_required")) or str(decision.get("type") or "") in ("needs_confirmation", "ask_clarifying")

    if not needs_human:
        return output_text, False

    follow = decision.get("follow_up_questions") if isinstance(decision.get("follow_up_questions"), list) else []
    prompt = ""
    for row in follow:
        if isinstance(row, dict) and str(row.get("prompt") or "").strip():
            prompt = str(row.get("prompt") or "").strip()
            break

    base = str(output_text or "").strip()
    low = base.lower()
    if "confirm" in low or "¿" in base:
        return base, False

    if not prompt:
        prompt = "Antes de continuar, confirma si te sientes en un estado seguro para tomar esta decisión hoy."
    merged = f"{base}\n\n{prompt}".strip()
    return merged, True


def _hito5_write_trace(
    *,
    request,
    user,
    session_id: str,
    release_ctx: dict,
    qaf_cognition: dict | None,
    protocol_cognitive_body: dict | None,
    status_code: int,
    guardrail_applied: bool,
    error: str = "",
) -> None:
    try:
        bench = _bench_event_get(request) if request else None
        latency_total = None
        latency_n8n = None
        latency_qaf = None
        flow = ""
        tokens_in = None
        tokens_out = None
        if isinstance(bench, dict):
            latency_total = bench.get("latency_ms_total")
            latency_n8n = bench.get("latency_ms_n8n")
            latency_qaf = bench.get("latency_ms_qaf")
            flow = str(bench.get("flow") or "")
            tokens_in = bench.get("tokens_in")
            tokens_out = bench.get("tokens_out")

        decision = qaf_cognition.get("decision") if isinstance(qaf_cognition, dict) and isinstance(qaf_cognition.get("decision"), dict) else {}
        flags = qaf_cognition.get("flags") if isinstance(qaf_cognition, dict) and isinstance(qaf_cognition.get("flags"), dict) else {}

        before_json = {
            "session_id": str(session_id or ""),
            "flow": flow,
            "release": {
                "enabled": bool(release_ctx.get("enabled")) if isinstance(release_ctx, dict) else False,
                "variant": str(release_ctx.get("variant") or "") if isinstance(release_ctx, dict) else "",
                "assigned": bool(release_ctx.get("assigned")) if isinstance(release_ctx, dict) else False,
                "bucket": release_ctx.get("bucket") if isinstance(release_ctx, dict) else None,
                "canary_percent": release_ctx.get("canary_percent") if isinstance(release_ctx, dict) else None,
            },
            "qaf_mode": str(decision.get("mode") or ""),
            "qaf_decision_type": str(decision.get("type") or ""),
            "human_validation_required": bool(flags.get("human_validation_required")),
        }

        after_json = {
            "status_code": int(status_code or 0),
            "guardrail_applied": bool(guardrail_applied),
            "protocol_present": bool(isinstance(protocol_cognitive_body, dict)),
            "latency_ms_total": latency_total,
            "latency_ms_n8n": latency_n8n,
            "latency_ms_qaf": latency_qaf,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "error": str(error or "")[:400],
        }

        req = getattr(request, "_request", request)
        ip = ""
        ua = ""
        try:
            ip = req.META.get("HTTP_X_FORWARDED_FOR", "").split(",")[0].strip() or req.META.get("REMOTE_ADDR", "")
            ua = req.META.get("HTTP_USER_AGENT", "")
        except Exception:
            pass

        AuditLog.objects.create(
            actor=user if getattr(user, "is_authenticated", False) else None,
            action="llm.hito5.inference",
            entity_type="chat",
            entity_id=str(getattr(user, "id", "") or "guest"),
            before_json=before_json,
            after_json=after_json,
            reason="hito5_runtime_trace",
            ip=ip or None,
            user_agent=str(ua or "")[:500],
        )
    except Exception:
        pass


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


def _mercadopago_monthly_factor(frequency, frequency_type):
    try:
        freq = float(frequency or 0)
    except Exception:
        return None
    if freq <= 0:
        return None

    ftype = str(frequency_type or '').strip().lower()
    if ftype in ('month', 'months'):
        return 1.0 / freq
    if ftype in ('year', 'years'):
        return 1.0 / (12.0 * freq)
    if ftype in ('week', 'weeks'):
        return 4.345 / freq
    if ftype in ('day', 'days'):
        return 30.0 / freq
    return None


def _mercadopago_current_mrr_cop():
    """Obtiene MRR mensual aproximado REAL desde Mercado Pago (suscripciones activas).

    Usa preapproval/search y normaliza a mensual según auto_recurring.
    """
    token = (os.getenv('MERCADOPAGO_ACCESS_TOKEN') or '').strip()
    if not token:
        return {'ok': False, 'error': 'mp_token_missing'}

    total_mrr = 0.0
    active_count = 0
    skipped_non_cop = 0

    limit = 100
    max_pages = 20
    statuses_to_scan = ('authorized', 'active')
    seen_ids = set()

    for status_to_scan in statuses_to_scan:
        offset = 0
        for _ in range(max_pages):
            try:
                resp = requests.get(
                    'https://api.mercadopago.com/preapproval/search',
                    headers={"Authorization": f"Bearer {token}"},
                    params={
                        'status': status_to_scan,
                        'limit': limit,
                        'offset': offset,
                    },
                    timeout=15,
                )
            except Exception as exc:
                return {'ok': False, 'error': f'mp_request_failed:{exc}'}

            if resp.status_code != 200:
                return {'ok': False, 'error': f'mp_http_{resp.status_code}'}

            data = resp.json() if resp.content else {}
            results = data.get('results') if isinstance(data, dict) else None
            if not isinstance(results, list) or len(results) == 0:
                break

            for sub in results:
                preapproval_id = str(sub.get('id') or '').strip()
                if preapproval_id and preapproval_id in seen_ids:
                    continue

                status_str = str(sub.get('status') or '').strip().lower()
                if status_str not in ('authorized', 'active'):
                    continue

                auto = sub.get('auto_recurring') if isinstance(sub.get('auto_recurring'), dict) else {}
                currency = str(auto.get('currency_id') or '').strip().upper()
                if currency and currency != 'COP':
                    skipped_non_cop += 1
                    continue

                try:
                    amount = float(auto.get('transaction_amount') or 0)
                except Exception:
                    amount = 0.0
                if amount <= 0:
                    continue

                factor = _mercadopago_monthly_factor(auto.get('frequency'), auto.get('frequency_type'))
                if factor is None:
                    continue

                if preapproval_id:
                    seen_ids.add(preapproval_id)
                total_mrr += (amount * factor)
                active_count += 1

            if len(results) < limit:
                break
            offset += limit

    return {
        'ok': True,
        'mrr_cop': round(total_mrr, 2),
        'active_subscriptions': int(active_count),
        'skipped_non_cop': int(skipped_non_cop),
    }


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

    mp_mrr = _mercadopago_current_mrr_cop()

    if mp_mrr.get('ok'):
        mrr_current_cop = float(mp_mrr.get('mrr_cop') or 0)
        mrr_source = 'mercadopago'
        mrr_active_subscriptions = int(mp_mrr.get('active_subscriptions') or 0)
    else:
        mrr_current_cop = 0.0
        mrr_source = 'mercadopago_unavailable'
        mrr_active_subscriptions = 0

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
                "mrr_current_cop": round(mrr_current_cop, 2),
                "mrr_source": mrr_source,
                "mrr_active_subscriptions": mrr_active_subscriptions,
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


def _normalize_experience_key(flow_value: str):
    raw = str(flow_value or '').strip().lower()
    if not raw:
        return None

    aliases = {
        'exp-001_calories': 'exp-001_calories',
        'exp-002_meal_coherence': 'exp-002_meal_coherence',
        'exp-003_metabolic_profile': 'exp-003_metabolic_profile',
        'exp-004_meal_plan': 'exp-004_meal_plan',
        'exp-005_body_trend': 'exp-005_body_trend',
        'exp-006_posture': 'exp-006_posture',
        'exp-007_lifestyle': 'exp-007_lifestyle',
        'exp-008_motivation': 'exp-008_motivation',
        'exp-009_progression': 'exp-009_progression',
        'exp-010_muscle_measure': 'exp-010_muscle_measure',
        'exp-011_skin_health': 'exp-011_skin_health',
        'exp-011_skin_habits': 'exp-011_skin_health',
        'exp-012_shape_presence': 'exp-012_shape_presence',
        'exp-012_couture_garments': 'exp-012_shape_presence',
        'exp-013_body_architecture': 'exp-013_body_architecture',
    }
    return aliases.get(raw)


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def admin_dashboard_ops_metrics(request):
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    if not getattr(request, "user", None) or not request.user.is_superuser:
        return Response({"error": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)

    range_info = _date_range_from_request(request, default_days=30)
    tz = range_info["tz"]

    experiences = [
        ('exp-001_calories', 'Exp-001 Calorías'),
        ('exp-002_meal_coherence', 'Exp-002 Coherencia'),
        ('exp-003_metabolic_profile', 'Exp-003 Metabólico'),
        ('exp-004_meal_plan', 'Exp-004 Menú'),
        ('exp-005_body_trend', 'Exp-005 Tendencia'),
        ('exp-006_posture', 'Exp-006 Postura'),
        ('exp-007_lifestyle', 'Exp-007 Lifestyle'),
        ('exp-008_motivation', 'Exp-008 Motivación'),
        ('exp-009_progression', 'Exp-009 Progresión'),
        ('exp-010_muscle_measure', 'Exp-010 Músculo'),
        ('exp-011_skin_health', 'Exp-011 Piel'),
        ('exp-012_shape_presence', 'Exp-012 Alta Costura'),
        ('exp-013_body_architecture', 'Exp-013 Arquitectura'),
    ]
    exp_keys = [k for k, _ in experiences]

    day_map = {}
    cur = range_info['date_from']
    while cur <= range_info['date_to']:
        key = cur.isoformat()
        day_map[key] = {'date': key, 'total': 0}
        for exp_key in exp_keys:
            day_map[key][exp_key] = 0
        cur += timedelta(days=1)

    trace_qs = AuditLog.objects.filter(
        action='llm.hito5.inference',
        occurred_at__gte=range_info['start_utc'],
        occurred_at__lte=range_info['end_utc'],
    ).values('occurred_at', 'before_json', 'after_json')

    requests_total = 0
    requests_success = 0
    requests_error = 0

    latency_total_sum = 0.0
    latency_total_count = 0
    latency_n8n_sum = 0.0
    latency_n8n_count = 0
    latency_qaf_sum = 0.0
    latency_qaf_count = 0

    tokens_in_total = 0
    tokens_out_total = 0

    for row in trace_qs:
        occurred_at = row.get('occurred_at')
        before = row.get('before_json') if isinstance(row.get('before_json'), dict) else {}
        after = row.get('after_json') if isinstance(row.get('after_json'), dict) else {}

        if occurred_at:
            try:
                local_day = occurred_at.astimezone(tz).date().isoformat()
            except Exception:
                local_day = occurred_at.date().isoformat()
        else:
            local_day = None

        exp_key = _normalize_experience_key(before.get('flow'))
        if local_day in day_map and exp_key in exp_keys:
            day_map[local_day][exp_key] += 1
            day_map[local_day]['total'] += 1

        requests_total += 1
        try:
            status_code = int(after.get('status_code') or 0)
        except Exception:
            status_code = 0
        if 200 <= status_code < 300:
            requests_success += 1
        if status_code >= 400 or bool(after.get('error')):
            requests_error += 1

        for key_name, acc_sum_name, acc_count_name in (
            ('latency_ms_total', 'latency_total_sum', 'latency_total_count'),
            ('latency_ms_n8n', 'latency_n8n_sum', 'latency_n8n_count'),
            ('latency_ms_qaf', 'latency_qaf_sum', 'latency_qaf_count'),
        ):
            try:
                val = float(after.get(key_name))
            except Exception:
                val = None
            if val is None or val < 0:
                continue
            if acc_sum_name == 'latency_total_sum':
                latency_total_sum += val
                latency_total_count += 1
            elif acc_sum_name == 'latency_n8n_sum':
                latency_n8n_sum += val
                latency_n8n_count += 1
            else:
                latency_qaf_sum += val
                latency_qaf_count += 1

        try:
            tokens_in_total += int(float(after.get('tokens_in') or 0))
        except Exception:
            pass
        try:
            tokens_out_total += int(float(after.get('tokens_out') or 0))
        except Exception:
            pass

    avg_latency_total = (latency_total_sum / latency_total_count) if latency_total_count else None
    avg_latency_n8n = (latency_n8n_sum / latency_n8n_count) if latency_n8n_count else None
    avg_latency_qaf = (latency_qaf_sum / latency_qaf_count) if latency_qaf_count else None

    success_rate = (requests_success / requests_total) if requests_total else 0.0
    error_rate = (requests_error / requests_total) if requests_total else 0.0

    def _env_float(name, default):
        try:
            return float(os.getenv(name, str(default)) or default)
        except Exception:
            return float(default)

    in_cost_per_1k = _env_float('HITO5_COST_INPUT_PER_1K_USD', 0.0)
    out_cost_per_1k = _env_float('HITO5_COST_OUTPUT_PER_1K_USD', 0.0)
    usd_to_cop = _env_float('USD_TO_COP', 4000.0)

    estimated_total_usd = (tokens_in_total / 1000.0) * in_cost_per_1k + (tokens_out_total / 1000.0) * out_cost_per_1k
    estimated_total_cop = estimated_total_usd * usd_to_cop

    active_users_range = User.objects.filter(
        is_active=True,
        last_login__isnull=False,
        last_login__gte=range_info['start_utc'],
        last_login__lte=range_info['end_utc'],
    ).count()
    cost_per_active_user_cop = (estimated_total_cop / active_users_range) if active_users_range else None

    series = [day_map[k] for k in sorted(day_map.keys())]

    return Response(
        {
            'data': {
                'experiences': [{'key': k, 'label': label} for k, label in experiences],
                'series': series,
                'benchmark': {
                    'requests_total': requests_total,
                    'requests_success': requests_success,
                    'requests_error': requests_error,
                    'success_rate': round(success_rate, 6),
                    'error_rate': round(error_rate, 6),
                    'avg_latency_ms_total': round(avg_latency_total, 2) if avg_latency_total is not None else None,
                    'avg_latency_ms_n8n': round(avg_latency_n8n, 2) if avg_latency_n8n is not None else None,
                    'avg_latency_ms_qaf': round(avg_latency_qaf, 2) if avg_latency_qaf is not None else None,
                    'tokens_in_total': int(tokens_in_total),
                    'tokens_out_total': int(tokens_out_total),
                },
                'costs': {
                    'estimated_total_usd': round(estimated_total_usd, 6),
                    'estimated_total_cop': round(estimated_total_cop, 2),
                    'active_users_range': int(active_users_range),
                    'cost_per_active_user_cop': round(cost_per_active_user_cop, 2) if cost_per_active_user_cop is not None else None,
                    'model': {
                        'input_cost_per_1k_usd': in_cost_per_1k,
                        'output_cost_per_1k_usd': out_cost_per_1k,
                        'usd_to_cop': usd_to_cop,
                    },
                },
            },
            'meta': {
                'dateFrom': str(range_info['date_from']),
                'dateTo': str(range_info['date_to']),
                'timezone': str(range_info['tz']),
                'days': range_info['days'],
            },
        }
    )


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
        # Si ya está activo por pasarela, no degradar plan al expirar trial.
        if str(getattr(user, 'billing_status', '') or '').strip().lower() != "active" and user.plan == "Premium":
            user.plan = "Gratis"
        if user.billing_status == "trial":
            user.billing_status = "expired"
        user.save()


def _is_premium_active(user):
    _apply_trial_status(user)
    billing_status = str(getattr(user, 'billing_status', '') or '').strip().lower()

    # Premium por cobro activo
    if billing_status == "active":
        return True

    # Premium por trial vigente
    if user.trial_active and user.trial_ends_at and timezone.now() < user.trial_ends_at:
        return True

    # Cualquier otro estado se considera no premium para bloquear funciones de pago.
    return False


def _qaf_premium_gate_payload(
    module_key: str,
    module_title: str,
    preview_text: str,
    *,
    request=None,
    user=None,
    source: str = "qaf",
):
    if request is not None:
        _audit_log(
            request,
            action="billing.premium_gate.shown",
            entity_type="qaf_module",
            entity_id=str(module_key),
            after={
                "module": module_key,
                "title": module_title,
                "source": source,
                "plan": getattr(user, "plan", None) if user else None,
                "trial_active": bool(getattr(user, "trial_active", False)) if user else None,
            },
        )

    output = (
        f"**{module_title}**\n"
        f"{preview_text}\n\n"
        "Esta función completa está disponible en **Premium**."
    )
    return {
        'success': False,
        'premium_required': True,
        'module': module_key,
        'output': output,
        'text': output,
        'quick_actions': [
            {
                'label': 'Activar Premium',
                'type': 'link',
                'href': '/pages/plans/Planes.html',
                'payload': {
                    'telemetry': {
                        'event': 'premium_cta_click',
                        'source': source,
                        'module': module_key,
                    }
                },
            }
        ],
        'preview': {
            'title': module_title,
            'summary': preview_text,
        },
    }


def _qaf_append_premium_cta(actions, module: str = 'unknown', source: str = 'qaf'):
    actions = actions if isinstance(actions, list) else []
    for action in actions:
        if not isinstance(action, dict):
            continue
        if str(action.get('type') or '').strip().lower() == 'link' and str(action.get('href') or '').strip() == '/pages/plans/Planes.html':
            return actions
    actions.append({
        'label': 'Activar Premium',
        'type': 'link',
        'href': '/pages/plans/Planes.html',
        'payload': {
            'telemetry': {
                'event': 'premium_cta_click',
                'source': source,
                'module': module,
            }
        },
    })
    return actions


def _free_image_monthly_limit() -> int:
    try:
        value = int(os.getenv('FREE_IMAGE_MONTHLY_LIMIT', '15') or 15)
    except Exception:
        value = 15
    return max(1, min(value, 500))


def _image_quota_month_key() -> str:
    now_local = timezone.localtime(timezone.now())
    return f"{now_local.year:04d}-{now_local.month:02d}"


def _get_image_quota_status(user):
    limit = _free_image_monthly_limit()
    month_key = _image_quota_month_key()
    coach_state = getattr(user, 'coach_state', {}) or {}
    counters = coach_state.get('image_upload_counts') if isinstance(coach_state.get('image_upload_counts'), dict) else {}
    used = 0
    try:
        used = int(counters.get(month_key) or 0)
    except Exception:
        used = 0
    used = max(0, used)
    remaining = max(0, limit - used)
    return {
        'month_key': month_key,
        'limit': int(limit),
        'used': int(used),
        'remaining': int(remaining),
    }


def _increment_image_quota_usage(user):
    status_now = _get_image_quota_status(user)
    month_key = status_now['month_key']

    coach_state = getattr(user, 'coach_state', {}) or {}
    counters = coach_state.get('image_upload_counts') if isinstance(coach_state.get('image_upload_counts'), dict) else {}
    counters2 = dict(counters)
    counters2[month_key] = int(status_now['used']) + 1

    # Mantener solo últimos 6 meses para no crecer indefinidamente.
    keys_sorted = sorted([k for k in counters2.keys() if isinstance(k, str)])
    if len(keys_sorted) > 6:
        for old_key in keys_sorted[:-6]:
            counters2.pop(old_key, None)

    coach_state2 = dict(coach_state)
    coach_state2['image_upload_counts'] = counters2
    user.coach_state = coach_state2
    user.coach_state_updated_at = timezone.now()
    user.save(update_fields=['coach_state', 'coach_state_updated_at'])

    return _get_image_quota_status(user)

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
            "- is_screenshot_or_ui (boolean): true si parece captura de pantalla, página web, app, chat, dashboard o interfaz\n"
            "- is_document_like (boolean): true si predomina texto/documento sobre foto real\n"
            "Reglas: si ves comida o etiqueta nutricional => route='nutrition'. Si es una persona entrenando o en contexto gimnasio => route='training'. "
            "Si es primer plano de piel/músculo/rostro para salud/belleza => route='health'. Si no encaja => route='quantum'."
            "Si parece captura de pantalla/UI/documento web, prioriza route='quantum' salvo que el usuario pida explícitamente una experiencia concreta."
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


def _normalize_attachment_url(value: str) -> str:
    """Normaliza una URL http(s) candidata.

    - Evita concatenaciones accidentales del tipo "...https://...https://..."
    - No intenta validar; solo sanea de forma conservadora.
    """

    raw = str(value or '').strip()
    if not raw:
        return ''

    # Si hay múltiples https://, tomar la última (suele ser la real tras concatenación)
    last_https = raw.rfind('https://')
    if last_https > 0:
        raw = raw[last_https:]

    last_http = raw.rfind('http://')
    if last_http > 0 and (last_https < 0 or last_http > last_https):
        raw = raw[last_http:]

    return raw.strip()


def _is_signed_chat_attachment_url(url_value: str) -> bool:
    """Valida de forma conservadora si es una URL SAS del container de attachments.

    Nota: esto NO garantiza ownership por username, pero el SAS es un secreto y se restringe a nuestro storage.
    """

    try:
        u = _normalize_attachment_url(url_value)
        if not u:
            return False
        if 'sig=' not in u.lower():
            return False

        parsed = urlparse(u)
        host = (parsed.hostname or '').strip().lower()
        if not host or not host.endswith('.blob.core.windows.net'):
            return False

        account_name = getattr(settings, 'AZURE_STORAGE_ACCOUNT_NAME', '').strip().lower()
        if account_name:
            expected = f"{account_name}.blob.core.windows.net"
            if host != expected:
                return False

        container_name, blob_name = _extract_blob_ref_from_url(u)
        if container_name != _chat_attachment_container() or not blob_name:
            return False

        return True
    except Exception:
        return False


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

    _apply_trial_status(user)

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


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def gamification_claim_daily(request):
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user, auth_error = _require_authenticated_user(request)
    if auth_error:
        return auth_error

    try:
        return Response(claim_daily_wow_reward(user))
    except Exception as exc:
        return Response({"ok": False, "error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _attach_wow_event_payload(user, payload: Any, *, event_key: str, label: str | None = None) -> dict[str, Any]:
    base = payload if isinstance(payload, dict) else {"result": payload}
    out = dict(base)
    try:
        wow_event = award_wow_event(user, event_key=event_key, label=label)
        out["wow_event"] = wow_event
        if isinstance(wow_event, dict) and isinstance(wow_event.get("gamification"), dict):
            out["gamification"] = wow_event.get("gamification")
    except Exception:
        pass
    return out


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

    _apply_trial_status(user)

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

    # Auto-sync al abrir app (vía coach_context), no solo en pantalla de dispositivos.
    # Encola sync invisible si la última sync está vieja (>=3h) y evita duplicados recientes.
    # Fast-on-open opcional: ejecuta 1 sync inmediato para mejorar frescura percibida.
    try:
        now = timezone.now()
        stale_after = timezone.timedelta(hours=3)
        fast_on_open = (os.getenv("AUTO_SYNC_FAST_ON_OPEN", "1") or "").strip().lower() in ("1", "true", "yes", "y", "on")
        try:
            fast_max = int(os.getenv("AUTO_SYNC_FAST_ON_OPEN_MAX_PROVIDERS", "1") or 1)
        except Exception:
            fast_max = 1
        if fast_max < 0:
            fast_max = 0
        if fast_max > 2:
            fast_max = 2

        fast_candidates: list[str] = []
        for d in device_qs:
            if d.status != "connected":
                continue
            if d.last_sync_at and (now - d.last_sync_at) < stale_after:
                continue
            recently = (
                user.sync_requests.filter(
                    status="pending",
                    provider=d.provider,
                    requested_at__gte=now - timezone.timedelta(minutes=30),
                ).exists()
            )
            if not recently:
                enqueue_sync_request(
                    user,
                    provider=d.provider,
                    reason="app_open_context",
                    priority=6,
                )
                fast_candidates.append(d.provider)

        if fast_on_open and fast_max > 0 and fast_candidates:
            for provider in fast_candidates[:fast_max]:
                try:
                    sync_device(user, provider)
                except Exception:
                    pass
    except Exception:
        pass

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
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def mercadopago_checkout_link(request):
    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=401)

    cycle = (request.data.get('cycle') or 'monthly').strip().lower()
    if cycle not in ('monthly', 'annual'):
        return Response({'error': 'Ciclo invalido. Usa monthly o annual'}, status=400)

    monthly_plan_id = (os.getenv('MERCADOPAGO_PREAPPROVAL_PLAN_ID_MONTHLY') or '').strip()
    annual_plan_id = (os.getenv('MERCADOPAGO_PREAPPROVAL_PLAN_ID_ANNUAL') or '').strip()
    plan_id = monthly_plan_id if cycle == 'monthly' else annual_plan_id

    if not plan_id:
        return Response({'error': 'Plan de suscripcion no configurado'}, status=500)

    base_url = 'https://www.mercadopago.com.co/subscriptions/checkout'
    query = urlencode({
        'preapproval_plan_id': plan_id,
        'external_reference': user.username,
    })
    checkout_url = f'{base_url}?{query}'

    _audit_log(
        request,
        action='billing.checkout_link.generated',
        entity_type='user',
        entity_id=str(user.id),
        after={
            'provider': 'mercadopago',
            'cycle': cycle,
            'source': 'plans_page',
        },
    )

    return Response({
        'success': True,
        'checkout_url': checkout_url,
        'cycle': cycle,
    })


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def premium_telemetry_event(request):
    if request.method == 'OPTIONS':
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=401)

    payload = request.data if isinstance(request.data, dict) else {}
    event = str(payload.get('event') or '').strip().lower()
    source = str(payload.get('source') or '').strip().lower() or 'unknown'
    module = str(payload.get('module') or '').strip().lower() or 'unknown'

    allowed_events = {
        'premium_cta_click',
        'premium_gate_shown_client',
    }
    if event not in allowed_events:
        return Response({'error': 'Evento no soportado'}, status=400)

    _audit_log(
        request,
        action=f'billing.{event}',
        entity_type='qaf_module',
        entity_id=module[:120],
        after={
            'event': event,
            'source': source,
            'module': module,
            'plan': getattr(user, 'plan', None),
            'trial_active': bool(getattr(user, 'trial_active', False)),
            'billing_status': getattr(user, 'billing_status', None),
        },
    )
    return Response({'success': True})


def _validate_mercadopago_webhook_secret(request) -> bool:
    expected = (os.getenv('MERCADOPAGO_WEBHOOK_SECRET') or '').strip()
    if not expected:
        return True

    candidates = [
        request.headers.get('X-Webhook-Secret'),
        request.headers.get('X-MercadoPago-Webhook-Secret'),
        request.headers.get('X-Mercadopago-Webhook-Secret'),
    ]

    try:
        query_secret = request.query_params.get('secret') if hasattr(request, 'query_params') else None
    except Exception:
        query_secret = None
    if query_secret:
        candidates.append(query_secret)

    for value in candidates:
        if value and secrets.compare_digest(str(value).strip(), expected):
            return True
    return False


@api_view(['POST'])
def mercadopago_webhook(request):
    if not _validate_mercadopago_webhook_secret(request):
        return Response({'error': 'Unauthorized webhook'}, status=401)

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

            if requested_plan == "Premium":
                # Extensión de prueba desde dashboard: 14 días, excepto si ya está en cobro activo.
                current_billing = str(getattr(user, 'billing_status', '') or '').strip().lower()
                if current_billing != 'active':
                    now = timezone.now()
                    base_start = now
                    try:
                        if user.trial_active and user.trial_ends_at and user.trial_ends_at > now:
                            base_start = user.trial_ends_at
                    except Exception:
                        base_start = now

                    user.trial_active = True
                    if not user.trial_started_at:
                        user.trial_started_at = now
                    user.trial_ends_at = base_start + timedelta(days=14)
                    user.billing_status = 'trial'
            elif requested_plan == "Gratis":
                # Revocación manual: bloquear premium inmediatamente.
                user.trial_active = False
                user.trial_started_at = None
                user.trial_ends_at = None
                user.billing_status = 'free'

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

        is_image_attachment = extension in ('.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif')

        if is_image_attachment and not _is_premium_active(user):
            quota = _get_image_quota_status(user)
            if quota['used'] >= quota['limit']:
                _audit_log(
                    request,
                    action='billing.image_quota.exceeded',
                    entity_type='user',
                    entity_id=str(user.id),
                    after={
                        'month': quota['month_key'],
                        'used': quota['used'],
                        'limit': quota['limit'],
                    },
                )
                return Response(
                    {
                        'success': False,
                        'premium_required': True,
                        'error': f"Has usado {quota['used']}/{quota['limit']} imágenes este mes. Activa Premium para continuar.",
                        'output': (
                            f"**Límite mensual alcanzado**\n"
                            f"Ya usaste {quota['used']}/{quota['limit']} imágenes en tu plan Gratis.\n\n"
                            "Activa Premium para seguir usando servicios con imágenes sin este límite."
                        ),
                        'quick_actions': [
                            {
                                'label': 'Activar Premium',
                                'type': 'link',
                                'href': '/pages/plans/Planes.html',
                                'payload': {
                                    'telemetry': {
                                        'event': 'premium_cta_click',
                                        'source': 'image_quota',
                                        'module': 'chat_image_upload_limit',
                                    }
                                },
                            }
                        ],
                        'image_quota': quota,
                    },
                    status=402,
                )

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

        if is_image_attachment:
            quota_after = _increment_image_quota_usage(user)
            payload['image_quota'] = quota_after
            if (not _is_premium_active(user)) and quota_after['used'] in (10, 14):
                payload['quota_notice'] = (
                    f"Llevas {quota_after['used']}/{quota_after['limit']} imágenes este mes en plan Gratis."
                )
            _audit_log(
                request,
                action='billing.image_quota.uploaded',
                entity_type='user',
                entity_id=str(user.id),
                after={
                    'month': quota_after['month_key'],
                    'used': quota_after['used'],
                    'limit': quota_after['limit'],
                    'remaining': quota_after['remaining'],
                },
            )

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

        # Benchmark: inferir flow temprano (si middleware está activo)
        try:
            _bench_event_note(request, flow=_infer_chat_flow(message, request.data))
            if attachment_url:
                _bench_event_note(request, attachment_bytes=0)
        except Exception:
            pass

        # 2. Configuración de n8n
        n8n_url = getattr(settings, "N8N_WEBHOOK_URL", "").strip() or "http://172.200.202.47/webhook/general-agent-gotogym-v2"

        # Resolver usuario temprano (para fallback de adjuntos y sessionId seguro)
        user = _resolve_request_user(request)
        if not user:
            try:
                username_hint = str(request.data.get('username') or '').strip()
                if username_hint:
                    user = User.objects.filter(
                        Q(username__iexact=username_hint) | Q(email__iexact=username_hint)
                    ).first()
            except Exception:
                user = None

        if user:
            session_id = _safe_session_id(user)
        elif not session_id or session_id == 'invitado':
            session_id = f"guest_{uuid.uuid4().hex}"

        release_ctx = _hito5_release_context(user, session_id)
        hito5_inject_qaf = _hito5_should_inject_qaf(release_ctx)
        _bench_event_note(
            request,
            hito5_release_enabled=bool(release_ctx.get("enabled")),
            hito5_variant=str(release_ctx.get("variant") or ""),
            hito5_canary_assigned=bool(release_ctx.get("assigned")),
            hito5_rollback_force=bool(release_ctx.get("rollback_force")),
        )

        # Router UX transversal (13 experiencias):
        # - confirmar intención (ver último vs iniciar nuevo)
        # - resolver "ver último" sin activar flujo de captura/medición
        # - devolver CTA de inicio explícito cuando corresponde
        try:
            data_in = request.data if isinstance(request.data, dict) else {}
            msg_low_router = str(message or '').strip().lower()

            service_intent = data_in.get('service_intent') if isinstance(data_in.get('service_intent'), dict) else {}
            service_exp = str(service_intent.get('experience') or '').strip().lower()
            service_action = str(service_intent.get('action') or '').strip().lower()

            explicit_start = bool(re.search(r"\b(iniciar|inicia|empezar|empecemos|activar|nuevo|nueva|arrancar|hacer)\b", msg_low_router or ""))
            explicit_last = bool(re.search(r"\b(ultimo|último|ultimos|últimos|resultado|resultados|anterior|semana pasada|ver|mostrar|muestrame|muéstrame)\b", msg_low_router or ""))

            def _service_spec(exp_code: str) -> dict:
                specs = {
                    'exp-001_calories': {
                        'label': 'Calorías Inteligentes (QAF)',
                        'aliases': [r"calor[ií]a", r"qaf", r"foto\s+de\s+comida", r"analiza\s+mi\s+comida", r"analizar\s+comida"],
                        'start_actions': [
                            {'label': 'Tomar foto de comida', 'type': 'open_camera'},
                            {'label': 'Adjuntar foto de comida', 'type': 'open_attach'},
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    },
                    'exp-002_meal_coherence': {
                        'label': 'Coherencia Nutricional',
                        'aliases': [r"coherencia\s+nutricional", r"coherencia\s+de\s+comida", r"meal\s+coherence"],
                        'start_actions': [
                            {'label': 'Evaluar coherencia de comida', 'type': 'message', 'text': 'Quiero evaluar la coherencia de esta comida'},
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    },
                    'exp-003_metabolic_profile': {
                        'label': 'Perfil Metabólico',
                        'aliases': [r"perfil\s+metab[oó]lico", r"metab[oó]lico", r"tdee", r"tmb"],
                        'start_actions': [
                            {
                                'label': 'Ejecutar evaluación metabólica',
                                'type': 'message',
                                'text': 'Ejecutar evaluación metabólica',
                                'payload': {
                                    'metabolic_profile_request': {'start': True},
                                    'service_intent': {'experience': 'exp-003_metabolic_profile', 'action': 'start_new'},
                                },
                            },
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    },
                    'exp-004_meal_plan': {
                        'label': 'Menú Semanal',
                        'aliases': [r"men[uú]\s+semanal", r"meal\s+plan", r"plan\s+de\s+comidas", r"plan\s+semanal\s+personalizado", r"plan\s+nutricional\s+semanal", r"plan\s+alimenticio\s+semanal"],
                        'start_actions': [
                            {'label': 'Generar menú semanal', 'type': 'message', 'text': 'Menú semanal', 'payload': {'meal_plan_request': {'variety': 'normal', 'meals_per_day': 3}}},
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    },
                    'exp-005_body_trend': {
                        'label': 'Tendencia 6 Semanas',
                        'aliases': [r"tendencia", r"6\s+semanas", r"body\s+trend"],
                        'start_actions': [
                            {'label': 'Calcular tendencia 6 semanas', 'type': 'message', 'text': 'Tendencia 6 semanas', 'payload': {'body_trend_request': {}}},
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    },
                    'exp-006_posture': {
                        'label': 'Corrección de Postura',
                        'aliases': [r"postura", r"posture", r"correcci[oó]n\s+de\s+postura"],
                        'start_actions': [
                            {'label': 'Tomar foto frontal', 'type': 'posture_capture', 'view': 'front', 'source': 'camera'},
                            {'label': 'Adjuntar foto frontal', 'type': 'posture_capture', 'view': 'front', 'source': 'attach'},
                            {'label': 'Cancelar', 'type': 'posture_cancel'},
                        ],
                    },
                    'exp-007_lifestyle': {
                        'label': 'Estado de Hoy (Lifestyle)',
                        'aliases': [r"estado\s+de\s+hoy", r"lifestyle", r"energ[ií]a\s+de\s+hoy"],
                        'start_actions': [
                            {'label': 'Iniciar estado de hoy', 'type': 'message', 'text': 'Estado de hoy', 'payload': {'lifestyle_request': {'days': 14}}},
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    },
                    'exp-008_motivation': {
                        'label': 'Motivación',
                        'aliases': [r"motivaci[oó]n", r"motivacion"],
                        'start_actions': [
                            {'label': 'Iniciar motivación', 'type': 'message', 'text': 'Motivación', 'payload': {'motivation_request': {'preferences': {'pressure': 'suave'}}}},
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    },
                    'exp-009_progression': {
                        'label': 'Evolución de Entrenamiento',
                        'aliases': [r"evoluci[oó]n\s+de\s+entrenamiento", r"progresi[oó]n", r"progression"],
                        'start_actions': [
                            {'label': 'Iniciar evolución', 'type': 'message', 'text': 'Evolución de entrenamiento', 'payload': {'progression_request': {}}},
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    },
                    'exp-010_muscle_measure': {
                        'label': 'Progreso Muscular',
                        'aliases': [r"progreso\s+muscular", r"medici[oó]n\s+muscular", r"muscle"],
                        'start_actions': [
                            {'label': 'Tomar foto frontal', 'type': 'muscle_capture', 'view': 'front_relaxed', 'source': 'camera'},
                            {'label': 'Adjuntar foto frontal', 'type': 'muscle_capture', 'view': 'front_relaxed', 'source': 'attach'},
                            {'label': 'Cancelar', 'type': 'muscle_cancel'},
                        ],
                    },
                    'exp-011_skin_health': {
                        'label': 'Vitalidad de la Piel',
                        'aliases': [r"vitalidad\s+de\s+la\s+piel", r"skin\s+health", r"piel"],
                        'start_actions': [
                            {'label': 'Tomar foto', 'type': 'open_camera'},
                            {'label': 'Adjuntar foto', 'type': 'open_attach'},
                            {'label': 'Cancelar', 'type': 'skin_cancel'},
                        ],
                    },
                    'exp-012_shape_presence': {
                        'label': 'Alta Costura Inteligente',
                        'aliases': [r"alta\s+costura", r"shape", r"presence", r"presencia"],
                        'start_actions': [
                            {'label': 'Tomar foto frontal', 'type': 'shape_capture', 'view': 'front_relaxed', 'source': 'camera'},
                            {'label': 'Adjuntar foto frontal', 'type': 'shape_capture', 'view': 'front_relaxed', 'source': 'attach'},
                            {'label': 'Cancelar', 'type': 'shape_cancel'},
                        ],
                    },
                    'exp-013_body_architecture': {
                        'label': 'Arquitectura Corporal',
                        'aliases': [r"arquitectura\s+corporal", r"postura\s+y\s+proporci[oó]n", r"postura\s*&\s*proporci[oó]n"],
                        'start_actions': [
                            {'label': 'Tomar foto frente', 'type': 'pp_capture', 'view': 'front_relaxed', 'source': 'camera'},
                            {'label': 'Adjuntar foto frente', 'type': 'pp_capture', 'view': 'front_relaxed', 'source': 'attach'},
                            {'label': 'Cancelar', 'type': 'pp_cancel'},
                        ],
                    },
                }
                return specs.get(exp_code) or {}

            def _detect_service_from_message(msg_low: str) -> str:
                # Heurística UX: cuando el usuario responde objetivos/preferencias para un plan,
                # mantenerlo en Experiencia 2 (menú semanal) y evitar desvío a calorías por la palabra "comida".
                try:
                    has_goal = bool(re.search(r"\b(mantener|mantenimiento|ganar\s+m[uú]sculo|subir\s+m[uú]sculo|definir|perder\s+grasa|objetivo)\b", msg_low or ""))
                    has_food_pref = bool(re.search(r"\b(comida|platos?|caser[oa]s?|bogotan[ao]|tradicional|restricci[oó]n|preferencia\s+alimentaria)\b", msg_low or ""))
                    has_training_pref = bool(re.search(r"\b(fuerza|entrenamiento|entrenar|resistencia|cardio|crossfit|gimnasio)\b", msg_low or ""))
                    has_weekly_frame = bool(re.search(r"\b(\d+\s*d[ií]as|semana|semanal)\b", msg_low or ""))
                    if (has_goal and has_food_pref and has_training_pref) or (has_food_pref and has_weekly_frame and has_training_pref):
                        return 'exp-004_meal_plan'
                except Exception:
                    pass

                for code in (
                    'exp-013_body_architecture', 'exp-012_shape_presence', 'exp-011_skin_health', 'exp-010_muscle_measure',
                    'exp-009_progression', 'exp-008_motivation', 'exp-007_lifestyle', 'exp-006_posture',
                    'exp-005_body_trend', 'exp-004_meal_plan', 'exp-003_metabolic_profile', 'exp-002_meal_coherence', 'exp-001_calories'
                ):
                    spec = _service_spec(code)
                    aliases = spec.get('aliases') if isinstance(spec.get('aliases'), list) else []
                    for pat in aliases:
                        try:
                            if re.search(pat, msg_low or ""):
                                return code
                        except Exception:
                            continue
                return ""

            def _latest_week_result(weekly_root: dict, key: str):
                try:
                    root = weekly_root.get(key) if isinstance(weekly_root.get(key), dict) else {}
                    if not isinstance(root, dict) or not root:
                        return None, ""
                    cand = sorted([k for k in root.keys() if isinstance(k, str)])
                    for wk in reversed(cand):
                        row = root.get(wk)
                        if not isinstance(row, dict):
                            continue
                        res = row.get('result') if isinstance(row.get('result'), dict) else None
                        if isinstance(res, dict):
                            return res, str(row.get('updated_at') or "").strip()
                except Exception:
                    return None, ""
                return None, ""

            def _resolve_last_result(exp_code: str, usr):
                if not usr:
                    return None, ""
                cs_loc = getattr(usr, 'coach_state', {}) or {}
                ws_loc = getattr(usr, 'coach_weekly_state', {}) or {}

                if exp_code == 'exp-013_body_architecture':
                    return _latest_week_result(ws_loc, 'posture_proportion')
                if exp_code == 'exp-012_shape_presence':
                    blob = cs_loc.get('couture_last_result') if isinstance(cs_loc.get('couture_last_result'), dict) else None
                    res = (blob or {}).get('result') if isinstance((blob or {}).get('result'), dict) else None
                    if isinstance(res, dict):
                        return res, str((blob or {}).get('updated_at') or "").strip()
                    return _latest_week_result(ws_loc, 'shape_presence')
                if exp_code == 'exp-011_skin_health':
                    blob = cs_loc.get('skin_last_result') if isinstance(cs_loc.get('skin_last_result'), dict) else None
                    res = (blob or {}).get('result') if isinstance((blob or {}).get('result'), dict) else None
                    return (res, str((blob or {}).get('updated_at') or "").strip()) if isinstance(res, dict) else _latest_week_result(ws_loc, 'skin_health')
                if exp_code == 'exp-010_muscle_measure':
                    blob = cs_loc.get('muscle_measure_last') if isinstance(cs_loc.get('muscle_measure_last'), dict) else None
                    res = (blob or {}).get('result') if isinstance((blob or {}).get('result'), dict) else None
                    return (res, str((blob or {}).get('updated_at') or "").strip()) if isinstance(res, dict) else _latest_week_result(ws_loc, 'muscle_measure')
                if exp_code == 'exp-009_progression':
                    blob = cs_loc.get('progression_last') if isinstance(cs_loc.get('progression_last'), dict) else None
                    res = (blob or {}).get('result') if isinstance((blob or {}).get('result'), dict) else None
                    return (res, str((blob or {}).get('updated_at') or "").strip()) if isinstance(res, dict) else (None, "")
                if exp_code == 'exp-008_motivation':
                    blob = cs_loc.get('motivation_last') if isinstance(cs_loc.get('motivation_last'), dict) else None
                    res = (blob or {}).get('result') if isinstance((blob or {}).get('result'), dict) else None
                    return (res, str((blob or {}).get('updated_at') or "").strip()) if isinstance(res, dict) else (None, "")
                if exp_code == 'exp-007_lifestyle':
                    blob = cs_loc.get('lifestyle_last') if isinstance(cs_loc.get('lifestyle_last'), dict) else None
                    res = (blob or {}).get('result') if isinstance((blob or {}).get('result'), dict) else None
                    return (res, str((blob or {}).get('updated_at') or "").strip()) if isinstance(res, dict) else (None, "")
                if exp_code == 'exp-006_posture':
                    hist = cs_loc.get('posture_measurements') if isinstance(cs_loc.get('posture_measurements'), list) else []
                    if hist:
                        row = hist[-1] if isinstance(hist[-1], dict) else {}
                        return row, str(row.get('ts') or "").strip()
                    return None, ""
                if exp_code == 'exp-005_body_trend':
                    blob = ws_loc.get('body_trend_last') if isinstance(ws_loc.get('body_trend_last'), dict) else None
                    res = (blob or {}).get('result') if isinstance((blob or {}).get('result'), dict) else None
                    return (res, str((blob or {}).get('updated_at') or "").strip()) if isinstance(res, dict) else (None, "")
                if exp_code == 'exp-004_meal_plan':
                    return _latest_week_result(ws_loc, 'meal_plan')
                if exp_code == 'exp-003_metabolic_profile':
                    res = ws_loc.get('metabolic_last') if isinstance(ws_loc.get('metabolic_last'), dict) else None
                    if isinstance(res, dict):
                        return res, str(res.get('updated_at') or "").strip()
                    return None, ""
                return None, ""

            if not service_exp:
                service_exp = _detect_service_from_message(msg_low_router)

            known_payload_keys = {
                'metabolic_profile_request',
                'posture_request', 'muscle_measure_request', 'posture_proportion_request',
                'shape_presence_request',
                'meal_plan_request', 'body_trend_request', 'lifestyle_request', 'motivation_request',
                'progression_request', 'skin_habits_request', 'skin_cancel', 'pp_cancel'
            }
            has_direct_payload = any(k in data_in for k in known_payload_keys)

            force_metabolic_start = False
            force_metabolic_show_last = False
            try:
                if service_exp == 'exp-003_metabolic_profile' and service_action in ('start_new', 'start', 'new_eval'):
                    force_metabolic_start = True
                if service_exp == 'exp-003_metabolic_profile' and service_action in ('show_last', 'view_last', 'last_result'):
                    force_metabolic_show_last = True
            except Exception:
                force_metabolic_start = False
                force_metabolic_show_last = False

            if force_metabolic_start or force_metabolic_show_last:
                has_direct_payload = True

            skip_service_router = False
            try:
                skip_service_router = bool(
                    service_exp == 'exp-003_metabolic_profile'
                    and service_action not in ('show_last', 'view_last', 'last_result')
                )
            except Exception:
                skip_service_router = False

            if user and service_exp and not attachment_url and not has_direct_payload and (not skip_service_router):
                spec = _service_spec(service_exp)
                label = str(spec.get('label') or 'Experiencia').strip()

                want_last = service_action in ('show_last', 'view_last', 'last_result') or explicit_last
                want_start = service_action in ('start_new', 'new_eval', 'start') or explicit_start

                if want_last:
                    last_res, updated_at = _resolve_last_result(service_exp, user)
                    start_payload = {'service_intent': {'experience': service_exp, 'action': 'start_new'}}
                    start_text = f'Iniciar {label}'
                    start_label = 'Iniciar evaluación nueva'
                    if service_exp == 'exp-003_metabolic_profile':
                        start_payload = {
                            'metabolic_profile_request': {'start': True},
                            'service_intent': {'experience': service_exp, 'action': 'start_new'},
                        }
                        start_text = 'Ejecutar evaluación metabólica'
                        start_label = 'Ejecutar evaluación metabólica'
                    if isinstance(last_res, dict):
                        interpreted = ""
                        try:
                            prompt = (
                                "Interpreta el último resultado del usuario para la experiencia solicitada.\n"
                                "Responde en español, en tono de coach, con coherencia y enfoque en la intención del usuario.\n"
                                "NO muestres JSON crudo ni campos técnicos irrelevantes.\n"
                                "Formato: 1) Resumen breve 2) Qué significa 3) Próximo paso recomendado.\n"
                                "Máximo 8 viñetas y sin pedir información que ya existe en el resultado.\n"
                                f"Experiencia: {label}.\n"
                                f"Mensaje del usuario: {str(message or '').strip()}\n"
                                f"Actualizado: {updated_at or 'N/D'}\n"
                            )
                            n8n_payload = {
                                "chatInput": prompt,
                                "message": prompt,
                                "sessionId": session_id,
                                "username": (getattr(user, 'username', '') or ''),
                                "auth_header": auth_header,
                                "attachment": "",
                                "attachment_text": "",
                                "service_context": {
                                    "experience": service_exp,
                                    "label": label,
                                    "updated_at": updated_at,
                                },
                                "qaf": {
                                    "type": service_exp,
                                    "result": last_res,
                                },
                                "system_rules": {
                                    "module": "service_last_result_interpreter",
                                    "no_raw_json": True,
                                    "strict_scope": True,
                                },
                            }
                            resp = _bench_n8n_post(request, n8n_url, n8n_payload, timeout=45)
                            if resp.status_code == 200:
                                try:
                                    data = resp.json()
                                except Exception:
                                    data = {"output": resp.text}
                                if isinstance(data, dict) and isinstance(data.get('output'), str):
                                    interpreted = (data.get('output') or '').strip()
                                elif isinstance(data, str):
                                    interpreted = data.strip()
                        except Exception:
                            interpreted = ""

                        # Fallback determinista (sin JSON crudo)
                        if not interpreted:
                            _bench_event_note(request, fallback_used=True)
                            key_takeaways = []
                            try:
                                if isinstance(last_res.get('insights'), list):
                                    key_takeaways = [str(x).strip() for x in last_res.get('insights') if str(x).strip()][:3]
                            except Exception:
                                key_takeaways = []

                            lines = [f"**Tu último resultado — {label}**"]
                            if updated_at:
                                lines.append(f"Actualizado: {updated_at}")
                            lines.append("")
                            lines.append("**Resumen**")
                            if key_takeaways:
                                for it in key_takeaways:
                                    lines.append(f"- {it}")
                            else:
                                lines.append("- Ya tengo tu última evaluación registrada y lista para usar en tu seguimiento.")
                            lines.append("\n**Próximo paso recomendado**")
                            lines.append(f"- Si quieres, iniciamos una nueva evaluación de {label} para comparar progreso.")
                            interpreted = "\n".join(lines).strip()

                        return Response(
                            {
                                'output': interpreted,
                                'quick_actions': [
                                    {
                                        'label': start_label,
                                        'type': 'message',
                                        'text': start_text,
                                        'payload': start_payload,
                                    },
                                    {'label': 'Servicios GoToGym (13)', 'type': 'services_menu', 'page': 'core'},
                                ],
                            }
                        )

                    return Response(
                        {
                            'output': f"No encuentro un resultado previo para **{label}**.",
                            'quick_actions': [
                                {
                                    'label': start_label,
                                    'type': 'message',
                                    'text': start_text,
                                    'payload': start_payload,
                                },
                                {'label': 'Servicios GoToGym (13)', 'type': 'services_menu', 'page': 'core'},
                            ],
                        }
                    )

                if want_start:
                    start_actions = spec.get('start_actions') if isinstance(spec.get('start_actions'), list) else []
                    if service_exp == 'exp-007_lifestyle':
                        return Response(
                            {
                                'output': (
                                    "**Estado de Hoy (Lifestyle)**\n"
                                    "En ~10 segundos evalúo tu sistema con señales de sueño, movimiento y carga diaria para definir tu foco del día sin sobreexigirte.\n\n"
                                    "No es un diagnóstico médico: es una guía de decisión para entrenar mejor hoy."
                                ),
                                'quick_actions': start_actions[:6],
                            }
                        )
                    if service_exp == 'exp-004_meal_plan':
                        return Response(
                            {
                                'output': (
                                    "**Menú Semanal Personalizado**\n"
                                    "En segundos te armo una propuesta semanal ajustada a tu objetivo calórico y adherencia, con opciones prácticas para ejecutar desde hoy.\n\n"
                                    "Luego puedes aplicar, ver lista de compras o cambiar comidas en 1 toque."
                                ),
                                'quick_actions': start_actions[:6],
                            }
                        )
                    return Response(
                        {
                            'output': f"Perfecto, iniciemos **{label}**.",
                            'quick_actions': start_actions[:6],
                        }
                    )

                return Response(
                    {
                        'output': f"¿Quieres ver tu último resultado o iniciar una evaluación nueva de **{label}**?",
                        'quick_actions': [
                            {
                                'label': 'Ver último resultado',
                                'type': 'message',
                                'text': f'Ver último resultado de {label}',
                                'payload': {'service_intent': {'experience': service_exp, 'action': 'show_last'}},
                            },
                            {
                                'label': ('Ejecutar evaluación metabólica' if service_exp == 'exp-003_metabolic_profile' else 'Iniciar evaluación nueva'),
                                'type': 'message',
                                'text': ('Ejecutar evaluación metabólica' if service_exp == 'exp-003_metabolic_profile' else f'Iniciar {label}'),
                                'payload': (
                                    {
                                        'metabolic_profile_request': {'start': True},
                                        'service_intent': {'experience': service_exp, 'action': 'start_new'},
                                    }
                                    if service_exp == 'exp-003_metabolic_profile'
                                    else {'service_intent': {'experience': service_exp, 'action': 'start_new'}}
                                ),
                            },
                            {'label': 'Cancelar', 'type': 'services_menu', 'page': 'core'},
                        ],
                    }
                )
        except Exception:
            pass

        # =========================
        # FAST-PATH (determinista): Exp-011 Vitalidad de la Piel
        # Evita que n8n/LLM responda primero y se pierdan los scores.
        # =========================
        try:
            msg_low = str(message or '').strip().lower()
            cs = getattr(user, 'coach_state', {}) or {} if user else {}

            def _skin_mode_active() -> bool:
                try:
                    if str(cs.get('health_mode') or '').strip().lower() != 'skin':
                        return False
                    until_raw = str(cs.get('health_mode_until') or '').strip()
                    if not until_raw:
                        return True
                    until_dt = datetime.fromisoformat(until_raw)
                    if until_dt.tzinfo is None:
                        until_dt = until_dt.replace(tzinfo=timezone.get_current_timezone())
                    return timezone.now() <= until_dt
                except Exception:
                    return bool(str(cs.get('health_mode') or '').strip().lower() == 'skin')

            want_skin = False
            try:
                if (('vitalidad' in msg_low) and ('piel' in msg_low or 'peil' in msg_low)):
                    want_skin = True
                if 'skin health' in msg_low or 'skincare' in msg_low:
                    want_skin = True
                if re.fullmatch(r"belleza\s*/\s*piel|belleza\s+piel", msg_low or ""):
                    want_skin = True
            except Exception:
                want_skin = False

            mode_active = _skin_mode_active()

            # Cancelar (server-side)
            try:
                if user and isinstance(request.data, dict) and request.data.get('skin_cancel') is True:
                    cs2 = dict(cs)
                    cs2['health_mode'] = ''
                    cs2['health_mode_until'] = ''
                    cs2.pop('skin_pending_attachment', None)
                    user.coach_state = cs2
                    user.coach_state_updated_at = timezone.now()
                    user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    return Response({'output': '**Vitalidad de la Piel**\nListo. Cerré el proceso.', 'skin_flow_stage': 'completed'})
            except Exception:
                pass

            # Post-análisis: ver hábitos sugeridos (sin pedir foto de nuevo)
            try:
                habits_requested = False
                try:
                    if user and isinstance(request.data, dict) and request.data.get('skin_habits_request') is True:
                        habits_requested = True
                    # Robustez: algunos clientes pueden enviar solo el texto del botón.
                    if (not habits_requested) and user:
                        if re.search(r"\b(ver\s+h[aá]bitos|h[aá]bitos\s+sugeridos)\b", msg_low or "") and re.search(r"\b(piel|skin)\b", msg_low or ""):
                            habits_requested = True
                except Exception:
                    habits_requested = False

                if user and habits_requested:
                    last_blob = cs.get('skin_last_result') if isinstance(cs.get('skin_last_result'), dict) else None
                    last_res = (last_blob or {}).get('result') if isinstance((last_blob or {}).get('result'), dict) else None

                    # Freshness guardrail: si es demasiado viejo, pedir que haga análisis primero.
                    try:
                        updated_at = str((last_blob or {}).get('updated_at') or '').strip()
                        if updated_at:
                            dt = datetime.fromisoformat(updated_at)
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.get_current_timezone())
                            if timezone.now() - dt > timedelta(hours=2):
                                last_res = None
                    except Exception:
                        pass

                    if not isinstance(last_res, dict):
                        # cerrar modo
                        try:
                            cs2 = dict(cs)
                            cs2['health_mode'] = ''
                            cs2['health_mode_until'] = ''
                            cs2.pop('skin_pending_attachment', None)
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                        except Exception:
                            pass

                        return Response({'output': "**Vitalidad de la Piel**\nPara darte sugerencias personalizadas necesito un análisis reciente. Haz el análisis con 1 foto y luego toca ‘Ver hábitos sugeridos para la piel’.", 'skin_flow_stage': 'completed'})

                    # Intentar hábitos vía n8n (más humanos), con fallback determinista.
                    habits_text = ""
                    try:
                        score = last_res.get('skin_health_score')
                        conf = last_res.get('confidence') if isinstance(last_res.get('confidence'), dict) else {}
                        conf_pct = None
                        try:
                            if conf.get('score') is not None:
                                conf_pct = int(round(float(conf.get('score')) * 100.0))
                        except Exception:
                            conf_pct = None

                        plan = last_res.get('recommendation_plan') if isinstance(last_res.get('recommendation_plan'), dict) else {}
                        prios = plan.get('priorities') if isinstance(plan.get('priorities'), list) else []
                        acts = plan.get('actions') if isinstance(plan.get('actions'), list) else []
                        ctx_sig = last_res.get('context_signals') if isinstance(last_res.get('context_signals'), dict) else {}
                        scores_all = last_res.get('scores') if isinstance(last_res.get('scores'), dict) else {}

                        prompt = (
                            "Genera sugerencias y hábitos personalizados basándote SOLO en este resultado.\n"
                            "IMPORTANTE: ya hay análisis; NO pidas otra foto ni más datos.\n"
                            "Responde SOLO en texto plano (sin HTML, sin <iframe>, sin markdown de iframe).\n"
                            "Formato: Mañana / Noche / Semanal. 5–8 bullets máximo.\n"
                            "Evita diagnósticos, preguntas al final, medicamentos o activos médicos y evita promesas.\n\n"
                            f"Vitalidad Score: {score}%" + (f" | Confianza de captura: {conf_pct}%" if conf_pct is not None else "") + "\n"
                            f"Subscores: {last_res.get('sub_scores')}\n"
                            f"Scores extra: {scores_all}\n"
                            f"Contexto: {ctx_sig}\n"
                            f"Prioridades (motor): {prios}\n"
                            f"Acciones (motor): {acts}\n"
                        )

                        n8n_payload = {
                            "chatInput": prompt,
                            "message": prompt,
                            "sessionId": session_id,
                            "username": (getattr(user, 'username', '') or ''),
                            "auth_header": auth_header,
                            "attachment": "",
                            "attachment_text": "",
                            "qaf": {"type": "exp-011_skin_habits", "result": last_res},
                            "qaf_skin_health": last_res,
                            "system_rules": {
                                "module": "exp-011_skin_habits",
                                "no_new_buttons": True,
                                "no_medical": True,
                            },
                        }
                        resp = _bench_n8n_post(request, n8n_url, n8n_payload, timeout=45)
                        if resp.status_code == 200:
                            try:
                                data = resp.json()
                            except Exception:
                                data = {"output": resp.text}
                            if isinstance(data, dict) and isinstance(data.get('output'), str):
                                habits_text = (data.get('output') or '').strip()
                            elif isinstance(data, str):
                                habits_text = data.strip()
                    except Exception:
                        habits_text = ""

                    # Si viene HTML/iframe, extraer texto de srcdoc y limpiar tags.
                    try:
                        raw_out = str(habits_text or '').strip()
                        if raw_out.lower().startswith('<iframe'):
                            try:
                                import html as _html
                                m = re.search(r"\bsrcdoc=(\"|')(.*?)(\1)", raw_out, flags=re.IGNORECASE | re.DOTALL)
                                if m:
                                    srcdoc = _html.unescape(m.group(2))
                                    srcdoc = re.sub(r"<\s*br\s*/?>", "\n", srcdoc, flags=re.IGNORECASE)
                                    srcdoc = re.sub(r"<[^>]+>", "", srcdoc)
                                    habits_text = (srcdoc or '').strip()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Evitar coletilla de pregunta final (n8n a veces la añade)
                    try:
                        lines = [ln.rstrip() for ln in str(habits_text or '').splitlines()]
                        while lines and not lines[-1].strip():
                            lines.pop()
                        if lines and ('¿' in lines[-1] or lines[-1].strip().endswith('?')):
                            lines.pop()
                        habits_text = "\n".join(lines).strip()
                    except Exception:
                        pass

                    # Guardrail: si n8n intenta pedir una foto, lo consideramos respuesta genérica y caemos a fallback.
                    try:
                        low_out = str(habits_text or '').lower()
                        if re.search(r"\b(env[ií]a|enviame|enviarme|adjunta|adjuntar)\b.*\bfoto\b", low_out):
                            habits_text = ""
                        if re.search(r"\bpara\s+poder\s+orientarte\s+mejor\b.*\bfoto\b", low_out):
                            habits_text = ""
                    except Exception:
                        pass

                    if not habits_text.strip():
                        _bench_event_note(request, fallback_used=True)
                        plan = last_res.get('recommendation_plan') if isinstance(last_res.get('recommendation_plan'), dict) else {}
                        prios = plan.get('priorities') if isinstance(plan.get('priorities'), list) else []
                        acts = plan.get('actions') if isinstance(plan.get('actions'), list) else []
                        prios = [str(x).strip() for x in prios if str(x).strip()]
                        acts = [str(x).strip() for x in acts if str(x).strip()]

                        out_lines = [
                            "**Vitalidad de la Piel**",
                            "De acuerdo a tus resultados del análisis, estas son tus sugerencias personalizadas a realizar:",
                            "",
                            "**✅ Hábitos sugeridos para la piel**",
                        ]
                        if prios:
                            out_lines.append("\n**🎯 En qué enfocarte**")
                            for i, p in enumerate(prios[:3], start=1):
                                out_lines.append(f"- Prioridad {i}: {p}")
                        if acts:
                            out_lines.append("\n**✅ Acciones simples**")
                            for a in acts[:5]:
                                out_lines.append(f"- {a}")
                        out_lines.append("\nSi quieres medir progreso real, repite con la misma luz/encuadre 1 vez por semana.")
                        habits_text = "\n".join(out_lines).strip()

                    # cerrar modo
                    try:
                        cs2 = dict(cs)
                        cs2['health_mode'] = ''
                        cs2['health_mode_until'] = ''
                        cs2.pop('skin_pending_attachment', None)
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    # Prefacio consistente (incluso si viene de n8n)
                    try:
                        if habits_text.strip() and ("de acuerdo a tus resultados" not in habits_text.lower()):
                            habits_text = (
                                "**Vitalidad de la Piel**\n"
                                "De acuerdo a tus resultados del análisis, estas son tus sugerencias personalizadas a realizar:\n\n"
                                + habits_text.strip()
                            )
                    except Exception:
                        pass

                    return Response({'output': habits_text, 'skin_flow_stage': 'completed'})
            except Exception:
                pass

            # Si el usuario está en el modo (o lo pidió) y NO hay foto aún, devolver CTAs siempre.
            if user and (want_skin or mode_active) and not attachment_url:
                # Activar modo si venía por intención
                if want_skin and not mode_active:
                    try:
                        cs2 = dict(cs)
                        cs2['health_mode'] = 'skin'
                        cs2['health_mode_until'] = (timezone.now() + timedelta(minutes=15)).isoformat()
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                        cs = cs2
                        mode_active = True
                    except Exception:
                        pass

                out = (
                    "**Vitalidad de la Piel**\n"
                    "**✅ Empecemos**\n"
                    "Con 1 foto puedo darte una lectura visual con métricas + acciones simples (sin diagnósticos).\n\n"
                    "**📷 Para que mida bien:** luz natural, sin filtros, sin contraluz, rostro centrado."
                )
                return Response(
                    {
                        'output': out,
                        'skin_flow_stage': 'need_photo',
                        'quick_actions': [
                            {'label': 'Tomar foto', 'type': 'open_camera'},
                            {'label': 'Adjuntar foto', 'type': 'open_attach'},
                            {'label': 'Cancelar', 'type': 'skin_cancel'},
                        ],
                    }
                )

            # Si hay foto y el usuario está en el modo (o lo pidió), analizar ya.
            if user and attachment_url and (want_skin or mode_active):
                # Asegurar modo activo
                if want_skin and not mode_active:
                    try:
                        cs2 = dict(cs)
                        cs2['health_mode'] = 'skin'
                        cs2['health_mode_until'] = (timezone.now() + timedelta(minutes=15)).isoformat()
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                        cs = cs2
                        mode_active = True
                    except Exception:
                        pass

                # Descargar bytes solo si es attachment del usuario.
                image_bytes = None
                image_content_type = None
                try:
                    normalized_url = _normalize_attachment_url(str(attachment_url))
                    container_name, blob_name = _extract_blob_ref_from_url(normalized_url)
                    if container_name == _chat_attachment_container() and blob_name:
                        blob_name = _resolve_blob_name(container_name, blob_name) or blob_name
                        safe_username = user.username.replace('/', '_')
                        allowed = bool(blob_name.startswith(f"{safe_username}/")) or _is_signed_chat_attachment_url(normalized_url)
                        if allowed:
                            max_bytes = int(os.getenv('CHAT_VISION_MAX_BYTES', str(4 * 1024 * 1024)) or (4 * 1024 * 1024))
                            resp = requests.get(normalized_url, timeout=20)
                            if resp.status_code == 200 and resp.content and len(resp.content) <= max_bytes:
                                image_bytes = resp.content
                                _bench_event_note(request, attachment_bytes=len(image_bytes))
                                image_content_type = (resp.headers.get('Content-Type') or 'image/jpeg')
                except Exception:
                    image_bytes = None

                if not image_bytes:
                    return Response(
                        {
                            'output': '**Vitalidad de la Piel**\nNo pude usar la foto (permiso/tamaño). Intenta adjuntarla de nuevo.',
                            'skin_flow_stage': 'need_photo',
                            'quick_actions': [
                                {'label': 'Tomar foto', 'type': 'open_camera'},
                                {'label': 'Adjuntar foto', 'type': 'open_attach'},
                                {'label': 'Cancelar', 'type': 'skin_cancel'},
                            ],
                        }
                    )

                # Contexto (lifestyle + auto-report)
                ctx = {}
                try:
                    lifestyle_last = (cs.get('lifestyle_last') or {}).get('result') if isinstance(cs.get('lifestyle_last'), dict) else None
                    if isinstance(lifestyle_last, dict):
                        sig = lifestyle_last.get('signals') if isinstance(lifestyle_last.get('signals'), dict) else {}
                        ctx['sleep_minutes'] = (sig.get('sleep') or {}).get('value')
                        ctx['steps'] = (sig.get('steps') or {}).get('value')
                        try:
                            stress_score01 = (sig.get('stress_inv') or {}).get('score01')
                            if stress_score01 is not None:
                                s = float(stress_score01)
                                s = max(0.0, min(1.0, s))
                                ctx['stress_1_5'] = round((1.0 - s) * 4.0 + 1.0, 1)
                        except Exception:
                            pass
                except Exception:
                    ctx = {}
                try:
                    sc = cs.get('skin_context') if isinstance(cs.get('skin_context'), dict) else {}
                    if isinstance(sc, dict):
                        for k in ('water_liters', 'stress_1_5', 'movement_1_5', 'sun_minutes', 'steps'):
                            if sc.get(k) is not None:
                                ctx[k] = sc.get(k)
                except Exception:
                    pass

                # Baseline semanal
                week_id_now = _week_id()
                weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
                baseline = None
                try:
                    sh = weekly_state.get('skin_health') if isinstance(weekly_state.get('skin_health'), dict) else {}
                    prev_keys = [k for k in sh.keys() if isinstance(k, str) and k != week_id_now]
                    if prev_keys:
                        prev_key = sorted(prev_keys)[-1]
                        prev_row = sh.get(prev_key)
                        if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                            baseline = prev_row.get('result')
                except Exception:
                    baseline = None

                from .qaf_skin_health.engine import evaluate_skin_health, render_professional_summary
                res = evaluate_skin_health(image_bytes=image_bytes, content_type=image_content_type, context=ctx, baseline=baseline).payload
                try:
                    if isinstance(res, dict):
                        res = dict(res)
                        res['user_display_name'] = (getattr(user, 'full_name', None) or getattr(user, 'username', '') or '').strip()
                except Exception:
                    pass
                text = render_professional_summary(res)

                # Persistir semanal
                try:
                    ws2 = dict(weekly_state)
                    sh = ws2.get('skin_health') if isinstance(ws2.get('skin_health'), dict) else {}
                    sh2 = dict(sh)
                    sh2[week_id_now] = {'result': res, 'updated_at': timezone.now().isoformat(), 'week_id': week_id_now}
                    ws2['skin_health'] = sh2
                    user.coach_weekly_state = ws2
                    user.coach_weekly_updated_at = timezone.now()
                    user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                except Exception:
                    pass

                # Guardar último para CTA de hábitos
                try:
                    cs2 = dict(cs)
                    cs2['skin_last_result'] = {'result': res, 'updated_at': timezone.now().isoformat()}
                    user.coach_state = cs2
                    user.coach_state_updated_at = timezone.now()
                    user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                except Exception:
                    pass

                # CTA post-análisis
                qas = []
                try:
                    if isinstance(res, dict) and str(res.get('decision') or '').strip().lower() != 'accepted':
                        qas = [
                            {'label': 'Tomar foto', 'type': 'open_camera'},
                            {'label': 'Adjuntar foto', 'type': 'open_attach'},
                            {'label': 'Cancelar', 'type': 'skin_cancel'},
                        ]
                    else:
                        qas = [
                            {'label': 'Ver hábitos sugeridos para la piel', 'type': 'message', 'text': 'Ver hábitos sugeridos para la piel', 'payload': {'skin_habits_request': True}},
                            {'label': 'Finalizar', 'type': 'skin_cancel'},
                        ]
                except Exception:
                    qas = [
                        {'label': 'Cancelar', 'type': 'skin_cancel'},
                    ]

                coach_output = ''
                try:
                    skin_context_text = (text or '').strip()
                    if skin_context_text:
                        skin_context_text = f"[VITALIDAD DE LA PIEL]\n{skin_context_text}"

                    n8n_payload = {
                        'chatInput': str(message or 'Vitalidad de la Piel').strip() or 'Vitalidad de la Piel',
                        'message': str(message or 'Vitalidad de la Piel').strip() or 'Vitalidad de la Piel',
                        'sessionId': session_id,
                        'username': username_for_payload,
                        'auth_header': auth_header,
                        'attachment': '',
                        'attachment_text': skin_context_text,
                        'attachment_text_diagnostic': '',
                        'qaf': {'type': 'exp-011_skin_health', 'result': res},
                        'qaf_skin_health': res,
                        'system_rules': {
                            'module': 'exp-011_skin_health',
                            'llm_role': 'coach_full_response',
                            'respond_from_qaf_result': True,
                        },
                    }
                    n8n_resp = _bench_n8n_post(request, n8n_url, n8n_payload, timeout=45)
                    if n8n_resp.status_code == 200:
                        try:
                            n8n_data = n8n_resp.json()
                        except Exception:
                            n8n_data = {'output': n8n_resp.text}
                        if isinstance(n8n_data, dict):
                            coach_output = str(n8n_data.get('output') or '').strip()
                        elif isinstance(n8n_data, str):
                            coach_output = n8n_data.strip()
                except Exception:
                    coach_output = ''

                final_output = coach_output or (
                    "Listo. Ya procesé tu análisis de Vitalidad de la Piel, "
                    "pero no pude generar la respuesta completa del Quantum Coach en este intento. "
                    "Inténtalo de nuevo en unos segundos."
                )
                return Response(
                    _attach_wow_event_payload(
                        user,
                        {
                            'output': final_output,
                            'qaf_skin_health': res,
                            'quick_actions': qas,
                            'skin_flow_stage': 'analysis_done',
                        },
                        event_key='qaf_skin_health',
                        label='Vitalidad de la piel analizada',
                    )
                )
        except Exception as ex:
            print(f"Vitalidad fast-path warning: {ex}")

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
                normalized_url = _normalize_attachment_url(str(attachment_url))
                container_name, blob_name = _extract_blob_ref_from_url(normalized_url)
                if container_name == _chat_attachment_container() and blob_name:
                    blob_name = _resolve_blob_name(container_name, blob_name) or blob_name
                    safe_username = user.username.replace("/", "_")
                    allowed = bool(blob_name.startswith(f"{safe_username}/")) or _is_signed_chat_attachment_url(normalized_url)
                    if allowed:
                        max_bytes = int(os.getenv('CHAT_ATTACHMENT_MAX_BYTES', str(15 * 1024 * 1024)) or (15 * 1024 * 1024))
                        resp = requests.get(normalized_url, timeout=20)
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
        vision_route_bias_guard = False
        vision_experience_activated = ""

        # Flags de intención (evitar sesgo por inyecciones no pedidas)
        try:
            _msg_low_flags = str(message or '').strip().lower()
        except Exception:
            _msg_low_flags = ''
        user_asked_metabolic = bool(re.search(r"\b(perfil\s+metab|metab[oó]lic|tdee|tmb|kcal|calor[ií]as)\b", _msg_low_flags or ""))
        try:
            if isinstance(request.data, dict) and isinstance(request.data.get('metabolic_profile_request'), dict):
                user_asked_metabolic = True
        except Exception:
            pass
        user_explicit_image_experience = bool(re.search(
            r"\b(vitalidad\s+de\s+la\s+pi?e?l|skin\s*health|skincare|postura|posture|arquitectura\s+corporal|proporci[oó]n|shape|presence|presencia|progreso\s+muscular|medici[oó]n\s+muscular)\b",
            _msg_low_flags or "",
        ))

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
        motivation_requested = False

        # Exp-009 (progresión): resultado + texto + quick-actions
        progression_result = None
        progression_text_for_output_override = ""
        progression_quick_actions_out: list[dict[str, Any]] = []

        # Guardrail UX: si el usuario está en un flujo de Postura, no mezclar check-ins/metabólico.
        try:
            pr0 = request.data.get('posture_request') if isinstance(request.data, dict) else None
            msg_low0 = str(message or '').lower()
            mr0 = request.data.get('muscle_measure_request') if isinstance(request.data, dict) else None
            ppr0 = request.data.get('posture_proportion_request') if isinstance(request.data, dict) else None
            suppress_weekly_checkins = bool(
                isinstance(pr0, dict)
                or isinstance(mr0, dict)
                or isinstance(ppr0, dict)
                or re.search(r"\b(postura|posture|proporci[oó]n|proporcion|proportion|m[uú]sculo|musculo|muscle|shape|presence|presencia)\b", msg_low0)
            )
            posture_requested = bool(isinstance(pr0, dict))
        except Exception:
            suppress_weekly_checkins = False
            posture_requested = False

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

        # 0.2.b) Exp-010: Medición muscular (solo fotografía; 1..4 vistas opcionales)
        try:
            if user and isinstance(request.data, dict):
                mm_req = request.data.get('muscle_measure_request')

                # Intención por texto (activación intuitiva)
                msg_low = str(message or '').strip().lower()
                want_mm = False
                if isinstance(mm_req, dict):
                    want_mm = True
                else:
                    # Frases naturales: medir/comparar progreso muscular, semana pasada, y focos comunes.
                    if re.search(
                        r"\b("
                        r"medir\s+m[uú]sculo|medici[oó]n\s+muscular|"
                        r"progreso\s+muscular|medici[oó]n\s+del\s+progreso\s+muscular|"
                        r"comparar\s+m[uú]sculo|comparar\s+musculo|"
                        r"medici[oó]n\s+del\s+progreso|"
                        r"semana\s+pasada|vs\s+la\s+semana\s+pasada|"
                        r"b[ií]ceps|tr[ií]ceps|brazo\w*|"
                        r"gl[uú]te\w*|pierna\w*|cu[aá]driceps|"
                        r"hombro\w*|espalda\w*"
                        r")\b",
                        msg_low or "",
                    ):
                        want_mm = True

                # Iniciar flujo desde quick-action / activación por texto
                # UX: primero elegir enfoque (menos botones al tiempo) y luego guiar la foto.
                if (not isinstance(mm_req, dict)) and want_mm:
                    out = (
                        "[MEDICIÓN DEL PROGRESO MUSCULAR]\n"
                        "Vamos a crear tu **tablero de evolución** y compararlo con tu última medición.\n\n"
                        "Para que funcione de verdad:\n"
                        "- Misma luz, misma distancia (2–3m), misma altura de cámara\n"
                        "- Cuerpo completo y sin recortar hombros/cadera\n\n"
                        "Primero dime qué quieres medir hoy:"
                    )
                    return Response(
                        {
                            'output': out,
                            'quick_actions': [
                                {'label': 'Progreso general', 'type': 'message', 'text': 'Progreso general', 'payload': {'muscle_measure_request': {'focus': 'general'}}},
                                {'label': 'Bíceps', 'type': 'message', 'text': 'Bíceps', 'payload': {'muscle_measure_request': {'focus': 'biceps'}}},
                                {'label': 'Glúteos', 'type': 'message', 'text': 'Glúteos', 'payload': {'muscle_measure_request': {'focus': 'glutes'}}},
                                {'label': 'Abdomen', 'type': 'message', 'text': 'Abdomen', 'payload': {'muscle_measure_request': {'focus': 'abs'}}},
                                {'label': 'Cancelar', 'type': 'muscle_cancel'},
                            ][:6],
                        }
                    )

                if isinstance(mm_req, dict):
                    from .qaf_muscle_measure.engine import evaluate_muscle_measure, render_professional_summary

                    week_id_now = _week_id()
                    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

                    baseline = None
                    baseline_source = None
                    try:
                        mm = weekly_state.get('muscle_measure') if isinstance(weekly_state.get('muscle_measure'), dict) else {}
                        prev_keys = [k for k in mm.keys() if isinstance(k, str) and k != week_id_now]
                        if prev_keys:
                            prev_key = sorted(prev_keys)[-1]
                            prev_row = mm.get(prev_key)
                            if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                                baseline = prev_row.get('result')
                                baseline_source = 'last_week'
                    except Exception:
                        baseline = None
                        baseline_source = None

                    poses = mm_req.get('poses') if isinstance(mm_req.get('poses'), dict) else {}
                    focus = mm_req.get('focus') if isinstance(mm_req.get('focus'), str) else None

                    # Memoria suave: última medición por enfoque (permite comparar incluso dentro de la misma semana)
                    try:
                        fx = str(focus or '').strip().lower() or None
                        if fx:
                            fx_key = 'abs' if fx in ('abs', 'abdomen', 'abdominales') else ('biceps' if fx in ('biceps', 'bíceps', 'bicep') else ('glutes' if fx in ('glutes', 'gluteos', 'glúteos') else fx))
                            cs0 = getattr(user, 'coach_state', {}) or {}
                            mem0 = cs0.get('muscle_measure_memory') if isinstance(cs0.get('muscle_measure_memory'), dict) else {}
                            last_by_focus = mem0.get('last_by_focus') if isinstance(mem0.get('last_by_focus'), dict) else {}
                            row = last_by_focus.get(fx_key) if isinstance(last_by_focus.get(fx_key), dict) else None
                            if isinstance(row, dict) and isinstance(row.get('result'), dict):
                                baseline = row.get('result')
                                baseline_source = 'last_same_focus'
                    except Exception:
                        pass

                    # Si el usuario solo seleccionó un foco (sin fotos aún), guiar captura con copy específico.
                    try:
                        if (not poses) and focus:
                            fx = str(focus).strip().lower()
                            fx_label = 'abdomen' if fx in ('abs', 'abdomen') else ('bíceps' if fx in ('biceps', 'bíceps') else ('glúteos' if fx in ('glutes', 'gluteos', 'glúteos') else fx))
                            guide = (
                                f"[MEDICIÓN DEL PROGRESO MUSCULAR — Enfoque {fx_label.upper()}]\n"
                                "Perfecto. Vamos paso a paso (una foto a la vez) para que el resultado sea comparable.\n\n"
                                "Cómo tomar la foto:\n"
                                "- Buena luz + fondo limpio\n"
                                "- Cámara a 2–3m, altura del pecho\n"
                                "- Cuerpo completo (pies a cabeza)\n"
                                "- Sin recortar hombros/cadera\n\n"
                                "Empecemos con **frente relajado**."
                            )
                            return Response(
                                {
                                    'output': guide,
                                    'quick_actions': [
                                        {'label': 'Tomar foto frente', 'type': 'muscle_capture', 'view': 'front_relaxed', 'source': 'camera'},
                                        {'label': 'Adjuntar foto frente', 'type': 'muscle_capture', 'view': 'front_relaxed', 'source': 'attach'},
                                        {'label': 'Cancelar', 'type': 'muscle_cancel'},
                                    ],
                                }
                            )
                    except Exception:
                        pass

                    height_cm = None
                    try:
                        height_cm = float(getattr(user, 'height', None)) if getattr(user, 'height', None) is not None else None
                    except Exception:
                        height_cm = None

                    res = evaluate_muscle_measure({'poses': poses, 'baseline': baseline, 'focus': focus, 'height_cm': height_cm}).payload

                    # marcar fuente del baseline para copy
                    try:
                        if isinstance(res, dict) and baseline_source:
                            prog0 = res.get('progress') if isinstance(res.get('progress'), dict) else {}
                            res = {**res, 'progress': {**prog0, 'baseline_source': baseline_source}}
                    except Exception:
                        pass

                    # UX: nombre amigable para saludo
                    try:
                        display = (getattr(user, 'full_name', '') or '').strip() or (getattr(user, 'username', '') or '').strip()
                        if display and isinstance(res, dict):
                            res = {**res, 'user_display_name': display}
                    except Exception:
                        pass
                    text = render_professional_summary(res)

                    # Persistir por semana
                    try:
                        ws2 = dict(weekly_state)
                        mm = ws2.get('muscle_measure') if isinstance(ws2.get('muscle_measure'), dict) else {}
                        mm2 = dict(mm)
                        mm2[week_id_now] = {'result': res, 'updated_at': timezone.now().isoformat(), 'week_id': week_id_now}
                        ws2['muscle_measure'] = mm2
                        user.coach_weekly_state = ws2
                        user.coach_weekly_updated_at = timezone.now()
                        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                    except Exception:
                        pass

                    # Memoria suave (coach_state): última medición global + por enfoque
                    try:
                        if isinstance(res, dict) and res.get('decision') == 'accepted':
                            cs0 = getattr(user, 'coach_state', {}) or {}
                            cs2 = dict(cs0)
                            mem0 = cs2.get('muscle_measure_memory') if isinstance(cs2.get('muscle_measure_memory'), dict) else {}
                            mem2 = dict(mem0)
                            last_by_focus = mem2.get('last_by_focus') if isinstance(mem2.get('last_by_focus'), dict) else {}
                            last_by_focus2 = dict(last_by_focus)

                            fx = str(focus or '').strip().lower() or 'general'
                            fx_key = 'abs' if fx in ('abs', 'abdomen', 'abdominales') else ('biceps' if fx in ('biceps', 'bíceps', 'bicep') else ('glutes' if fx in ('glutes', 'gluteos', 'glúteos') else fx))
                            last_by_focus2[fx_key] = {'result': res, 'updated_at': timezone.now().isoformat(), 'focus': fx_key}

                            # Mantener un set pequeño de focos
                            for k in list(last_by_focus2.keys()):
                                if str(k) not in ('general', 'biceps', 'glutes', 'abs'):
                                    last_by_focus2.pop(k, None)

                            mem2['last_by_focus'] = last_by_focus2
                            mem2['last'] = {'result': res, 'updated_at': timezone.now().isoformat(), 'focus': fx_key}
                            cs2['muscle_measure_memory'] = mem2
                            cs2['muscle_measure_last'] = {'result': res, 'updated_at': timezone.now().isoformat(), 'focus': fx_key}

                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    # Si el motor pide repetir, ofrecer retomar captura
                    qas = []
                    try:
                        if isinstance(res, dict) and res.get('decision') != 'accepted':
                            qas = [
                                {'label': 'Repetir frente', 'type': 'muscle_capture', 'view': 'front_relaxed', 'source': 'camera'},
                                {'label': 'Adjuntar frente', 'type': 'muscle_capture', 'view': 'front_relaxed', 'source': 'attach'},
                                {'label': 'Cancelar', 'type': 'muscle_cancel'},
                            ]
                        else:
                            fx = str(focus or '').strip().lower()
                            has_flex = bool(isinstance(poses.get('front_flex'), dict) and isinstance(poses.get('front_flex').get('keypoints'), list) and len(poses.get('front_flex').get('keypoints')))

                            if fx in ('biceps', 'bíceps', 'bicep') and not has_flex:
                                # Para bíceps necesitamos flex. Ofrecerlo directo.
                                qas = [
                                    {'label': 'Tomar frente flex suave', 'type': 'muscle_capture', 'view': 'front_flex', 'source': 'camera'},
                                    {'label': 'Adjuntar frente flex suave', 'type': 'muscle_capture', 'view': 'front_flex', 'source': 'attach'},
                                    {'label': 'Cancelar', 'type': 'muscle_cancel'},
                                ]
                            else:
                                # CTAs de foco (sin UI nueva)
                                qas = [
                                    {'label': 'Enfocar bíceps', 'type': 'message', 'text': 'Enfocar bíceps', 'payload': {'muscle_measure_request': {'poses': poses, 'focus': 'biceps'}}},
                                    {'label': 'Enfocar glúteos', 'type': 'message', 'text': 'Enfocar glúteos', 'payload': {'muscle_measure_request': {'poses': poses, 'focus': 'glutes'}}},
                                    {'label': 'Enfocar abdomen', 'type': 'message', 'text': 'Enfocar abdomen', 'payload': {'muscle_measure_request': {'poses': poses, 'focus': 'abs'}}},
                                ]
                    except Exception:
                        qas = []

                    return Response(
                        _attach_wow_event_payload(
                            user,
                            {
                                'output': text or 'Medición lista.',
                                'qaf_muscle_measure': res,
                                'quick_actions': qas,
                            },
                            event_key='qaf_muscle_measure',
                            label='Medición muscular',
                        )
                    )
        except Exception as ex:
            print(f"QAF muscle measure warning: {ex}")

        # 0.2.c.1) Exp-013: Arquitectura Corporal (antes Postura & Proporción) (3 fotos; keypoints 2D)
        try:
            if user and isinstance(request.data, dict):
                pp_req = request.data.get('posture_proportion_request')
                pp_intent = request.data.get('posture_proportion_intent')
                pp_intent_action = ""
                if isinstance(pp_intent, dict):
                    pp_intent_action = str(pp_intent.get('action') or '').strip().lower()

                week_id_now = _week_id()
                weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

                msg_low = str(message or '').strip().lower()
                # Activación intuitiva: tolera frases largas y acentos.
                msg_norm = msg_low
                try:
                    import unicodedata

                    msg_norm = "".join(
                        ch for ch in unicodedata.normalize('NFKD', msg_low) if not unicodedata.combining(ch)
                    )
                except Exception:
                    msg_norm = msg_low
                msg_norm = re.sub(r"\s+", " ", msg_norm or "").strip()

                explicit = bool(
                    re.search(
                        r"\b(arquitectura corporal|postura\s*&\s*proporcion|postura\s+y\s+proporcion|posture\s*&\s*proportion|posture\s+and\s+proportion)\b",
                        msg_norm,
                    )
                )

                natural = bool(
                    re.search(
                        r"\b("
                        r"mejorar postura|corregir postura|alineacion|alinear|eje|"
                        r"hombro\w* caid\w*|cadera\w*|pelvis|asimetr\w*|"
                        r"cabeza adelant\w*|cuello adelant\w*|"
                        r"hombros redonde\w*|escapul\w*|"
                        r"distribu\w* peso|apoyo desigual|"
                        r"perfil derecho|frente relajado"
                        r")\b",
                        msg_norm,
                    )
                )

                start_verb = bool(
                    re.search(
                        r"\b(iniciar|inicia|empecemos|empezar|arrancar|activar|hazme\s+una\s+nueva|nueva\s+evaluaci[oó]n|"
                        r"nueva\s+medici[oó]n|volver\s+a\s+medir|repetir\s+evaluaci[oó]n|tomar\s+fotos|adjuntar\s+fotos)\b",
                        msg_norm,
                    )
                )

                read_verb = bool(
                    re.search(
                        r"\b(muestrame|mu[eé]strame|mostrar|mostrarme|dame|ver|consulta|ensename|enséñame|quiero\s+ver)\b",
                        msg_norm,
                    )
                )
                result_noun = bool(
                    re.search(
                        r"\b(resultado|resultados|ultima\s+evaluacion|[uú]ltima\s+evaluaci[oó]n|evaluacion\s+anterior|evaluaci[oó]n\s+anterior)\b",
                        msg_norm,
                    )
                )
                posture_domain = bool(
                    re.search(
                        r"\b(arquitectura\s+corporal|postura|proporci[oó]n|proporcion|shape|presence|presencia|musculo|m[uú]sculo)\b",
                        msg_norm,
                    )
                )

                want_pp_results = posture_domain and (result_noun or (read_verb and ("ultima" in msg_norm or "última" in msg_low)))
                if pp_intent_action in ('show_last', 'view_last', 'last_result'):
                    want_pp_results = True

                want_pp = explicit or (start_verb and posture_domain)
                if pp_intent_action in ('start_new', 'new_eval', 'start'):
                    want_pp = True

                want_confirmation = (
                    (not isinstance(pp_req, dict))
                    and posture_domain
                    and (not want_pp_results)
                    and (not want_pp)
                    and (natural or read_verb or result_noun)
                )

                if want_confirmation:
                    return Response(
                        {
                            'output': "¿Quieres ver tu último resultado o iniciar una evaluación nueva de Arquitectura Corporal?",
                            'quick_actions': [
                                {
                                    'label': 'Ver último resultado',
                                    'type': 'message',
                                    'text': 'Ver último resultado de Arquitectura Corporal',
                                    'payload': {'posture_proportion_intent': {'action': 'show_last'}},
                                },
                                {
                                    'label': 'Iniciar evaluación nueva',
                                    'type': 'message',
                                    'text': 'Iniciar evaluación nueva de Arquitectura Corporal',
                                    'payload': {'posture_proportion_intent': {'action': 'start_new'}},
                                },
                                {'label': 'Cancelar', 'type': 'pp_cancel'},
                            ],
                        }
                    )

                if (not isinstance(pp_req, dict)) and want_pp_results:
                    from .qaf_posture_proportion.engine import render_professional_summary

                    latest_result = None
                    latest_week_id = ""
                    latest_updated_at = ""
                    try:
                        pp_map = weekly_state.get('posture_proportion') if isinstance(weekly_state.get('posture_proportion'), dict) else {}
                        if isinstance(pp_map, dict) and pp_map:
                            candidate_keys = sorted([k for k in pp_map.keys() if isinstance(k, str)])
                            for key in reversed(candidate_keys):
                                row = pp_map.get(key)
                                if not isinstance(row, dict):
                                    continue
                                r = row.get('result') if isinstance(row.get('result'), dict) else None
                                if isinstance(r, dict):
                                    latest_result = r
                                    latest_week_id = key
                                    latest_updated_at = str(row.get('updated_at') or '').strip()
                                    break
                    except Exception:
                        latest_result = None

                    if isinstance(latest_result, dict):
                        summary = render_professional_summary(latest_result)
                        header_lines = ["**Tu última evaluación de Arquitectura Corporal**"]
                        if latest_week_id:
                            header_lines.append(f"Semana: {latest_week_id}")
                        if latest_updated_at:
                            header_lines.append(f"Actualizada: {latest_updated_at}")

                        return Response(
                            {
                                'output': "\n".join(header_lines) + "\n\n" + (summary or "No encontré detalles de la evaluación."),
                                'qaf_posture_proportion': latest_result,
                                'quick_actions': [
                                    {'label': 'Finalizar', 'type': 'pp_cancel'},
                                ],
                            }
                        )

                    return Response(
                        {
                            'output': (
                                "No encuentro una evaluación previa de Arquitectura Corporal para mostrarte.\n\n"
                                "Si quieres, la hacemos ahora y en menos de un minuto te doy el resultado."
                            ),
                            'quick_actions': [
                                {
                                    'label': 'Iniciar evaluación nueva',
                                    'type': 'message',
                                    'text': 'Iniciar evaluación nueva de Arquitectura Corporal',
                                    'payload': {'posture_proportion_intent': {'action': 'start_new'}},
                                },
                                {'label': 'Tomar foto frente', 'type': 'pp_capture', 'view': 'front_relaxed', 'source': 'camera'},
                                {'label': 'Adjuntar foto frente', 'type': 'pp_capture', 'view': 'front_relaxed', 'source': 'attach'},
                                {'label': 'Cancelar', 'type': 'pp_cancel'},
                            ],
                        }
                    )


                if (not isinstance(pp_req, dict)) and want_pp:
                    out = (
                        "**Arquitectura Corporal**\n\n"
                        "Experiencia de nivel profesional: traduzco tu alineación en decisiones prácticas para mejorar tu estabilidad, eficiencia y presencia.\n\n"
                        "Trabajo con puntos de referencia visuales (keypoints 2D).\n"
                        "No son medidas en centímetros ni constituyen diagnóstico médico.\n\n"
                        "Para darte un resultado preciso necesito 3 vistas guiadas:\n\n"
                        "• Frente relajado (obligatoria)\n"
                        "• Perfil derecho (obligatoria)\n"
                        "• Espalda (opcional, recomendada para máxima precisión)\n\n"
                        "**Empecemos con frente relajado:**\n\n"
                        "Cuerpo completo (pies a cabeza)\n"
                        "Buena luz frontal\n"
                        "Cámara a la altura del pecho\n"
                        "Distancia de 2–3 metros\n\n"
                        "En menos de un minuto te diré qué ajustar hoy y qué trabajar esta semana para optimizar tu estructura."
                    )
                    return Response(
                        {
                            'output': out,
                            'quick_actions': [
                                {'label': 'Tomar foto frente', 'type': 'pp_capture', 'view': 'front_relaxed', 'source': 'camera'},
                                {'label': 'Adjuntar foto frente', 'type': 'pp_capture', 'view': 'front_relaxed', 'source': 'attach'},
                                {'label': 'Cancelar', 'type': 'pp_cancel'},
                            ],
                        }
                    )

                if isinstance(pp_req, dict):
                    from .qaf_posture_proportion.engine import evaluate_posture_proportion, render_professional_summary

                    baseline = None
                    try:
                        pp = weekly_state.get('posture_proportion') if isinstance(weekly_state.get('posture_proportion'), dict) else {}
                        prev_keys = [k for k in pp.keys() if isinstance(k, str) and k != week_id_now]
                        if prev_keys:
                            prev_key = sorted(prev_keys)[-1]
                            prev_row = pp.get(prev_key)
                            if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                                baseline = prev_row.get('result')
                    except Exception:
                        baseline = None

                    poses = pp_req.get('poses') if isinstance(pp_req.get('poses'), dict) else {}
                    res = evaluate_posture_proportion({'poses': poses, 'baseline': baseline}).payload
                    text = render_professional_summary(res)

                    # Post-proceso (lujo): delegar explicación al Quantum Coach (n8n) usando el resultado QAF.
                    # Guardrails: no pedir nuevas fotos/datos, solo texto plano, sin HTML/iframe.
                    try:
                        qc_text = ""
                        if isinstance(res, dict) and str(res.get('decision') or '').strip().lower() == 'accepted':
                            vars_ = res.get('variables') if isinstance(res.get('variables'), dict) else {}
                            patterns = res.get('patterns') if isinstance(res.get('patterns'), list) else []
                            imm = res.get('immediate_corrections') if isinstance(res.get('immediate_corrections'), list) else []
                            weekly = res.get('weekly_adjustment') if isinstance(res.get('weekly_adjustment'), dict) else {}
                            conf = res.get('confidence') if isinstance(res.get('confidence'), dict) else {}
                            conf_pct = None
                            try:
                                if conf.get('score') is not None:
                                    conf_pct = int(round(float(conf.get('score')) * 100.0))
                            except Exception:
                                conf_pct = None

                            prompt = (
                                "Eres Quantum Coach. Tu trabajo es convertir un reporte QAF técnico en una explicación clara, premium y útil para una persona sin conocimientos.\n"
                                "IMPORTANTE: ya existe análisis; NO pidas más fotos ni más datos.\n"
                                "No diagnóstico médico. No promesas. No centímetros.\n"
                                "Responde SOLO en texto plano (sin HTML, sin <iframe>).\n\n"
                                "Estructura obligatoria (en este orden):\n"
                                "1) Título: Arquitectura Corporal\n"
                                "2) Resumen de 2–3 líneas (qué significa y por qué importa)\n"
                                "3) Lo que detecté (3 bullets máximo, sustentado en métricas)\n"
                                "4) Qué hago hoy (4–6 bullets): micro-ajustes + 2 ejercicios extra con series/reps/tiempo\n"
                                "5) Plan semanal (3 sesiones): 4–6 bullets con foco y progresión simple\n"
                                "6) Fit/ropa (2–3 bullets): cómo la postura afecta cómo cae ropa\n"
                                "7) Seguimiento: cómo repetir fotos y qué cambio esperar\n\n"
                                "Tono: lujo, claridad, cero jerga. Explica los porcentajes como ‘nivel de estabilidad/limpieza del eje’.\n\n"
                                f"Confianza de captura: {conf_pct}%\n"
                                f"Índices: {json.dumps({k: vars_.get(k) for k in ('postural_efficiency_score','posture_score','proportion_score','alignment_silhouette_index')}, ensure_ascii=False)}\n"
                                f"Señales: {json.dumps({'pose_line': vars_.get('pose_line'), 'symmetry_monitor': vars_.get('symmetry_monitor')}, ensure_ascii=False)}\n"
                                f"Patrones: {patterns}\n"
                                f"Micro-ajustes (motor): {imm}\n"
                                f"Ajuste semanal (motor): {weekly}\n"
                            )

                            n8n_payload = {
                                'chatInput': prompt,
                                'message': prompt,
                                'sessionId': session_id,
                                'username': (getattr(user, 'username', '') or ''),
                                'auth_header': auth_header,
                                'attachment': '',
                                'attachment_text': '',
                                'qaf': {'type': 'exp-013_body_architecture_explainer', 'result': res},
                                'system_rules': {
                                    'module': 'exp-013_body_architecture_explainer',
                                    'no_new_buttons': True,
                                    'no_medical': True,
                                },
                            }
                            resp = _bench_n8n_post(request, n8n_url, n8n_payload, timeout=45)
                            if resp.status_code == 200:
                                try:
                                    data = resp.json()
                                except Exception:
                                    data = {'output': resp.text}
                                if isinstance(data, dict) and isinstance(data.get('output'), str):
                                    qc_text = (data.get('output') or '').strip()
                                elif isinstance(data, str):
                                    qc_text = data.strip()

                        # Si viene HTML/iframe, extraer texto de srcdoc y limpiar tags.
                        try:
                            raw_out = str(qc_text or '').strip()
                            if raw_out.lower().startswith('<iframe'):
                                try:
                                    import html as _html
                                    m = re.search(r"\bsrcdoc=(\"|')(.*?)(\1)", raw_out, flags=re.IGNORECASE | re.DOTALL)
                                    if m:
                                        srcdoc = _html.unescape(m.group(2))
                                        srcdoc = re.sub(r"<\s*br\s*/?>", "\n", srcdoc, flags=re.IGNORECASE)
                                        srcdoc = re.sub(r"<[^>]+>", "", srcdoc)
                                        qc_text = (srcdoc or '').strip()
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Guardrail: si n8n intenta pedir foto/datos, se descarta.
                        try:
                            low_out = str(qc_text or '').lower()
                            if re.search(r"\b(env[ií]a|enviame|enviarme|adjunta|adjuntar)\b.*\bfoto\b", low_out):
                                qc_text = ""
                            if re.search(r"\bnecesito\b.*\b(foto|imagen)\b", low_out):
                                qc_text = ""
                            if '<iframe' in low_out:
                                qc_text = ""
                        except Exception:
                            pass

                        # Evitar coletilla de pregunta final
                        try:
                            lines2 = [ln.rstrip() for ln in str(qc_text or '').splitlines()]
                            while lines2 and not lines2[-1].strip():
                                lines2.pop()
                            if lines2 and ('¿' in lines2[-1] or lines2[-1].strip().endswith('?')):
                                lines2.pop()
                            qc_text = "\n".join(lines2).strip()
                        except Exception:
                            pass

                        # Si intentamos Quantum Coach y no entregó texto usable, se considera fallback.
                        try:
                            if (not str(qc_text or '').strip()) and isinstance(res, dict) and str(res.get('decision') or '').strip().lower() == 'accepted':
                                _bench_event_note(request, fallback_used=True)
                        except Exception:
                            pass

                        if qc_text.strip():
                            text = qc_text
                    except Exception:
                        pass

                    # Persistir por semana
                    try:
                        ws2 = dict(weekly_state)
                        pp = ws2.get('posture_proportion') if isinstance(ws2.get('posture_proportion'), dict) else {}
                        pp2 = dict(pp)
                        pp2[week_id_now] = {'result': res, 'updated_at': timezone.now().isoformat(), 'week_id': week_id_now}
                        ws2['posture_proportion'] = pp2
                        user.coach_weekly_state = ws2
                        user.coach_weekly_updated_at = timezone.now()
                        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                    except Exception:
                        pass

                    # Guardar último resultado (memoria corta) para usarlo como contexto en sugerencias/entrenamientos posteriores.
                    try:
                        csx = dict(getattr(user, 'coach_state', {}) or {})
                        csx['body_architecture_last_result'] = {'result': res, 'updated_at': timezone.now().isoformat()}
                        user.coach_state = csx
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    qas = []
                    try:
                        if isinstance(res, dict) and res.get('decision') != 'accepted':
                            qas = [
                                {'label': 'Repetir frente', 'type': 'pp_capture', 'view': 'front_relaxed', 'source': 'camera'},
                                {'label': 'Repetir perfil', 'type': 'pp_capture', 'view': 'side_right_relaxed', 'source': 'camera'},
                                {'label': 'Cancelar', 'type': 'pp_cancel'},
                            ]
                        else:
                            # Al cierre: botón de finalización (sin mezclar más acciones aquí).
                            qas = [
                                {'label': 'Finalizar', 'type': 'pp_cancel'},
                            ]
                    except Exception:
                        qas = []

                    return Response(
                        _attach_wow_event_payload(
                            user,
                            {
                                'output': text or 'Arquitectura Corporal listo.',
                                'qaf_posture_proportion': res,
                                'quick_actions': qas,
                            },
                            event_key='qaf_posture_proportion',
                            label='Arquitectura corporal evaluada',
                        )
                    )
        except Exception as ex:
            print(f"QAF posture proportion warning: {ex}")

        # 0.2.c) Exp-012: Alta Costura Inteligente (antes Shape & Presence) (1–2 fotos; keypoints 2D)
        try:
            if user and isinstance(request.data, dict):
                sp_req = request.data.get('shape_presence_request')

                # Post-análisis: pedir lista de prendas sugeridas (via Quantum Coach / n8n)
                try:
                    if request.data.get('couture_garments_request') is True:
                        cs_local = getattr(user, 'coach_state', {}) or {}
                        last_blob = cs_local.get('couture_last_result') if isinstance(cs_local.get('couture_last_result'), dict) else None
                        last_res = (last_blob or {}).get('result') if isinstance((last_blob or {}).get('result'), dict) else None

                        # freshness: si es viejo, pedir repetir análisis
                        try:
                            updated_at = str((last_blob or {}).get('updated_at') or '').strip()
                            if updated_at:
                                dt = datetime.fromisoformat(updated_at)
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.get_current_timezone())
                                if timezone.now() - dt > timedelta(hours=6):
                                    last_res = None
                        except Exception:
                            pass

                        if not isinstance(last_res, dict):
                            return Response({'output': "**Alta Costura Inteligente**\nPrimero haz el análisis con 1–2 fotos y luego toca ‘Ver prendas sugeridas’.", 'quick_actions': []})

                        garments_text = ""
                        try:
                            couture = last_res.get('couture') if isinstance(last_res.get('couture'), dict) else {}
                            plan = last_res.get('couture_plan') if isinstance(last_res.get('couture_plan'), dict) else {}
                            vars_ = last_res.get('variables') if isinstance(last_res.get('variables'), dict) else {}

                            prompt = (
                                "Eres un asesor de moda premium (alta costura).\n"
                                "Con base en este análisis de proporciones ópticas, genera una LISTA de prendas sugeridas que le queden bien.\n"
                                "IMPORTANTE: no pidas fotos ni más datos; ya existe análisis.\n"
                                "Responde SOLO en texto plano (sin HTML, sin <iframe>).\n"
                                "Formato: 8–12 bullets máximo. Incluye: prenda + corte + largo/tiro + tela/estructura.\n"
                                "Evita: diagnósticos, promesas, preguntas al final.\n\n"
                                f"Variables: {vars_}\n"
                                f"Plan: {plan}\n"
                                f"Couture: {couture}\n"
                            )

                            n8n_payload = {
                                'chatInput': prompt,
                                'message': prompt,
                                'sessionId': session_id,
                                'username': (getattr(user, 'username', '') or ''),
                                'auth_header': auth_header,
                                'attachment': '',
                                'attachment_text': '',
                                'qaf': {'type': 'exp-012_couture_garments', 'result': last_res},
                                'system_rules': {
                                    'module': 'exp-012_couture_garments',
                                    'no_new_buttons': True,
                                    'no_medical': True,
                                },
                            }
                            resp = _bench_n8n_post(request, n8n_url, n8n_payload, timeout=45)
                            if resp.status_code == 200:
                                try:
                                    data = resp.json()
                                except Exception:
                                    data = {'output': resp.text}
                                if isinstance(data, dict) and isinstance(data.get('output'), str):
                                    garments_text = (data.get('output') or '').strip()
                                elif isinstance(data, str):
                                    garments_text = data.strip()
                        except Exception:
                            garments_text = ""

                        # Guardrails: no pedir foto / no iframe
                        try:
                            low = str(garments_text or '').lower()
                            if low.startswith('<iframe'):
                                try:
                                    import html as _html
                                    m = re.search(r"\bsrcdoc=(\"|')(.*?)(\1)", garments_text, flags=re.IGNORECASE | re.DOTALL)
                                    if m:
                                        srcdoc = _html.unescape(m.group(2))
                                        srcdoc = re.sub(r"<\s*br\s*/?>", "\n", srcdoc, flags=re.IGNORECASE)
                                        srcdoc = re.sub(r"<[^>]+>", "", srcdoc)
                                        garments_text = (srcdoc or '').strip() or garments_text
                                except Exception:
                                    pass
                            if re.search(r"\b(env[ií]a|adjunta|adjuntar)\b.*\bfoto\b", low):
                                garments_text = ""
                        except Exception:
                            pass

                        if not (garments_text or '').strip():
                            _bench_event_note(request, fallback_used=True)
                            # Fallback determinista
                            vars_ = last_res.get('variables') if isinstance(last_res.get('variables'), dict) else {}
                            couture = last_res.get('couture') if isinstance(last_res.get('couture'), dict) else {}
                            proxies = couture.get('proxies') if isinstance(couture.get('proxies'), dict) else {}
                            tl = proxies.get('torso_leg_ratio')
                            vt = int(vars_.get('silhouette_v_taper') or 0)
                            prof = int(vars_.get('profile_stack') or 0)

                            out = [
                                "**Alta Costura Inteligente**",
                                "De acuerdo a tu análisis, estas son prendas sugeridas para elevar tu presencia:",
                                "",
                            ]
                            if tl is not None:
                                try:
                                    tlf = float(tl)
                                    if tlf >= 0.66:
                                        out.append("- Pantalón tiro medio‑alto, pierna recta, tela con caída (lana fría / sarga fina).")
                                        out.append("- Chaqueta corta‑media con cintura visual alta (1 botón o cruzada corta).")
                                    elif tlf <= 0.52:
                                        out.append("- Pantalón tiro medio y línea limpia, evita ultra‑alto; caída recta.")
                                        out.append("- Chaqueta largo medio para equilibrar verticalidad (sin cortar torso).")
                                except Exception:
                                    pass
                            if vt >= 75:
                                out.append("- Blazer estructurado suave en hombro + solapa limpia (quiet luxury).")
                            else:
                                out.append("- Chaqueta con estructura ligera en hombro + entalle sutil para arquitectura.")
                            if prof and prof < 70:
                                out.append("- Camisa/cuello: escote en V o cuello abierto + solapa en punta para estilizar la línea superior.")
                            out.append("- Paleta recomendada para foto: monocromo o 2 tonos con contraste controlado.")
                            out.append("- Fit: evita exceso de tela en cintura/cadera; costuras alineadas y caída continua.")
                            garments_text = "\n".join(out).strip()

                        return Response({'output': garments_text, 'quick_actions': [{'label': 'Finalizar', 'type': 'shape_cancel'}]})
                except Exception:
                    pass

                # Iniciar flujo por intención explícita
                msg_low = str(message or '').strip().lower()

                # Activación intuitiva (texto libre): tolera acentos y frases más largas.
                msg_norm = msg_low
                try:
                    import unicodedata

                    msg_norm = "".join(
                        ch for ch in unicodedata.normalize('NFKD', msg_low) if not unicodedata.combining(ch)
                    )
                except Exception:
                    msg_norm = msg_low
                msg_norm = re.sub(r"\s+", " ", msg_norm or "").strip()

                # 1) Triggers directos (si el usuario lo pide por nombre)
                explicit = bool(
                    re.search(
                        r"\b("
                        r"alta costura inteligente|alta costura|couture|"
                        r"shape\s*&\s*presence|shape presence|shape and presence|"
                        r"forma\s*&\s*presencia|forma y presencia|presencia\s*&\s*forma"
                        r")\b",
                        msg_norm,
                    )
                )

                # 2) Frases naturales: intención de asesoría visual + acción de analizar/foto/outfit
                style_kw = bool(
                    re.search(
                        r"\b(asesoria visual|asesoria de imagen|disenador|dise[ñn]ador|"
                        r"silueta|proporcion(?:es)?|proporciones|verticalidad|presencia|v\s*-?taper|quiet luxury|"
                        r"sastreria|solapa|escote|cuello)\b",
                        msg_norm,
                    )
                )
                action_kw = bool(
                    re.search(
                        r"\b(analiza|analizar|analisis|evaluar|mide|medir|foto|imagen|outfit|ropa|look|"
                        r"cuerpo completo|perfil|frente)\b",
                        msg_norm,
                    )
                )

                want_sp = explicit or (style_kw and action_kw)
                if (not isinstance(sp_req, dict)) and want_sp:
                    out = (
                        "**Alta Costura Inteligente**\n"
                        "Asesoría visual tipo diseñador (estética de atelier): arquitectura de silueta + verticalidad + presencia (ratios *ópticos*, sin cm reales).\n\n"
                        "- Mínimo: 1 foto (frente relajado)\n"
                        "- Mejor: agrega 1 foto (perfil derecho)\n\n"
                        "Empecemos con **frente relajado** (cuerpo completo, buena luz, cámara a 2–3m). Voy a leer proporciones y caída en cámara."
                    )
                    return Response(
                        {
                            'output': out,
                            'quick_actions': [
                                {'label': 'Tomar foto frente', 'type': 'shape_capture', 'view': 'front_relaxed', 'source': 'camera'},
                                {'label': 'Adjuntar foto frente', 'type': 'shape_capture', 'view': 'front_relaxed', 'source': 'attach'},
                                {'label': 'Cancelar', 'type': 'shape_cancel'},
                            ],
                        }
                    )

                if isinstance(sp_req, dict):
                    from .qaf_shape_presence.engine import evaluate_shape_presence, render_professional_summary

                    week_id_now = _week_id()
                    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

                    baseline = None
                    try:
                        sp = weekly_state.get('shape_presence') if isinstance(weekly_state.get('shape_presence'), dict) else {}
                        prev_keys = [k for k in sp.keys() if isinstance(k, str) and k != week_id_now]
                        if prev_keys:
                            prev_key = sorted(prev_keys)[-1]
                            prev_row = sp.get(prev_key)
                            if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                                baseline = prev_row.get('result')
                    except Exception:
                        baseline = None

                    poses = sp_req.get('poses') if isinstance(sp_req.get('poses'), dict) else {}
                    res = evaluate_shape_presence({'poses': poses, 'baseline': baseline}).payload
                    text = render_professional_summary(res)

                    # Guardar último resultado para botón de prendas sugeridas
                    try:
                        cs2 = dict(getattr(user, 'coach_state', {}) or {})
                        cs2['couture_last_result'] = {'result': res, 'updated_at': timezone.now().isoformat()}
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    # Persistir por semana
                    try:
                        ws2 = dict(weekly_state)
                        sp = ws2.get('shape_presence') if isinstance(ws2.get('shape_presence'), dict) else {}
                        sp2 = dict(sp)
                        sp2[week_id_now] = {'result': res, 'updated_at': timezone.now().isoformat(), 'week_id': week_id_now}
                        ws2['shape_presence'] = sp2
                        user.coach_weekly_state = ws2
                        user.coach_weekly_updated_at = timezone.now()
                        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                    except Exception:
                        pass

                    qas = []
                    try:
                        if isinstance(res, dict) and res.get('decision') != 'accepted':
                            qas = [
                                {'label': 'Repetir frente', 'type': 'shape_capture', 'view': 'front_relaxed', 'source': 'camera'},
                                {'label': 'Adjuntar frente', 'type': 'shape_capture', 'view': 'front_relaxed', 'source': 'attach'},
                                {'label': 'Cancelar', 'type': 'shape_cancel'},
                            ]
                        else:
                            qas = [
                                {'label': 'Ver prendas sugeridas', 'type': 'message', 'text': 'Ver prendas sugeridas', 'payload': {'couture_garments_request': True}},
                                {'label': 'Finalizar', 'type': 'shape_cancel'},
                            ]
                    except Exception:
                        qas = []

                    return Response(
                        _attach_wow_event_payload(
                            user,
                            {
                                'output': text or 'Alta Costura Inteligente listo.',
                                'qaf_shape_presence': res,
                                'quick_actions': qas,
                            },
                            event_key='qaf_shape_presence',
                            label='Shape & Presence actualizado',
                        )
                    )
        except Exception as ex:
            print(f"QAF shape presence warning: {ex}")

        # 0.2.e) Exp-011: Skin Health Intelligence (1 foto; energía visible + salud de piel)
        try:
            if user and isinstance(request.data, dict):
                msg_low = str(message or '').strip().lower()
                want_skin = False
                try:
                    # Activación intuitiva (sin ser demasiado amplia)
                    if re.search(r"\b(vitalidad\s+de\s+la\s+pi?e?l|vitalidad\s+pi?e?l|pi?e?l|skin\s*health|skincare|rutina\s+de\s+pi?e?l|cara|rostro|ojeras|manchas)\b", msg_low):
                        want_skin = True
                    # Tolerancia a typo común ("peil")
                    if ("vitalidad" in msg_low) and ("piel" in msg_low or "peil" in msg_low):
                        want_skin = True
                    # Casos comunes (tildes/variantes)
                    if any(k in msg_low for k in ("acne", "acné", "irritación", "irritacion", "reseca", "resequedad", "grasa", "brillo")):
                        want_skin = True
                    if re.fullmatch(r"belleza\s*/\s*piel|belleza\s+piel|piel|skin\s*health", msg_low or ""):
                        want_skin = True
                except Exception:
                    want_skin = bool(re.fullmatch(r"belleza\s*/\s*piel|belleza\s+p\s*iel|piel|skin\s*health", msg_low or ""))

                cs = getattr(user, 'coach_state', {}) or {}
                health_mode = str(cs.get('health_mode') or '').strip().lower()
                health_mode_until = str(cs.get('health_mode_until') or '').strip()

                # Cancelar: cierre limpio del modo (server-side)
                try:
                    if request.data.get('skin_cancel') is True:
                        cs2 = dict(cs)
                        cs2['health_mode'] = ''
                        cs2['health_mode_until'] = ''
                        cs2.pop('skin_pending_attachment', None)
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                        return Response({'output': '**Vitalidad de la Piel**\nListo. Cerré el proceso.', 'skin_flow_stage': 'completed'})
                except Exception:
                    pass

                # 1) Si el usuario selecciona el modo, lo guardamos y pedimos foto.
                if want_skin:
                    cs2 = dict(cs)
                    cs2['health_mode'] = 'skin'
                    cs2['health_mode_until'] = (timezone.now() + timedelta(minutes=15)).isoformat()
                    user.coach_state = cs2
                    user.coach_state_updated_at = timezone.now()
                    user.save(update_fields=['coach_state', 'coach_state_updated_at'])

                    # Reflejar el modo en variables locales (evita depender del cs viejo)
                    health_mode = 'skin'
                    health_mode_until = str(cs2.get('health_mode_until') or '').strip()

                    # Si el usuario ya había enviado una foto y quedó "pendiente" por el router de Vision,
                    # reutilizarla para analizar sin pedir que la adjunte de nuevo.
                    if not attachment_url:
                        try:
                            pending = cs.get('skin_pending_attachment') if isinstance(cs.get('skin_pending_attachment'), dict) else None
                            if isinstance(pending, dict):
                                pending_url = str(pending.get('attachment_url') or '').strip()
                                pending_until = str(pending.get('until') or '').strip()
                                ok = False
                                if pending_url and pending_until:
                                    until_dt = datetime.fromisoformat(pending_until)
                                    if until_dt.tzinfo is None:
                                        until_dt = until_dt.replace(tzinfo=timezone.get_current_timezone())
                                    ok = timezone.now() <= until_dt
                                if ok and pending_url:
                                    attachment_url = pending_url
                        except Exception:
                            pass

                    # Si aún no hay foto, pedirla con CTAs claros.
                    if not attachment_url:
                        out = (
                            "**Vitalidad de la Piel**\n"
                            "**✅ Empecemos** (sin diagnósticos médicos).\n\n"
                            "Para que el análisis sea confiable:\n"
                            "- Luz natural, sin contraluz\n"
                            "- Sin filtros\n"
                            "- Rostro centrado\n\n"
                            "Cuando estés listo, envía **1 foto**."
                        )
                        return Response(
                            {
                                'output': out,
                                'quick_actions': [
                                    {'label': 'Tomar foto', 'type': 'open_camera'},
                                    {'label': 'Adjuntar foto', 'type': 'open_attach'},
                                    {'label': 'Cancelar', 'type': 'skin_cancel'},
                                ],
                            }
                        )

                # 2) Si hay imagen y el modo salud=skin está activo, ejecutamos análisis.
                mode_active = False
                try:
                    if health_mode == 'skin' and health_mode_until:
                        until_dt = datetime.fromisoformat(health_mode_until)
                        if until_dt.tzinfo is None:
                            until_dt = until_dt.replace(tzinfo=timezone.get_current_timezone())
                        mode_active = timezone.now() <= until_dt
                except Exception:
                    mode_active = (health_mode == 'skin')

                # Si el modo está activo y el usuario responde "sí" pero aún no hay foto, re-mostrar CTAs.
                try:
                    if mode_active and not attachment_url:
                        is_yes = bool(re.fullmatch(r"(si|sí|ok|dale|listo|de\s*una|vamos|claro)", msg_low or ""))
                        if is_yes:
                            out = (
                                "**Vitalidad de la Piel**\n"
                                "Perfecto. Envíame **1 foto** del rostro (luz natural, sin filtros, sin contraluz)."
                            )
                            return Response(
                                {
                                    'output': out,
                                    'skin_flow_stage': 'need_photo',
                                    'quick_actions': [
                                        {'label': 'Tomar foto', 'type': 'open_camera'},
                                        {'label': 'Adjuntar foto', 'type': 'open_attach'},
                                        {'label': 'Cancelar', 'type': 'skin_cancel'},
                                    ],
                                }
                            )
                except Exception:
                    pass

                # Si el usuario pide hábitos (post-análisis), devolver hábitos y cerrar.
                try:
                    if mode_active and isinstance(request.data.get('skin_habits_request'), bool) and request.data.get('skin_habits_request') is True:
                        last = cs.get('skin_last_result') if isinstance(cs.get('skin_last_result'), dict) else None
                        last_res = (last or {}).get('result') if isinstance((last or {}).get('result'), dict) else {}
                        plan = last_res.get('recommendation_plan') if isinstance(last_res.get('recommendation_plan'), dict) else {}
                        prios = plan.get('priorities') if isinstance(plan.get('priorities'), list) else []
                        acts = plan.get('actions') if isinstance(plan.get('actions'), list) else []
                        prios = [str(x).strip() for x in prios if str(x).strip()]
                        acts = [str(x).strip() for x in acts if str(x).strip()]

                        out_lines = [
                            "**Vitalidad de la Piel**",
                            "**✅ Hábitos sugeridos para la piel**",
                        ]
                        if prios:
                            out_lines.append("\n**🎯 En qué enfocarte**")
                            for i, p in enumerate(prios[:3], start=1):
                                out_lines.append(f"- Prioridad {i}: {p}")
                        if acts:
                            out_lines.append("\n**✅ Acciones simples**")
                            for a in acts[:5]:
                                out_lines.append(f"- {a}")
                        out_lines.append("\nSi quieres medir progreso real, repite con la misma luz/encuadre 1 vez por semana.")

                        # cerrar modo
                        try:
                            cs2 = dict(cs)
                            cs2['health_mode'] = ''
                            cs2['health_mode_until'] = ''
                            cs2.pop('skin_pending_attachment', None)
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                        except Exception:
                            pass

                        return Response(
                            {
                                'output': "\n".join(out_lines).strip(),
                                'skin_flow_stage': 'completed',
                            }
                        )
                except Exception:
                    pass

                # 2.a) Si el modo está activo, permitir registrar contexto auto-reportado (sin foto todavía).
                if mode_active and isinstance(request.data, dict):
                    prompt = str(request.data.get('skin_context_prompt') or '').strip().lower()
                    if prompt in ('water', 'stress', 'sun', 'movement'):
                        if prompt == 'water':
                            out = (
                                "**Vitalidad de la Piel**\n"
                                "**💧 Agua (auto‑reporte)**\n"
                                "Elige lo que aplique hoy (esto mejora la lectura contextual)."
                            )
                            return Response(
                                {
                                    'output': out,
                                    'quick_actions': [
                                        {'label': '+250ml', 'type': 'message', 'text': '+250ml', 'payload': {'skin_context_update': {'water_delta_liters': 0.25}}},
                                        {'label': '+500ml', 'type': 'message', 'text': '+500ml', 'payload': {'skin_context_update': {'water_delta_liters': 0.5}}},
                                        {'label': '+750ml', 'type': 'message', 'text': '+750ml', 'payload': {'skin_context_update': {'water_delta_liters': 0.75}}},
                                        {'label': 'Listo (enviar foto)', 'type': 'message', 'text': 'Listo (enviar foto)', 'payload': {'skin_context_prompt': 'photo'}},
                                        {'label': 'Cancelar', 'type': 'skin_cancel'},
                                    ],
                                }
                            )

                        if prompt == 'stress':
                            out = (
                                "**Vitalidad de la Piel**\n"
                                "**🧠 Estrés (1–5)**\n"
                                "1 = bajo · 5 = alto"
                            )
                            qas = []
                            for i in (1, 2, 3, 4, 5):
                                qas.append({'label': str(i), 'type': 'message', 'text': str(i), 'payload': {'skin_context_update': {'stress_1_5': i}}})
                            qas.append({'label': 'Listo (enviar foto)', 'type': 'message', 'text': 'Listo (enviar foto)', 'payload': {'skin_context_prompt': 'photo'}})
                            qas.append({'label': 'Cancelar', 'type': 'skin_cancel'})
                            return Response({'output': out, 'quick_actions': qas})

                        if prompt == 'movement':
                            out = (
                                "**Vitalidad de la Piel**\n"
                                "**🚶 Movimiento (1–5)**\n"
                                "1 = muy bajo · 5 = excelente"
                            )
                            qas = []
                            for i in (1, 2, 3, 4, 5):
                                qas.append({'label': str(i), 'type': 'message', 'text': str(i), 'payload': {'skin_context_update': {'movement_1_5': i}}})
                            qas.append({'label': 'Listo (enviar foto)', 'type': 'message', 'text': 'Listo (enviar foto)', 'payload': {'skin_context_prompt': 'photo'}})
                            qas.append({'label': 'Cancelar', 'type': 'skin_cancel'})
                            return Response({'output': out, 'quick_actions': qas})

                        if prompt == 'sun':
                            out = (
                                "**Vitalidad de la Piel**\n"
                                "**☀️ Exposición al sol (minutos)**\n"
                                "Estimado hoy (solo para contexto)."
                            )
                            return Response(
                                {
                                    'output': out,
                                    'quick_actions': [
                                        {'label': '0', 'type': 'message', 'text': '0', 'payload': {'skin_context_update': {'sun_minutes': 0}}},
                                        {'label': '10', 'type': 'message', 'text': '10', 'payload': {'skin_context_update': {'sun_minutes': 10}}},
                                        {'label': '20', 'type': 'message', 'text': '20', 'payload': {'skin_context_update': {'sun_minutes': 20}}},
                                        {'label': '40+', 'type': 'message', 'text': '40+', 'payload': {'skin_context_update': {'sun_minutes': 40}}},
                                        {'label': 'Listo (enviar foto)', 'type': 'message', 'text': 'Listo (enviar foto)', 'payload': {'skin_context_prompt': 'photo'}},
                                        {'label': 'Cancelar', 'type': 'skin_cancel'},
                                    ],
                                }
                            )

                    # Conveniencia: volver a CTA de foto
                    if prompt == 'photo':
                        return Response(
                            {
                                'output': "**Vitalidad de la Piel**\nPerfecto. Ahora envía **1 foto** (luz natural, sin filtros, rostro centrado).",
                                'quick_actions': [
                                    {'label': 'Tomar foto', 'type': 'open_camera'},
                                    {'label': 'Adjuntar foto', 'type': 'open_attach'},
                                    {'label': 'Cancelar', 'type': 'skin_cancel'},
                                ],
                            }
                        )

                    upd = request.data.get('skin_context_update')
                    if isinstance(upd, dict) and upd:
                        try:
                            cs2 = dict(cs)
                            sc = cs2.get('skin_context') if isinstance(cs2.get('skin_context'), dict) else {}
                            sc2 = dict(sc)

                            # Agua puede venir como delta o valor absoluto
                            if upd.get('water_delta_liters') is not None:
                                try:
                                    d = float(upd.get('water_delta_liters') or 0.0)
                                    cur = float(sc2.get('water_liters') or 0.0)
                                    sc2['water_liters'] = round(max(0.0, cur + d), 2)
                                except Exception:
                                    pass
                            if upd.get('water_liters') is not None:
                                try:
                                    sc2['water_liters'] = round(max(0.0, float(upd.get('water_liters'))), 2)
                                except Exception:
                                    pass

                            for k in ('stress_1_5', 'movement_1_5', 'sun_minutes', 'steps'):
                                if upd.get(k) is None:
                                    continue
                                try:
                                    sc2[k] = float(upd.get(k))
                                except Exception:
                                    sc2[k] = upd.get(k)

                            cs2['skin_context'] = sc2
                            cs2['skin_context_updated_at'] = timezone.now().isoformat()
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])

                            return Response(
                                {
                                    'output': "**Vitalidad de la Piel**\nListo. Ya lo integro a tu lectura de hoy. Ahora envía **1 foto**.",
                                    'quick_actions': [
                                        {'label': 'Tomar foto', 'type': 'open_camera'},
                                        {'label': 'Adjuntar foto', 'type': 'open_attach'},
                                        {'label': 'Cancelar', 'type': 'skin_cancel'},
                                    ],
                                }
                            )
                        except Exception:
                            pass

                if mode_active and attachment_url:
                    # Descargar bytes solo si es attachment del usuario.
                    image_bytes = None
                    image_content_type = None
                    try:
                        normalized_url = _normalize_attachment_url(str(attachment_url))
                        container_name, blob_name = _extract_blob_ref_from_url(normalized_url)
                        if container_name == _chat_attachment_container() and blob_name:
                            blob_name = _resolve_blob_name(container_name, blob_name) or blob_name
                            safe_username = user.username.replace('/', '_')
                            allowed = bool(blob_name.startswith(f"{safe_username}/")) or _is_signed_chat_attachment_url(normalized_url)
                            if allowed:
                                max_bytes = int(os.getenv('CHAT_VISION_MAX_BYTES', str(4 * 1024 * 1024)) or (4 * 1024 * 1024))
                                resp = requests.get(normalized_url, timeout=20)
                                if resp.status_code == 200 and resp.content and len(resp.content) <= max_bytes:
                                    image_bytes = resp.content
                                    image_content_type = (resp.headers.get('Content-Type') or 'image/jpeg')
                    except Exception:
                        image_bytes = None

                    if not image_bytes:
                        return Response({'output': 'No pude descargar la imagen. Intenta adjuntarla de nuevo.', 'quick_actions': [{'label': 'Tomar foto', 'type': 'open_camera'}, {'label': 'Adjuntar foto', 'type': 'open_attach'}]})

                    # Contexto opcional desde lifestyle_last (si existe)
                    ctx = {}
                    try:
                        lifestyle_last = (cs.get('lifestyle_last') or {}).get('result') if isinstance(cs.get('lifestyle_last'), dict) else None
                        if isinstance(lifestyle_last, dict):
                            sig = lifestyle_last.get('signals') if isinstance(lifestyle_last.get('signals'), dict) else {}
                            ctx['sleep_minutes'] = (sig.get('sleep') or {}).get('value')
                            # Movimiento: pasos si existen
                            ctx['steps'] = (sig.get('steps') or {}).get('value')
                            # Estrés: aproximación 1–5 basada en stress_inv.score01 (más alto = mejor)
                            try:
                                stress_score01 = (sig.get('stress_inv') or {}).get('score01')
                                if stress_score01 is not None:
                                    s = float(stress_score01)
                                    s = max(0.0, min(1.0, s))
                                    # 1 (bajo estrés) .. 5 (alto estrés)
                                    ctx['stress_1_5'] = round((1.0 - s) * 4.0 + 1.0, 1)
                            except Exception:
                                pass
                    except Exception:
                        ctx = {}

                    # Contexto auto-reportado específico para Skin (si existe)
                    try:
                        sc = cs.get('skin_context') if isinstance(cs.get('skin_context'), dict) else {}
                        if isinstance(sc, dict):
                            # Preferir auto-reporte explícito sobre inferencias
                            for k in ('water_liters', 'stress_1_5', 'movement_1_5', 'sun_minutes', 'steps'):
                                if sc.get(k) is None:
                                    continue
                                ctx[k] = sc.get(k)
                    except Exception:
                        pass

                    week_id_now = _week_id()
                    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
                    baseline = None
                    try:
                        sh = weekly_state.get('skin_health') if isinstance(weekly_state.get('skin_health'), dict) else {}
                        prev_keys = [k for k in sh.keys() if isinstance(k, str) and k != week_id_now]
                        if prev_keys:
                            prev_key = sorted(prev_keys)[-1]
                            prev_row = sh.get(prev_key)
                            if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                                baseline = prev_row.get('result')
                    except Exception:
                        baseline = None

                    from .qaf_skin_health.engine import evaluate_skin_health, render_professional_summary
                    res = evaluate_skin_health(image_bytes=image_bytes, content_type=image_content_type, context=ctx, baseline=baseline).payload

                    # Enriquecer para renderer
                    try:
                        if isinstance(res, dict):
                            res = dict(res)
                            res['user_display_name'] = (getattr(user, 'full_name', None) or getattr(user, 'username', '') or '').strip()
                    except Exception:
                        pass

                    text = render_professional_summary(res)

                    # Persistir
                    try:
                        ws2 = dict(weekly_state)
                        sh = ws2.get('skin_health') if isinstance(ws2.get('skin_health'), dict) else {}
                        sh2 = dict(sh)
                        sh2[week_id_now] = {'result': res, 'updated_at': timezone.now().isoformat(), 'week_id': week_id_now}
                        ws2['skin_health'] = sh2
                        user.coach_weekly_state = ws2
                        user.coach_weekly_updated_at = timezone.now()
                        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                    except Exception:
                        pass

                    # Guardar último resultado en coach_state para acciones post-análisis (hábitos)
                    try:
                        cs2 = dict(cs)
                        cs2['skin_last_result'] = {'result': res, 'updated_at': timezone.now().isoformat()}
                        user.coach_state = cs2
                        user.coach_state_updated_at = timezone.now()
                        user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    # Mantener el modo activo para permitir CTA post-análisis (hábitos) y cerrar al finalizar.

                    qas = []
                    try:
                        if isinstance(res, dict) and str(res.get('decision') or '').strip().lower() != 'accepted':
                            qas = [
                                {'label': 'Tomar foto', 'type': 'open_camera'},
                                {'label': 'Adjuntar foto', 'type': 'open_attach'},
                                {'label': 'Cancelar', 'type': 'skin_cancel'},
                            ]
                    except Exception:
                        qas = []

                    # Después del análisis: ofrecer CTA de hábitos y cancelar.
                    qas2 = []
                    try:
                        qas2 = [
                            {'label': 'Ver hábitos sugeridos para la piel', 'type': 'message', 'text': 'Ver hábitos sugeridos para la piel', 'payload': {'skin_habits_request': True}},
                            {'label': 'Finalizar', 'type': 'skin_cancel'},
                        ]
                        # Si no fue accepted, prevalecen los CTAs de reintento.
                        if qas:
                            qas2 = qas
                    except Exception:
                        qas2 = qas

                    coach_output = ''
                    try:
                        skin_context_text = (text or '').strip()
                        if skin_context_text:
                            skin_context_text = f"[VITALIDAD DE LA PIEL]\n{skin_context_text}"

                        n8n_payload = {
                            'chatInput': str(message or 'Vitalidad de la Piel').strip() or 'Vitalidad de la Piel',
                            'message': str(message or 'Vitalidad de la Piel').strip() or 'Vitalidad de la Piel',
                            'sessionId': session_id,
                            'username': username_for_payload,
                            'auth_header': auth_header,
                            'attachment': '',
                            'attachment_text': skin_context_text,
                            'attachment_text_diagnostic': '',
                            'qaf': {'type': 'exp-011_skin_health', 'result': res},
                            'qaf_skin_health': res,
                            'system_rules': {
                                'module': 'exp-011_skin_health',
                                'llm_role': 'coach_full_response',
                                'respond_from_qaf_result': True,
                            },
                        }
                        n8n_resp = _bench_n8n_post(request, n8n_url, n8n_payload, timeout=45)
                        if n8n_resp.status_code == 200:
                            try:
                                n8n_data = n8n_resp.json()
                            except Exception:
                                n8n_data = {'output': n8n_resp.text}
                            if isinstance(n8n_data, dict):
                                coach_output = str(n8n_data.get('output') or '').strip()
                            elif isinstance(n8n_data, str):
                                coach_output = n8n_data.strip()
                    except Exception:
                        coach_output = ''

                    final_output = coach_output or (
                        "Listo. Ya procesé tu análisis de Vitalidad de la Piel, "
                        "pero no pude generar la respuesta completa del Quantum Coach en este intento. "
                        "Inténtalo de nuevo en unos segundos."
                    )
                    return Response(
                        _attach_wow_event_payload(
                            user,
                            {
                                'output': final_output,
                                'qaf_skin_health': res,
                                'quick_actions': qas2,
                                'skin_flow_stage': 'analysis_done',
                            },
                            event_key='qaf_skin_health',
                            label='Vitalidad de la piel analizada',
                        )
                    )
        except Exception as ex:
            print(f"QAF skin health warning: {ex}")

        # 0.3) Exp-004: generar menú por payload o por intención en el mensaje
        try:
            if user:
                # Acciones directas desde botones (quick-actions)
                meal_plan_apply_only = False
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
                        meal_plan_apply_only = True
                    except Exception:
                        pass

                # UX: confirmar aplicación sin depender de n8n cuando el tap viene directo.
                try:
                    if meal_plan_apply_only and (not isinstance(request.data.get('meal_plan_request'), dict)):
                        return Response(
                            {
                                'output': (
                                    "✅ **Listo, menú aplicado para esta semana.**\n"
                                    "¿Qué ganas con esto? Menos fricción diaria: ya tienes una estructura clara para decidir más rápido y sostener constancia."
                                ),
                                'quick_actions': [
                                    {'label': 'Lista de compras', 'type': 'message', 'text': 'Lista de compras', 'payload': {'meal_plan_view': 'shopping_list'}},
                                    {'label': 'Cambiar cena (mañana)', 'type': 'message', 'text': 'Cambiar cena (mañana)', 'payload': {'meal_plan_mutate': {'day_index': 1, 'slot': 'cena', 'direction': 'high'}}},
                                    {'label': 'Finalizar', 'type': 'services_menu', 'page': 'close'},
                                ],
                            },
                            status=200,
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
                                from .qaf_meal_planner.engine import mutate_plan_slot

                                before_names = []
                                before_slot_kcal = None
                                before_day_total = None
                                try:
                                    plan0 = current_result.get('plan') if isinstance(current_result.get('plan'), dict) else {}
                                    days0 = plan0.get('days') if isinstance(plan0.get('days'), list) else []
                                    if 0 <= int(day_index) < len(days0):
                                        d0 = days0[int(day_index)] if isinstance(days0[int(day_index)], dict) else {}
                                        before_day_total = float(d0.get('total_kcal')) if d0.get('total_kcal') is not None else None
                                        meals0 = d0.get('meals') if isinstance(d0.get('meals'), list) else []
                                        for mm in meals0:
                                            if not isinstance(mm, dict):
                                                continue
                                            if str(mm.get('slot') or '').strip().lower() != slot:
                                                continue
                                            items0 = mm.get('items') if isinstance(mm.get('items'), list) else []
                                            before_names = [str(it.get('name') or '').strip() for it in items0 if isinstance(it, dict) and str(it.get('name') or '').strip()]
                                            before_slot_kcal = float(mm.get('kcal')) if mm.get('kcal') is not None else None
                                            break
                                except Exception:
                                    before_names = []
                                    before_slot_kcal = None
                                    before_day_total = None

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

                                after_names = []
                                after_slot_kcal = None
                                after_day_total = None
                                kcal_target = None
                                micro_cov_pct = None
                                variety_health_pct = None
                                try:
                                    inp = mutated.get('inputs') if isinstance(mutated.get('inputs'), dict) else {}
                                    kcal_target = float(inp.get('kcal_day')) if inp.get('kcal_day') is not None else None

                                    scores = mutated.get('scores') if isinstance(mutated.get('scores'), dict) else {}
                                    if scores.get('micro_coverage') is not None:
                                        micro_cov_pct = int(round(float(scores.get('micro_coverage')) * 100.0))
                                    if scores.get('variety_penalty') is not None:
                                        variety_health_pct = int(round((1.0 - float(scores.get('variety_penalty'))) * 100.0))

                                    plan1 = mutated.get('plan') if isinstance(mutated.get('plan'), dict) else {}
                                    days1 = plan1.get('days') if isinstance(plan1.get('days'), list) else []
                                    if 0 <= int(day_index) < len(days1):
                                        d1 = days1[int(day_index)] if isinstance(days1[int(day_index)], dict) else {}
                                        after_day_total = float(d1.get('total_kcal')) if d1.get('total_kcal') is not None else None
                                        meals1 = d1.get('meals') if isinstance(d1.get('meals'), list) else []
                                        for mm in meals1:
                                            if not isinstance(mm, dict):
                                                continue
                                            if str(mm.get('slot') or '').strip().lower() != slot:
                                                continue
                                            items1 = mm.get('items') if isinstance(mm.get('items'), list) else []
                                            after_names = [str(it.get('name') or '').strip() for it in items1 if isinstance(it, dict) and str(it.get('name') or '').strip()]
                                            after_slot_kcal = float(mm.get('kcal')) if mm.get('kcal') is not None else None
                                            break
                                except Exception:
                                    pass

                                day_label = int(day_index) + 1
                                slot_label = slot.capitalize()
                                lines = []
                                lines.append(f"✅ **{slot_label} actualizada (Día {day_label}).**")
                                if before_names or after_names:
                                    lines.append("")
                                    lines.append(f"Antes: {', '.join(before_names) if before_names else 'N/D'}")
                                    lines.append(f"Ahora: {', '.join(after_names) if after_names else 'N/D'}")
                                if (before_slot_kcal is not None) and (after_slot_kcal is not None):
                                    diff_slot = float(after_slot_kcal) - float(before_slot_kcal)
                                    lines.append(f"Impacto en {slot}: {after_slot_kcal:.0f} kcal ({diff_slot:+.0f} kcal vs antes).")
                                if (after_day_total is not None) and (kcal_target is not None) and float(kcal_target) > 0:
                                    dev_pct = ((float(after_day_total) - float(kcal_target)) / float(kcal_target)) * 100.0
                                    lines.append(f"Día {day_label}: {after_day_total:.0f}/{kcal_target:.0f} kcal ({dev_pct:+.1f}% vs objetivo).")
                                if micro_cov_pct is not None:
                                    lines.append(f"Cobertura micro estimada: {micro_cov_pct}%.")
                                if variety_health_pct is not None:
                                    lines.append(f"Variedad del plan: {variety_health_pct}%." )
                                lines.append("")
                                lines.append("Este ajuste sí cuenta en el algoritmo: actualiza tu plan guardado, la lista de compras y los puntajes de calidad del menú.")

                                mutate_text = "\n".join(lines).strip()
                                meal_plan_text_for_output_override = mutate_text
                                attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[MENÚ ACTUALIZADO]\n{mutate_text}".strip()
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

                                try:
                                    ws_now = getattr(user, 'coach_weekly_state', {}) or {}
                                    is_applied = str(ws_now.get('meal_plan_active_week_id') or '') == str(week_id_now)
                                except Exception:
                                    is_applied = False

                                qa = [
                                    {'label': 'Lista de compras', 'type': 'message', 'text': 'Lista de compras', 'payload': {'meal_plan_view': 'shopping_list'}},
                                    {'label': 'Ver menú completo', 'type': 'message', 'text': 'Menú semanal', 'payload': {'meal_plan_request': {'variety': 'normal', 'meals_per_day': 3}}},
                                    {'label': 'Finalizar', 'type': 'services_menu', 'page': 'close'},
                                ]
                                if not bool(is_applied):
                                    qa[0] = {'label': 'Aplicar menú', 'type': 'message', 'text': 'Aplicar menú', 'payload': {'meal_plan_apply': True}}

                                return Response({'output': mutate_text, 'quick_actions': qa[:3], 'qaf_meal_plan': mutated}, status=200)
                    except Exception:
                        pass

                meal_req = request.data.get('meal_plan_request') if isinstance(request.data, dict) else None
                want_menu = False
                if isinstance(meal_req, dict):
                    want_menu = True
                else:
                    msg_low = str(message or '').lower()
                    if re.search(r"\b(men[uú]|meal\s*plan|plan\s+de\s+comidas|menu\s+semanal|men[uú]\s+semanal|plan\s+semanal\s+personalizado|plan\s+nutricional\s+semanal|plan\s+alimenticio\s+semanal)\b", msg_low):
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

                    # Narrativa premium/humana vía Quantum Coach (fallback determinista si falla).
                    try:
                        qc_text = ""
                        prompt = (
                            "Eres Quantum Coach. Convierte este menú semanal técnico en una explicación clara, útil y accionable para el usuario.\n"
                            "Responde SOLO en texto plano (sin HTML, sin <iframe>).\n"
                            "No diagnóstico médico. No promesas absolutas.\n"
                            "Formato:\n"
                            "1) Resumen de la semana (2-3 bullets)\n"
                            "2) Cómo ejecutar sin fricción (3-5 bullets)\n"
                            "3) Qué hacer si se rompe el plan (2 bullets)\n"
                            "4) Próximo paso inmediato (1 frase)\n"
                            "Evita mostrar JSON o métricas técnicas crudas.\n\n"
                            f"Mensaje del usuario: {str(message or '').strip()}\n"
                            f"Resultado QAF meal plan: {json.dumps(meal_plan_result, ensure_ascii=False)}\n"
                        )
                        n8n_payload = {
                            'chatInput': prompt,
                            'message': prompt,
                            'sessionId': session_id,
                            'username': (getattr(user, 'username', '') or ''),
                            'auth_header': auth_header,
                            'attachment': '',
                            'attachment_text': '',
                            'qaf': {'type': 'exp-004_meal_plan_explainer', 'result': meal_plan_result},
                            'system_rules': {
                                'module': 'exp-004_meal_plan_explainer',
                                'no_new_buttons': True,
                                'no_medical': True,
                            },
                        }
                        resp = _bench_n8n_post(request, n8n_url, n8n_payload, timeout=45)
                        if resp.status_code == 200:
                            try:
                                data = resp.json()
                            except Exception:
                                data = {'output': resp.text}
                            if isinstance(data, dict) and isinstance(data.get('output'), str):
                                qc_text = (data.get('output') or '').strip()
                            elif isinstance(data, str):
                                qc_text = data.strip()

                        low_qc = str(qc_text or '').lower()
                        if '<iframe' in low_qc:
                            qc_text = ''
                        if re.search(r"\b(env[ií]a|adjunta|adjuntar)\b.*\b(foto|imagen)\b", low_qc):
                            qc_text = ''

                        if qc_text.strip():
                            menu_text = qc_text
                        else:
                            _bench_event_note(request, fallback_used=True)
                    except Exception:
                        _bench_event_note(request, fallback_used=True)

                    if menu_text:
                        meal_intro = (
                            "**Menú Semanal Personalizado**\n"
                            "Ya construí tu propuesta semanal con foco en adherencia y ejecución simple.\n"
                        )
                        meal_plan_text_for_output_override = (meal_intro + "\n" + menu_text).strip()
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[MENÚ SEMANAL PROPUESTO]\n{meal_plan_text_for_output_override}".strip()

                    # Quick-actions para regeneración (wow: 1 tap)
                    try:
                        is_applied = False
                        try:
                            ws_now = getattr(user, 'coach_weekly_state', {}) or {}
                            is_applied = str(ws_now.get('meal_plan_active_week_id') or '') == str(week_id_now)
                        except Exception:
                            is_applied = False
                        quick_actions_out.extend(build_quick_actions_for_menu(variety_level=variety, is_applied=bool(is_applied)))
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
                    if not _is_premium_active(user):
                        gate_payload = _qaf_premium_gate_payload(
                            'exp-005_body_trend',
                            'Tendencia corporal (6 semanas)',
                            'Preview: lectura inicial de tu tendencia. Para escenarios completos y ajuste fino por calorías, activa Premium.',
                            request=request,
                            user=user,
                            source='chat_qaf',
                        )
                        body_trend_result = gate_payload
                        body_trend_text_for_output_override = gate_payload.get('text')
                        quick_actions_out = _qaf_append_premium_cta(quick_actions_out, module='exp-005_body_trend', source='chat_qaf')
                        raise StopIteration()

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

                    bt_text = render_professional_summary(body_trend_result, preferred_scenario=scenario)
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
                        has_intake = (kcal_in_avg is not None)
                        excluded_simulations: list[str] = []
                        if has_intake:
                            try:
                                cs0 = getattr(user, 'coach_state', {}) or {}
                                cs1 = dict(cs0)
                                by_sess = cs1.get('body_trend_sim_usage')
                                by_sess = by_sess if isinstance(by_sess, dict) else {}
                                by_sess2 = dict(by_sess)

                                sess_key = str(session_id or '').strip() or 'default'
                                row = by_sess2.get(sess_key)
                                row = row if isinstance(row, dict) else {}
                                row_week = str(row.get('week_id') or '').strip()
                                used_raw = row.get('used_scenarios') if isinstance(row.get('used_scenarios'), list) else []
                                used = {
                                    str(x or '').strip().lower()
                                    for x in used_raw
                                    if str(x or '').strip().lower() in ('follow_plan', 'minus_200', 'plus_200')
                                }
                                if row_week != str(week_id_now):
                                    used = set()

                                scen = str((bt_req or {}).get('scenario') or '').strip().lower() if isinstance(bt_req, dict) else ''

                                # Si arranca nueva evaluación, reseteamos simulaciones ya usadas.
                                if isinstance(bt_req, dict) and not scen:
                                    used = set()

                                # Si seleccionó una simulación, ocultamos solo esa para la siguiente respuesta.
                                if scen in ('follow_plan', 'minus_200', 'plus_200'):
                                    used.add(scen)

                                by_sess2[sess_key] = {
                                    'week_id': str(week_id_now),
                                    'used_scenarios': sorted(list(used)),
                                    'updated_at': timezone.now().isoformat(),
                                }
                                cs1['body_trend_sim_usage'] = by_sess2
                                user.coach_state = cs1
                                user.coach_state_updated_at = timezone.now()
                                user.save(update_fields=['coach_state', 'coach_state_updated_at'])

                                excluded_simulations = sorted(list(used))
                            except Exception:
                                excluded_simulations = []

                        quick_actions_out.extend(
                            build_quick_actions_for_trend(
                                has_intake=bool(has_intake),
                                allow_simulations=True,
                                excluded_scenarios=excluded_simulations,
                            )
                        )
                    except Exception:
                        pass
        except StopIteration:
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
                    if not _is_premium_active(user):
                        gate_payload = _qaf_premium_gate_payload(
                            'exp-006_posture',
                            'Postura correctiva',
                            'Preview: guía base de captura y postura. El análisis técnico completo y plan correctivo personalizado requieren Premium.',
                            request=request,
                            user=user,
                            source='chat_qaf',
                        )
                        posture_result = gate_payload
                        posture_text_for_output_override = gate_payload.get('text')
                        posture_quick_actions_out = _qaf_append_premium_cta(posture_quick_actions_out, module='exp-006_posture', source='chat_qaf')
                        raise StopIteration()

                    from .qaf_posture.engine import evaluate_posture, render_professional_summary

                    def _posture_extract_metrics(res: dict[str, Any]) -> dict[str, float]:
                        sigs = res.get('signals') if isinstance(res.get('signals'), list) else []
                        out: dict[str, float] = {}
                        for s in sigs:
                            if not isinstance(s, dict):
                                continue
                            name = str(s.get('name') or '').strip()
                            v = s.get('value')
                            if not name or v is None:
                                continue
                            try:
                                out[name] = float(v)
                            except Exception:
                                continue
                        return out

                    def _posture_delta(prev: dict[str, float], cur: dict[str, float]) -> list[dict[str, Any]]:
                        """Deltas vs medición previa. Positive/negative se interpreta según métrica."""
                        deltas: list[dict[str, Any]] = []

                        # Métricas donde MENOR es mejor
                        lower_better = {
                            'shoulder_asymmetry': 'Hombros (asimetría)',
                            'hip_asymmetry': 'Cadera (asimetría)',
                            'forward_head': 'Cabeza adelantada (proxy)',
                            'rounded_shoulders': 'Hombros redondeados (proxy)',
                            'pelvis_knee_offset': 'Pelvis/rodilla (proxy)',
                            'head_center_offset': 'Cabeza descentrada (proxy)',
                        }

                        # Métricas donde MAYOR es mejor (más cerca a extensión)
                        higher_better = {
                            'knee_angle_left': 'Rodilla izq (ángulo)',
                            'knee_angle_right': 'Rodilla der (ángulo)',
                        }

                        def _mk(key: str, label: str, d: float, kind: str):
                            deltas.append({
                                'key': key,
                                'label': label,
                                'delta': round(float(d), 4),
                                'kind': kind,
                            })

                        for key, label in lower_better.items():
                            if key in prev and key in cur:
                                _mk(key, label, float(cur[key]) - float(prev[key]), 'lower_better')

                        for key, label in higher_better.items():
                            if key in prev and key in cur:
                                _mk(key, label, float(cur[key]) - float(prev[key]), 'higher_better')

                        # Ordenar por impacto absoluto
                        deltas.sort(key=lambda x: abs(float(x.get('delta') or 0.0)), reverse=True)
                        return deltas

                    # Seguridad mínima: no inferimos desde 'visión descriptiva'. Solo usamos keypoints.
                    poses = None
                    user_ctx = {}
                    locale = 'es-CO'
                    if isinstance(pr, dict):
                        poses = pr.get('poses') if isinstance(pr.get('poses'), dict) else None
                        user_ctx = pr.get('user_context') if isinstance(pr.get('user_context'), dict) else {}
                        locale = (pr.get('locale') or '').strip() or 'es-CO'

                        # Calibración opcional por altura del perfil (cm).
                        try:
                            if isinstance(user_ctx, dict) and user_ctx.get('height_cm') is None:
                                hcm = _normalize_height_cm_from_user_value(getattr(user, 'height', None))
                                if hcm:
                                    user_ctx = {**user_ctx, 'height_cm': float(hcm)}
                        except Exception:
                            pass

                    def _posture_payload_ok(p: dict) -> bool:
                        if not isinstance(p, dict):
                            return False
                        has_any = False
                        for v in ('front', 'side'):
                            pose = p.get(v)
                            if pose is None:
                                continue
                            if not isinstance(pose, dict):
                                continue
                            kps = pose.get('keypoints')
                            if not isinstance(kps, list) or not kps:
                                continue
                            has_any = True
                            if len(kps) > 80:
                                return False
                            for kp in kps[:40]:
                                if not isinstance(kp, dict):
                                    return False
                                if not str(kp.get('name') or '').strip():
                                    return False
                                if kp.get('x') is None or kp.get('y') is None:
                                    return False
                        if not has_any:
                            return False
                        return True

                    if isinstance(poses, dict) and _posture_payload_ok(poses):
                        posture_result = evaluate_posture({'poses': poses, 'user_context': user_ctx, 'locale': locale}).payload

                        # Persistir últimas 4 mediciones (solo cuando hay output estructurado)
                        try:
                            if isinstance(posture_result, dict):
                                cur_metrics = _posture_extract_metrics(posture_result)
                                # Solo guardamos si hay señales suficientes (evitar basura)
                                if cur_metrics:
                                    cs0 = getattr(user, 'coach_state', {}) or {}
                                    cs1 = dict(cs0)
                                    hist = cs1.get('posture_measurements')
                                    hist_list = hist if isinstance(hist, list) else []
                                    hist_list = [x for x in hist_list if isinstance(x, dict)]

                                    prev_metrics = None
                                    if hist_list:
                                        prev_metrics = hist_list[-1].get('metrics') if isinstance(hist_list[-1].get('metrics'), dict) else None

                                    entry = {
                                        'ts': timezone.now().isoformat(),
                                        'week_id': _week_id(),
                                        'decision': str(posture_result.get('decision') or ''),
                                        'confidence': float(((posture_result.get('confidence') or {}).get('score')) or 0.0) if isinstance(posture_result.get('confidence'), dict) else 0.0,
                                        'metrics': {k: round(float(v), 6) for k, v in cur_metrics.items()},
                                        'labels': [str(x.get('key') or '') for x in (posture_result.get('labels') or []) if isinstance(x, dict) and str(x.get('key') or '').strip()],
                                    }

                                    hist_list.append(entry)
                                    # Mantener solo últimas 4
                                    hist_list = hist_list[-4:]
                                    cs1['posture_measurements'] = hist_list
                                    user.coach_state = cs1
                                    user.coach_state_updated_at = timezone.now()
                                    user.save(update_fields=['coach_state', 'coach_state_updated_at'])

                                    # Inyectar progreso vs última medición (si existía)
                                    if isinstance(prev_metrics, dict):
                                        deltas = _posture_delta(prev_metrics, cur_metrics)
                                        posture_result = {**posture_result, 'progress': {'vs_last': deltas[:6]}}
                        except Exception:
                            pass

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
                            "Ideal: 2 fotos (frontal + lateral) para mejor precisión.\n"
                            "También puedo darte un análisis parcial con 1 foto, pero no es 100% fiable sin la segunda vista.\n"
                            "Cuerpo completo, buena luz, cámara a la altura del pecho, 2–3m."
                        )
        except StopIteration:
            pass
        except Exception as ex:
            print(f"QAF posture warning: {ex}")

        # 0.6) Exp-007: Estado de hoy (Lifestyle Intelligence)
        try:
            if user:
                lr = request.data.get('lifestyle_request') if isinstance(request.data, dict) else None
                habit_done = request.data.get('lifestyle_habit_done') if isinstance(request.data, dict) else None

                lifestyle_habit_only = False

                want_lifestyle = False
                lifestyle_activation_mode = None  # payload | explicit | heuristic
                if isinstance(lr, dict) or isinstance(habit_done, dict):
                    want_lifestyle = True
                    lifestyle_activation_mode = 'payload'
                else:
                    msg_low = str(message or '').lower()
                    # Activación explícita
                    if re.search(r"\b(estado\s+de\s+hoy|como\s+voy\s+hoy|mi\s+energ[ií]a\s+hoy|dhss|lifestyle)\b", msg_low):
                        want_lifestyle = True
                        lifestyle_activation_mode = 'explicit'

                    # Activación intuitiva por auto-reporte + intención (entrenar hoy / sin quemarme)
                    if not want_lifestyle:
                        has_sleep = bool(re.search(r"\b(dorm[ií]|sueñ[oa]|sleep)\b", msg_low))
                        has_stress = bool(re.search(r"\b(estr[eé]s|estres|ansios|ansiedad|agotad|cansad|fatig)\b", msg_low))
                        has_move = bool(re.search(r"\b(camin[eé]|caminar|pasos|steps|movim)\b", msg_low))
                        wants_training = bool(re.search(r"\b(entren|training|rutina|workout)\b", msg_low))
                        todayish = bool(re.search(r"\b(hoy|ahora)\b", msg_low))
                        burnout = bool(re.search(r"\b(quemarme|burn\s*out|sin\s+quemarme|sin\s+agotarme|sin\s+reventarme)\b", msg_low))

                        # Heurística: si el usuario expresa 2+ señales (sueño/estrés/movimiento) y pregunta por entreno hoy.
                        signals = int(has_sleep) + int(has_stress) + int(has_move)
                        if (signals >= 2) and (wants_training or burnout) and todayish:
                            want_lifestyle = True
                            lifestyle_activation_mode = 'heuristic'

                # Rate limit: evitar interrumpir conversaciones con análisis automático muchas veces al día.
                try:
                    if want_lifestyle and lifestyle_activation_mode == 'heuristic':
                        cs = getattr(user, 'coach_state', {}) or {}
                        cs2 = dict(cs)
                        key = 'lifestyle_auto_activations'
                        counts = cs2.get(key) if isinstance(cs2.get(key), dict) else {}
                        today_key = timezone.localdate().isoformat()
                        cur = counts.get(today_key)
                        try:
                            cur_i = int(cur) if cur is not None else 0
                        except Exception:
                            cur_i = 0

                        MAX_AUTO_PER_DAY = 2
                        if cur_i >= MAX_AUTO_PER_DAY:
                            want_lifestyle = False
                            lifestyle_activation_mode = None
                        else:
                            counts2 = dict(counts)
                            counts2[today_key] = cur_i + 1
                            cs2[key] = counts2
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                except Exception:
                    pass

                # Registrar hábito como hecho (sin UI nueva)
                try:
                    if isinstance(habit_done, dict):
                        hid = str(habit_done.get('id') or '').strip()
                        if hid:
                            lifestyle_habit_only = True
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

                lifestyle_requested = bool(want_lifestyle)

                if want_lifestyle:
                    if not _is_premium_active(user):
                        gate_payload = _qaf_premium_gate_payload(
                            'exp-007_lifestyle',
                            'Estado de hoy (Lifestyle)',
                            'Preview: estado general del día. Para DHSS completo y micro-hábitos personalizados en continuidad, activa Premium.',
                            request=request,
                            user=user,
                            source='chat_qaf',
                        )
                        lifestyle_result = gate_payload
                        lifestyle_text_for_output_override = gate_payload.get('text')
                        lifestyle_quick_actions_out = _qaf_append_premium_cta(lifestyle_quick_actions_out, module='exp-007_lifestyle', source='chat_qaf')
                        raise StopIteration()

                    from datetime import timedelta
                    from .qaf_lifestyle.engine import evaluate_lifestyle, render_professional_summary

                    # UX: si el usuario solo está marcando un micro-hábito como hecho,
                    # no repetir el análisis completo en el chat.
                    if lifestyle_habit_only and (not isinstance(lr, dict)):
                        try:
                            hid_done = str((habit_done or {}).get('id') or '').strip()
                            cs_last = getattr(user, 'coach_state', {}) or {}
                            last_blob = cs_last.get('lifestyle_last') if isinstance(cs_last.get('lifestyle_last'), dict) else {}
                            last_res = last_blob.get('result') if isinstance(last_blob.get('result'), dict) else {}
                            micro_last = last_res.get('microhabits') if isinstance(last_res.get('microhabits'), list) else []

                            chosen = None
                            for mh in micro_last:
                                if not isinstance(mh, dict):
                                    continue
                                if str(mh.get('id') or '').strip() == hid_done:
                                    chosen = mh
                                    break

                            label_done = str((chosen or {}).get('label') or '').strip() or 'micro-hábito'
                            benefit_map = {
                                'breath_5': 'Baja activación del sistema nervioso y mejora tu sensación de control en pocos minutos.',
                                'mobility_3': 'Libera tensión acumulada y mejora la calidad del movimiento para el resto del día.',
                                'sleep_earlier_30': 'Mejora recuperación nocturna y eleva tu capacidad para entrenar mejor mañana.',
                                'walk_3x5': 'Sube actividad diaria sin fatigar; mejora energía y adherencia.',
                                'walk_5_postmeal': 'Ayuda a la regulación glucémica y reduce rigidez por sedentarismo.',
                                'break_60': 'Reduce fatiga por sedentarismo y mejora enfoque físico/mental.',
                                'water_500': 'Mejora hidratación útil para rendimiento, concentración y recuperación.',
                                'sunlight_5': 'Refuerza ritmo circadiano y ayuda a estabilizar energía durante el día.',
                            }
                            benefit = benefit_map.get(hid_done) or 'Te ayuda a sostener constancia hoy sin sobrecargar el sistema.'

                            dhss_last = None
                            try:
                                d0 = last_res.get('dhss') if isinstance(last_res.get('dhss'), dict) else {}
                                dhss_last = int(d0.get('score')) if d0.get('score') is not None else None
                            except Exception:
                                dhss_last = None

                            done_text = (
                                f"✅ Listo. Registré: **{label_done}**.\n"
                                f"**¿Para qué sirve hoy?** {benefit}\n"
                                "**¿Cómo usa esto el sistema?** Lo guardo en tu historial de hoy para ajustar tu lectura de Estado de Hoy y afinar tus próximas recomendaciones sin sobrecargarte."
                            )

                            try:
                                qc_done = ""
                                prompt_done = (
                                    "Eres Quantum Coach. El usuario acaba de marcar un micro-hábito como completado en Estado de Hoy (Lifestyle).\n"
                                    "Responde en español, tono cercano y con contexto (máx 5 líneas, sin formato rígido).\n"
                                    "Incluye: 1) confirmación breve, 2) para qué sirve hoy, 3) cómo usamos este dato en el sistema (DHSS y ajuste de recomendaciones), 4) siguiente paso de 1 línea.\n"
                                    "No pidas foto ni datos adicionales. No uses HTML.\n\n"
                                    f"Micro-hábito registrado: {label_done}\n"
                                    f"Beneficio base: {benefit}\n"
                                    f"DHSS actual (si existe): {dhss_last if dhss_last is not None else 'N/D'}/100\n"
                                    f"Mensaje del usuario: {str(message or '').strip()}\n"
                                )
                                n8n_payload_done = {
                                    'chatInput': prompt_done,
                                    'message': prompt_done,
                                    'sessionId': session_id,
                                    'username': (getattr(user, 'username', '') or ''),
                                    'auth_header': auth_header,
                                    'attachment': '',
                                    'attachment_text': '',
                                    'qaf': {'type': 'exp-007_lifestyle_habit_ack', 'result': {'habit_id': hid_done, 'habit_label': label_done, 'benefit': benefit, 'dhss': dhss_last}},
                                    'system_rules': {
                                        'module': 'exp-007_lifestyle_habit_ack',
                                        'no_medical': True,
                                    },
                                }
                                resp_done = _bench_n8n_post(request, n8n_url, n8n_payload_done, timeout=35)
                                if resp_done.status_code == 200:
                                    try:
                                        data_done = resp_done.json()
                                    except Exception:
                                        data_done = {'output': resp_done.text}
                                    if isinstance(data_done, dict) and isinstance(data_done.get('output'), str):
                                        qc_done = (data_done.get('output') or '').strip()
                                    elif isinstance(data_done, str):
                                        qc_done = data_done.strip()
                                low_done = str(qc_done or '').lower()
                                if qc_done and '<iframe' not in low_done:
                                    done_text = qc_done
                            except Exception:
                                pass

                            return Response({'output': done_text}, status=200)
                        except Exception:
                            return Response({'output': '✅ Listo. Registré ese micro-hábito para hoy.'}, status=200)

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

                    dhss_obj = lifestyle_result.get('dhss') if isinstance(lifestyle_result.get('dhss'), dict) else {}
                    dhss_score = None
                    try:
                        dhss_score = int(dhss_obj.get('score')) if dhss_obj.get('score') is not None else None
                    except Exception:
                        dhss_score = None
                    band = str(dhss_obj.get('band') or '').strip().lower()
                    band_label = (
                        'Recuperación' if band == 'recovery' else
                        'Fatiga' if band == 'fatigue' else
                        'Capacidad moderada' if band == 'moderate' else
                        'Alta capacidad' if band == 'high_capacity' else
                        'Estado'
                    )

                    def _lifestyle_contextual_fallback(res: dict) -> str:
                        if not isinstance(res, dict):
                            return ltext or ''

                        conf = res.get('confidence') if isinstance(res.get('confidence'), dict) else {}
                        conf_pct = None
                        try:
                            conf_pct = int(round(float(conf.get('score')) * 100.0)) if conf.get('score') is not None else None
                        except Exception:
                            conf_pct = None

                        sig = res.get('signals') if isinstance(res.get('signals'), dict) else {}
                        patterns = res.get('patterns') if isinstance(res.get('patterns'), list) else []
                        micro = res.get('microhabits') if isinstance(res.get('microhabits'), list) else []

                        sleep_val = (sig.get('sleep') or {}).get('value') if isinstance(sig.get('sleep'), dict) else None
                        steps_val = (sig.get('steps') or {}).get('value') if isinstance(sig.get('steps'), dict) else None

                        lines = []
                        lines.append('Hoy ya tengo una lectura integrada de tu sistema para ayudarte a decidir con criterio y sin sobrecargarte.')
                        if dhss_score is not None:
                            lines.append(f"DHSS de hoy: **{dhss_score}/100 ({dhss_score}%)** — {band_label}.")

                        if conf_pct is not None:
                            if conf_pct >= 70:
                                lines.append(f"Precisión estimada: **{conf_pct}% (alta)**.")
                            elif conf_pct >= 45:
                                lines.append(f"Precisión estimada: **{conf_pct}% (media)**.")
                            else:
                                lines.append(f"Precisión estimada: **{conf_pct}% (baja)**; conviene completar 1–2 datos para afinar.")

                        quick = []
                        try:
                            if sleep_val is None or float(sleep_val) < 30:
                                quick.append('Sueño: sin dato')
                            else:
                                quick.append(f"Sueño: {int(round(float(sleep_val)/60.0))}h")
                        except Exception:
                            quick.append('Sueño: sin dato')
                        try:
                            if steps_val is None or float(steps_val) < 50:
                                quick.append('Pasos: sin dato')
                            else:
                                quick.append(f"Pasos: {int(round(float(steps_val)))}")
                        except Exception:
                            quick.append('Pasos: sin dato')
                        lines.append('Señales base de hoy: ' + ' | '.join(quick[:2]) + '.')

                        msgs = []
                        for p in patterns[:2]:
                            if isinstance(p, dict) and p.get('message'):
                                msgs.append(str(p.get('message')).strip())
                        if msgs:
                            lines.append('Lectura estratégica: ' + ' '.join(msgs))

                        if band in ('recovery', 'fatigue'):
                            lines.append('Enfoque recomendado: baja carga y alta consistencia (caminata suave o movilidad) para recuperar margen hoy y rendir mejor mañana.')
                        elif band == 'moderate':
                            lines.append('Enfoque recomendado: carga moderada, cuidando técnica y recuperaciones.')
                        else:
                            lines.append('Enfoque recomendado: puedes progresar hoy, manteniendo control de la técnica y la fatiga.')

                        if micro:
                            names = [str(x.get('label')).strip() for x in micro if isinstance(x, dict) and str(x.get('label') or '').strip()]
                            if names:
                                lines.append('Micro-hábitos sugeridos para mover la aguja hoy: ' + '; '.join(names[:3]) + '.')

                        lines.append('Si quieres, te lo ajusto en 10 segundos con Sueño 1/5, Movimiento 1/5 y Estrés 1/5 (equivale a 20%-100%).')
                        return '\n\n'.join([x for x in lines if str(x).strip()]).strip()

                    # Prefacio breve para explicar el examen en su primera activación del flujo.
                    lifestyle_intro = "**Estado de Hoy (Lifestyle)**\n"
                    if dhss_score is not None:
                        lifestyle_intro += f"DHSS: {dhss_score}/100 ({dhss_score}%) — {band_label}.\n"
                    lifestyle_intro += "Acabo de integrar sueño, movimiento y carga diaria para definir tu foco de hoy con el menor riesgo de sobrecarga.\n"

                    # Pasar el resultado a Quantum Coach para una narrativa más natural/humana.
                    try:
                        qc_text = ""
                        prompt = (
                            "Eres Quantum Coach. Convierte este resultado técnico de 'Estado de Hoy' en una explicación clara, cercana y accionable.\n"
                            "Responde SOLO en texto plano (sin HTML, sin <iframe>).\n"
                            "No diagnóstico médico. No promesas absolutas.\n"
                            "No uses plantilla rígida numerada; escribe en tono conversacional con bloques cortos y contexto.\n"
                            "Reglas de salida obligatorias:\n"
                            "- Incluye la línea exacta: DHSS: X/100 (Y%) — [banda].\n"
                            "- Si aparece escala 1/5, conviértela también a porcentaje (20%, 40%, 60%, 80%, 100%).\n"
                            "- Explica qué significa para hoy y cómo impacta mañana.\n"
                            "- Si faltan datos, explica qué falta y por qué mejora la precisión.\n"
                            "- Micro-hábitos: máximo 3, indicando para qué sirve cada uno en una frase breve.\n"
                            "Si faltan datos, dilo sin dramatizar y pide solo lo mínimo.\n\n"
                            f"DHSS calculado: {dhss_score if dhss_score is not None else 'N/D'}/100 ({(str(dhss_score) + '%') if dhss_score is not None else 'N/D'})\n"
                            f"Banda: {band_label}\n"
                            f"Mensaje del usuario: {str(message or '').strip()}\n"
                            f"Resultado Lifestyle: {json.dumps(lifestyle_result, ensure_ascii=False)}\n"
                        )
                        n8n_payload = {
                            'chatInput': prompt,
                            'message': prompt,
                            'sessionId': session_id,
                            'username': (getattr(user, 'username', '') or ''),
                            'auth_header': auth_header,
                            'attachment': '',
                            'attachment_text': '',
                            'qaf': {'type': 'exp-007_lifestyle_explainer', 'result': lifestyle_result},
                            'system_rules': {
                                'module': 'exp-007_lifestyle_explainer',
                                'no_new_buttons': True,
                                'no_medical': True,
                            },
                        }
                        resp = _bench_n8n_post(request, n8n_url, n8n_payload, timeout=45)
                        if resp.status_code == 200:
                            try:
                                data = resp.json()
                            except Exception:
                                data = {'output': resp.text}
                            if isinstance(data, dict) and isinstance(data.get('output'), str):
                                qc_text = (data.get('output') or '').strip()
                            elif isinstance(data, str):
                                qc_text = data.strip()

                        # Guardrails de salida
                        low_qc = str(qc_text or '').lower()
                        if '<iframe' in low_qc:
                            qc_text = ""
                        if re.search(r"\b(env[ií]a|adjunta|adjuntar)\b.*\b(foto|imagen)\b", low_qc):
                            qc_text = ""

                        if qc_text.strip():
                            ltext = qc_text
                        else:
                            _bench_event_note(request, fallback_used=True)
                            ltext = _lifestyle_contextual_fallback(lifestyle_result)
                    except Exception:
                        _bench_event_note(request, fallback_used=True)
                        ltext = _lifestyle_contextual_fallback(lifestyle_result)

                    if ltext:
                        lifestyle_text_for_output_override = (lifestyle_intro + "\n" + ltext).strip()
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[ESTADO DE HOY]\n{lifestyle_text_for_output_override}".strip()

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
                                    pct_i = i * 20
                                    out.append({
                                        'label': f"{prefix_label} {i}/5 ({pct_i}%)",
                                        'type': 'message',
                                        'text': f"{prefix_label} {i}/5 ({pct_i}%)",
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
        except StopIteration:
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

                    # Guardrail UX: no activar motivación solo por saludo/despedida/cortesía.
                    try:
                        is_greeting_or_farewell = bool(
                            re.search(
                                r"\b(hola|buenas|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|saludos|gracias|chao|chau|ad[ií]os|me\s+voy|a\s+dormir|solo\s+saludarte|probarte)\b",
                                msg_low,
                            )
                        )
                        explicit_motivation_text = bool(
                            re.search(
                                r"\b(necesito\s+motivaci[oó]n|quiero\s+motivaci[oó]n|activa\s+motivaci[oó]n|dame\s+motivaci[oó]n|ay[uú]dame\s+a\s+motivarme)\b",
                                msg_low,
                            )
                        )
                        if is_greeting_or_farewell and not explicit_motivation_text:
                            want_motivation = False
                    except Exception:
                        pass

                    if re.search(
                        r"\b("
                        r"motivaci[oó]n|"
                        r"necesito\s+motivaci[oó]n|"
                        r"me\s+cuesta|"
                        r"no\s+quiero|"
                        r"no\s+pude|"
                        r"no\s+tengo\s+ganas|"
                        r"sin\s+ganas|"
                        r"me\s+dio\s+pereza|"
                        r"procrastin\w*|"
                        r"posterg\w*|pospon\w*|"
                        r"me\s+cuesta\s+arrancar|me\s+cuesta\s+empezar|"
                        r"no\s+arranco|no\s+empiezo|"
                        r"sin\s+[aá]nimo|desmotiv\w*|"
                        r"estoy\s+agotad\w*|"
                        r"estoy\s+cansad\w*"
                        r")\b",
                        msg_low,
                    ):
                        want_motivation = True

                    # Activación por intención implícita: pide un plan corto (minutos) para volver a la constancia.
                    # Esto captura frases tipo "dame un plan de 10 minutos para volver a la constancia".
                    if not want_motivation:
                        try:
                            asks_short_plan = bool(re.search(r"\b(plan|rutina|ses[ií]on)\b", msg_low)) and bool(
                                re.search(r"\b(\d{1,2})\s*(min|minutos)\b", msg_low)
                            )
                            mentions_consistency = bool(re.search(r"\b(constancia|h[aá]bito|rutina|racha|volver\s+a\s+la\s+constancia)\b", msg_low))
                            mentions_struggle = bool(
                                re.search(r"\b(no\s+tengo\s+ganas|sin\s+ganas|procrastin\w*|desmotiv\w*|sin\s+[aá]nimo|me\s+cuesta\s+arrancar|me\s+cuesta\s+empezar)\b", msg_low)
                            )
                            if asks_short_plan and mentions_consistency and mentions_struggle:
                                want_motivation = True
                        except Exception:
                            pass

                # Rate limit (heurístico): máximo 1 activación por ventana de 8 horas (3/día).
                # La activación explícita (payload o frases tipo "Necesito motivación") siempre pasa.
                motivation_heuristic_window_to_mark = None
                try:
                    implicit = not (isinstance(mr, dict) or isinstance(ma, dict))
                    msg_low = str(message or '').lower()
                    explicit_text_intent = bool(
                        re.search(
                            r"\b(necesito\s+motivaci[oó]n|quiero\s+motivaci[oó]n|activa\s+motivaci[oó]n|dame\s+motivaci[oó]n|ay[uú]dame\s+a\s+motivarme)\b",
                            msg_low,
                        )
                    )
                    if want_motivation and implicit and not explicit_text_intent and user:
                        cs0 = getattr(user, 'coach_state', {}) or {}
                        rl = cs0.get('motivation_heuristic_rate_limit') if isinstance(cs0.get('motivation_heuristic_rate_limit'), dict) else {}
                        used = rl.get('used') if isinstance(rl.get('used'), dict) else {}

                        today = timezone.localdate().isoformat()
                        if str(rl.get('day') or '') != today:
                            used = {}

                        try:
                            hour = int(getattr(timezone.localtime(timezone.now()), 'hour', 0) or 0)
                        except Exception:
                            hour = 0
                        window = str(max(0, min(2, hour // 8)))

                        if window in used:
                            want_motivation = False
                        else:
                            motivation_heuristic_window_to_mark = window
                except Exception:
                    pass

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

                motivation_requested = bool(want_motivation)

                # Estado del modo Renacer para controlar CTAs y evitar loops de botones.
                renacer_mode_active = False
                try:
                    mem_local = cs.get('motivation_memory') if isinstance(cs.get('motivation_memory'), dict) else {}
                    ren_until = str(mem_local.get('renacer_until') or '').strip()
                    if ren_until:
                        until_d = date.fromisoformat(ren_until[:10])
                        renacer_mode_active = bool(date.today() <= until_d)
                except Exception:
                    renacer_mode_active = False

                if want_motivation:
                    if not _is_premium_active(user):
                        gate_payload = _qaf_premium_gate_payload(
                            'exp-008_motivation',
                            'Motivación psicológica',
                            'Preview: impulso inicial para hoy. El perfil motivacional adaptativo, continuidad y retos personalizados requieren Premium.',
                            request=request,
                            user=user,
                            source='chat_qaf',
                        )
                        motivation_result = gate_payload
                        motivation_text_for_output_override = gate_payload.get('text')
                        motivation_quick_actions_out = _qaf_append_premium_cta(motivation_quick_actions_out, module='exp-008_motivation', source='chat_qaf')
                        raise StopIteration()

                    # Nombre amigable para copy (sin depender de n8n)
                    user_display_name = None
                    try:
                        user_display_name = (getattr(user, 'full_name', '') or '').strip() or (getattr(user, 'username', '') or '').strip() or None
                    except Exception:
                        user_display_name = None

                    # Si viene acción explícita, priorizar reconocimiento breve.
                    try:
                        if isinstance(ma, dict) and ma.get('accept') is True:
                            streak_now = int(getattr(user, 'current_streak', 0) or 0)
                            greeting = (f"Hola {user_display_name},\n" if user_display_name else "Hola,\n")
                            motivation_text_for_output_override = (
                                greeting + "Perfecto. Eso cuenta como constancia de hoy.\n"
                                f"Racha actual: {streak_now} días.\n"
                                "¿Quieres que subamos un poco el reto mañana o mantenemos estabilidad?"
                            )
                    except Exception:
                        pass
                    try:
                        if isinstance(ma, dict) and str(ma.get('mode') or '').strip().lower() == 'renacer_7d':
                            greeting = (f"Hola {user_display_name},\n" if user_display_name else "Hola,\n")
                            motivation_text_for_output_override = (
                                greeting + "Listo: activé Modo Renacer por 7 días.\n"
                                "Objetivo: hábitos mínimos, sin presión, sin culpa.\n"
                                "Hoy solo vamos por el siguiente paso pequeño: 6 minutos y lo celebramos."
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

                    # UX: permitir que el renderer personalice el saludo.
                    try:
                        if user_display_name and isinstance(motivation_result, dict):
                            motivation_result = {**motivation_result, 'user_display_name': user_display_name}
                    except Exception:
                        pass

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

                        # Persistir rate limit de activación heurística por ventana (si aplica)
                        try:
                            if motivation_heuristic_window_to_mark:
                                today = timezone.localdate().isoformat()
                                rl0 = cs2.get('motivation_heuristic_rate_limit') if isinstance(cs2.get('motivation_heuristic_rate_limit'), dict) else {}
                                used0 = rl0.get('used') if isinstance(rl0.get('used'), dict) else {}
                                # reset si cambió el día
                                if str(rl0.get('day') or '') != today:
                                    used0 = {}
                                used1 = dict(used0)
                                used1[str(motivation_heuristic_window_to_mark)] = timezone.now().isoformat()
                                # mantener solo 3 ventanas
                                used1 = {k: used1[k] for k in used1.keys() if str(k) in ('0', '1', '2')}
                                cs2['motivation_heuristic_rate_limit'] = {'day': today, 'used': used1}
                        except Exception:
                            pass

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

                            # Evitar loop: no reofrecer Modo fácil si ya está activo
                            # o si el usuario acaba de pulsar una acción de motivación.
                            just_acted = isinstance(ma, dict)
                            if (not renacer_mode_active) and (not just_acted):
                                motivation_quick_actions_out.append({
                                    'label': '🟡 Modo fácil 7 días',
                                    'type': 'message',
                                    'text': '🟡 Modo fácil 7 días',
                                    'payload': {'motivation_action': {'mode': 'renacer_7d'}},
                                })

                            # Si ya aceptó hoy, ofrecer continuidad más intuitiva en lugar de repetir loop.
                            if isinstance(ma, dict) and ma.get('accept') is True:
                                motivation_quick_actions_out.append({
                                    'label': 'Mantener estabilidad',
                                    'type': 'message',
                                    'text': 'Mantener estabilidad',
                                    'payload': {'motivation_request': {'preferences': {'pressure': 'suave'}}},
                                })
                                motivation_quick_actions_out.append({
                                    'label': 'Subir reto mañana',
                                    'type': 'message',
                                    'text': 'Subir reto mañana',
                                    'payload': {'motivation_request': {'preferences': {'pressure': 'firme'}}},
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
                            return Response(
                                _attach_wow_event_payload(
                                    user,
                                    {
                                        'output': motivation_text_for_output_override,
                                        'quick_actions': out_actions,
                                        'qaf_motivation': motivation_result,
                                    },
                                    event_key='qaf_motivation',
                                    label='Motivación personalizada',
                                )
                            )
                    except Exception:
                        pass
        except StopIteration:
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
                    # Compat UX: si el usuario escribe (o el botón manda) el texto nuevo,
                    # también debe activar el flujo de Exp-009.
                    if re.search(r"\b(progres|evoluci|entrenamiento|estanc|plateau|subir\s+peso|subir\s+carga|mas\s+reps|cardio|correr|bici|eliptica)\b", msg_low):
                        want_prog = True

                msg_low = str(message or '').strip().lower()
                from_prog_button_text = msg_low in ("evolución de entrenamiento", "evolucion de entrenamiento")

                if want_prog:
                    if not _is_premium_active(user):
                        gate_payload = _qaf_premium_gate_payload(
                            'exp-009_progression',
                            'Evolución de entrenamiento',
                            'Preview: sugerencia inicial del siguiente paso. Para progresión inteligente completa por historial y señales, activa Premium.',
                            request=request,
                            user=user,
                            source='chat_qaf',
                        )
                        progression_result = gate_payload
                        progression_text_for_output_override = gate_payload.get('text')
                        progression_quick_actions_out = _qaf_append_premium_cta(progression_quick_actions_out, module='exp-009_progression', source='chat_qaf')
                        raise StopIteration()

                    from .qaf_progression.engine import evaluate_progression, render_professional_summary, parse_strength_line

                    cs = getattr(user, 'coach_state', {}) or {}
                    mem = cs.get('progression_history') if isinstance(cs.get('progression_history'), dict) else {}

                    # Draft por sesión: mantiene selección parcial (modalidad/RPE/%/ui) mientras falten inputs.
                    draft_by_session = cs.get('progression_draft') if isinstance(cs.get('progression_draft'), dict) else {}
                    sess_key = str(session_id or '').strip() or 'default'
                    draft = draft_by_session.get(sess_key) if isinstance(draft_by_session.get(sess_key), dict) else {}
                    had_draft = bool(draft)

                    # Base session payload
                    session = draft.get('session') if isinstance(draft.get('session'), dict) else {}
                    strength = draft.get('strength') if isinstance(draft.get('strength'), dict) else None
                    cardio = draft.get('cardio') if isinstance(draft.get('cardio'), dict) else None
                    if isinstance(pr, dict):
                        # override/merge con lo que venga explícito
                        if isinstance(pr.get('session'), dict):
                            session = {**session, **pr.get('session')}
                        if isinstance(pr.get('strength'), dict):
                            strength = pr.get('strength')
                        if isinstance(pr.get('cardio'), dict):
                            cardio = pr.get('cardio')

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

                        # merge ui state
                        if isinstance(pa.get('ui'), dict):
                            ui0 = draft.get('ui') if isinstance(draft.get('ui'), dict) else {}
                            draft['ui'] = {**ui0, **pa.get('ui')}

                    progression_result = evaluate_progression({
                        'session': session,
                        'strength': strength,
                        'cardio': cardio,
                        'history': mem,
                        'signals': signals,
                    }).payload

                    # UX: en pasos con botones (pa) o si ya había draft, no repetir el bloque largo.
                    try:
                        if isinstance(progression_result, dict):
                            ui_out = progression_result.get('ui') if isinstance(progression_result.get('ui'), dict) else {}
                            # Regla:
                            # - primer paso: show_intro=True
                            # - pasos intermedios (botones/draft): show_intro=False
                            # - paso final accepted: show_intro=True para entregar cierre con valor
                            show_intro = (not (isinstance(pa, dict) or had_draft))
                            if progression_result.get('decision') == 'accepted':
                                show_intro = True
                            progression_result = {**progression_result, 'ui': {**ui_out, 'show_intro': bool(show_intro)}}
                    except Exception:
                        pass

                    # UX: pasar un nombre amigable al renderer (sin depender de n8n).
                    try:
                        display = (getattr(user, 'full_name', '') or '').strip() or (getattr(user, 'username', '') or '').strip()
                        if display and isinstance(progression_result, dict):
                            progression_result = {**progression_result, 'user_display_name': display}
                    except Exception:
                        pass

                    ptext = render_professional_summary(progression_result)
                    if ptext:
                        progression_text_for_output_override = ptext
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + f"[EVOLUCIÓN DE ENTRENAMIENTO]\n{ptext}".strip()

                    # Quick actions guiadas para missing (sin UI nueva)
                    try:
                        missing = ((progression_result.get('confidence') or {}).get('missing') or [])
                        missing = [str(x) for x in missing]

                        def _rpe_buttons():
                            # UX: 5 botones (2-4-6-8-10) para evitar paging.
                            out = []
                            for i in (2, 4, 6, 8, 10):
                                out.append({
                                    'label': f"{i}",
                                    'type': 'message',
                                    'text': f"{i}",
                                    'payload': {'progression_action': {'session': {'rpe_1_10': i}}},
                                })
                            return out

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
                            progression_quick_actions_out.extend(_rpe_buttons()[:6])
                        elif 'completion_pct' in missing and len(progression_quick_actions_out) < 6:
                            # completion presets
                            opts = [
                                (0.2, '20%'),
                                (0.4, '40%'),
                                (0.6, '60%'),
                                (0.8, '80%'),
                                (1.0, '100%'),
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

                    # Persistir draft mientras falten inputs; limpiar cuando accepted.
                    try:
                        if isinstance(progression_result, dict):
                            missing_now = ((progression_result.get('confidence') or {}).get('missing') or [])
                            missing_now = [str(x) for x in missing_now]
                            cs2 = dict(cs)
                            dbs2 = dict(draft_by_session)
                            if progression_result.get('decision') == 'accepted':
                                dbs2.pop(sess_key, None)
                            else:
                                # Guardar el estado parcial
                                dbs2[sess_key] = {
                                    'session': session,
                                    'strength': strength,
                                    'cardio': cardio,
                                    'ui': draft.get('ui') if isinstance(draft.get('ui'), dict) else {},
                                    'missing': missing_now,
                                    'updated_at': timezone.now().isoformat(),
                                }
                                # limitar a 5 sesiones para no inflar JSON
                                if len(dbs2) > 5:
                                    keys = sorted([k for k in dbs2.keys() if isinstance(k, str)])
                                    for k in keys[:-5]:
                                        dbs2.pop(k, None)
                            cs2['progression_draft'] = dbs2
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    # Responder directo si viene de botones/payloads
                    try:
                        if (isinstance(pr, dict) or isinstance(pa, dict) or from_prog_button_text) and progression_text_for_output_override:
                            existing = quick_actions_out if isinstance(quick_actions_out, list) else []
                            existing2 = [x for x in existing if isinstance(x, dict)]
                            out_actions = (existing2 + progression_quick_actions_out)[:6] if progression_quick_actions_out else existing2[:6]
                            return Response(
                                _attach_wow_event_payload(
                                    user,
                                    {
                                        'output': progression_text_for_output_override,
                                        'quick_actions': out_actions,
                                        'qaf_progression': progression_result,
                                    },
                                    event_key='qaf_progression',
                                    label='Progresión de entrenamiento',
                                )
                            )
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

        # Normalizar CTAs del frontend para Vitalidad de la Piel
        try:
            if isinstance(request.data, dict) and isinstance(request.data.get('skin_habits_request'), dict):
                # (No esperado) mantener compat
                pass
        except Exception:
            pass

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

        def _vision_bool(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "y", "on", "si", "sí")
            return False

        def _vision_route_conf(v: Any) -> float:
            try:
                score = float(v)
            except Exception:
                return 0.0
            if score < 0:
                return 0.0
            if score > 1:
                return 1.0
            return score

        def _vision_is_screenshot_or_ui(vp: dict[str, Any]) -> bool:
            try:
                if _vision_bool(vp.get('is_screenshot_or_ui')):
                    return True
                if _vision_bool(vp.get('is_document_like')):
                    return True
                notes_low = str(vp.get('notes') or '').strip().lower()
                return bool(re.search(r"\b(captura|screenshot|pantalla|web|sitio|p[aá]gina|ui|interfaz|dashboard|chat|whatsapp|instagram|youtube|pdf|documento)\b", notes_low))
            except Exception:
                return False

        def _vision_can_activate_health(vp: dict[str, Any]) -> bool:
            if _vision_is_screenshot_or_ui(vp):
                return False
            conf = _vision_route_conf(vp.get('route_confidence'))
            if conf < 0.65:
                return False
            closeup = _vision_bool(vp.get('is_closeup_skin_or_muscle'))
            has_person = _vision_bool(vp.get('has_person'))
            return bool(closeup or has_person)

        def _vision_can_activate_training(vp: dict[str, Any]) -> bool:
            if _vision_is_screenshot_or_ui(vp):
                return False
            conf = _vision_route_conf(vp.get('route_confidence'))
            if conf < 0.65:
                return False
            has_person = _vision_bool(vp.get('has_person'))
            return has_person
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
                            normalized_url = _normalize_attachment_url(str(attachment_url))
                            container_name, blob_name = _extract_blob_ref_from_url(normalized_url)
                            if container_name == _chat_attachment_container() and blob_name:
                                blob_name = _resolve_blob_name(container_name, blob_name) or blob_name
                                safe_username = user.username.replace("/", "_")
                                allowed = bool(blob_name.startswith(f"{safe_username}/")) or _is_signed_chat_attachment_url(normalized_url)
                                if allowed:
                                    resp = requests.get(normalized_url, timeout=20)
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

                                # Guardrail UX: capturas/pantallas no deben secuestrar la conversación
                                # hacia experiencias si el usuario no lo pidió explícitamente.
                                if _vision_is_screenshot_or_ui(parsed):
                                    route_now = str(parsed.get('route') or '').strip().lower()
                                    if route_now in ('training', 'entrenamiento', 'health', 'salud') and not user_explicit_image_experience:
                                        parsed['route'] = 'quantum'
                                        parsed['route_confidence'] = 0.51
                                        parsed['routing_guardrail'] = 'screenshot_ui_to_quantum'
                                        vision_route_bias_guard = True
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
                    can_activate = _vision_can_activate_health(vision_parsed) or user_explicit_image_experience
                    if not can_activate:
                        vision_route_bias_guard = True
                        # No forzar experiencias en imágenes ambiguas/capturas.
                        # Dejamos que n8n converse de forma amplia con el contexto ya extraído.
                        raise StopIteration()

                    msg_low2 = str(message or '').strip().lower()
                    cs_local = getattr(user, 'coach_state', {}) or {}
                    prefer_skin = False
                    try:
                        if str(cs_local.get('health_mode') or '').strip().lower() == 'skin':
                            prefer_skin = True
                        if re.search(r"\b(vitalidad\s+de\s+la\s+pi?e?l|vitalidad\s+pi?e?l|pi?e?l|skincare|skin\s*health)\b", msg_low2):
                            prefer_skin = True
                        if ("vitalidad" in msg_low2) and ("piel" in msg_low2 or "peil" in msg_low2):
                            prefer_skin = True
                    except Exception:
                        prefer_skin = False

                    # Guardar attachment como pendiente para que el usuario pueda elegir Vitalidad sin re-adjuntar.
                    try:
                        if user:
                            cs2 = dict(cs_local)
                            cs2['skin_pending_attachment'] = {
                                'attachment_url': str(attachment_url),
                                'until': (timezone.now() + timedelta(minutes=10)).isoformat(),
                            }
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                    except Exception:
                        pass

                    if prefer_skin:
                        vision_experience_activated = 'exp-011_skin_health'
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                            "**Vitalidad de la Piel**\n"
                            "Ya tengo tu foto. Para mantener esta experiencia limpia, sigo con **Vitalidad de la Piel**.\n\n"
                            "¿Listo para el análisis?"
                        )
                        qa = quick_actions_out if isinstance(quick_actions_out, list) else []
                        qa.extend([
                            {'label': 'Vitalidad de la Piel', 'type': 'message', 'text': 'Vitalidad de la Piel'},
                            {'label': 'Cancelar', 'type': 'skin_cancel'},
                        ])
                        quick_actions_out = qa[:6]
                    else:
                        vision_experience_activated = 'exp-health-menu'
                        attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                            "**Salud / Imagen**\n"
                            "Detecté una imagen tipo salud (primer plano de piel/músculo/rostro).\n"
                            "¿Qué quieres hacer con esta foto?\n"
                            "- Medición del progreso muscular\n"
                            "- Vitalidad de la Piel\n"
                            "Responde con una opción."
                        )
                        qa = quick_actions_out if isinstance(quick_actions_out, list) else []
                        qa.extend([
                            {'label': 'Medición del progreso muscular', 'type': 'message', 'text': 'Medición del progreso muscular'},
                            {'label': 'Vitalidad de la Piel', 'type': 'message', 'text': 'Vitalidad de la Piel'},
                        ])
                        quick_actions_out = qa[:6]
        except StopIteration:
            pass
        except Exception:
            pass

        # Router: si la imagen cae en Entrenamiento, guiar captura postural (sin intentar calorías).
        try:
            if attachment_url and isinstance(vision_parsed, dict):
                vr = str(vision_parsed.get('route') or '').strip().lower()
                if vr in ('training', 'entrenamiento'):
                    can_activate = _vision_can_activate_training(vision_parsed) or user_explicit_image_experience
                    if not can_activate:
                        vision_route_bias_guard = True
                        # Evitar sesgo: continuar conversación amplia en n8n sin imponer flujo de postura.
                        raise StopIteration()

                    # Guardrail UX: no mezclar check-ins/metabólico si el usuario está en modo entrenamiento por imagen.
                    try:
                        suppress_weekly_checkins = True
                    except Exception:
                        pass

                    vision_experience_activated = 'exp-006/012/013_menu'
                    attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                        "**Entrenamiento / Imagen**\n"
                        "Puedo ayudarte con técnica/postura, pero necesito 2 fotos: frontal y lateral (cuerpo completo, buena luz, cámara a la altura del pecho, 2–3m).\n"
                        "Si solo es una selfie o una foto casual, también puedo responder como Quantum Coach.\n\n"
                        "Si tu objetivo es **comparar progreso muscular** semana a semana, también lo podemos hacer con fotos (sin prometer cm exactos)."
                    )
                    qa = quick_actions_out if isinstance(quick_actions_out, list) else []
                    qa.extend([
                        {'label': 'Postura', 'type': 'posture_start'},
                        {'label': 'Arquitectura Corporal', 'type': 'pp_start'},
                        {'label': 'Alta Costura Inteligente', 'type': 'shape_start'},
                        {'label': 'Medición del progreso muscular', 'type': 'message', 'text': 'Medición del progreso muscular'},
                        {'label': 'Tomar foto', 'type': 'open_camera'},
                        {'label': 'Adjuntar foto', 'type': 'open_attach'},
                    ])
                    quick_actions_out = qa[:6]
        except StopIteration:
            pass
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

                # Gamificación Exp-001: acreditar experiencia de calorías cuando se ejecuta.
                try:
                    if user:
                        award_wow_event(user, event_key='qaf_calories', label='Calorías inteligentes evaluadas')
                except Exception:
                    pass

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
                metabolic_start_payload = request.data.get('metabolic_profile_request') if isinstance(request.data, dict) else None
                explicit_metabolic_start = isinstance(metabolic_start_payload, dict)
                if not explicit_metabolic_start:
                    try:
                        msg_low_met = str(message or '').strip().lower()
                        explicit_metabolic_start = bool(
                            re.search(r"\b(iniciar|inicia|empezar|activar)\b", msg_low_met)
                            and re.search(r"\b(perfil\s+metab[oó]lico|metab[oó]lic)\b", msg_low_met)
                        )
                    except Exception:
                        explicit_metabolic_start = False

                if not explicit_metabolic_start:
                    try:
                        svc_int = request.data.get('service_intent') if isinstance(request.data, dict) and isinstance(request.data.get('service_intent'), dict) else {}
                        svc_exp = str(svc_int.get('experience') or '').strip().lower()
                        svc_act = str(svc_int.get('action') or '').strip().lower()
                        explicit_metabolic_start = bool(svc_exp == 'exp-003_metabolic_profile' and svc_act in ('start_new', 'start', 'new_eval'))
                    except Exception:
                        explicit_metabolic_start = False

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

                # Guardrail UX: no interrumpir flujos/experimentos.
                # Solo pedimos peso semanal cuando el usuario está idle (mensaje vacío, sin adjunto)
                # o cuando explícitamente pidió perfil metabólico.
                should_prompt = (
                    (cur_avg is None)
                    and (
                        explicit_metabolic_start
                        or (
                            (prompted_week != week_id_now)
                            and (snoozed_week != week_id_now)
                            and (
                                user_asked_metabolic
                                or (
                                    not (str(message or '').strip())
                                    and not attachment_url
                                    and not isinstance(request.data.get('posture_request') if isinstance(request.data, dict) else None, dict)
                                    and not isinstance(request.data.get('muscle_measure_request') if isinstance(request.data, dict) else None, dict)
                                    and not isinstance(request.data.get('shape_presence_request') if isinstance(request.data, dict) else None, dict)
                                    and not isinstance(request.data.get('posture_proportion_request') if isinstance(request.data, dict) else None, dict)
                                )
                            )
                        )
                    )
                )
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

                    # Importante: NO anexar esto a attachment_text.
                    # Si se añade al prompt de n8n, sesga conversaciones no relacionadas.

                # Fast-path UX: si el usuario inició explícitamente Perfil Metabólico
                # y aún falta check-in semanal, responder aquí y no delegar a n8n.
                if explicit_metabolic_start and cur_avg is None:
                    qa = [x for x in quick_actions_out if isinstance(x, dict)][:6]
                    out = (
                        "**Perfil Metabólico**\n"
                        "Para calcularlo bien hoy necesito completar tu check-in semanal (1 dato de peso y, si falta, sexo biológico).\n"
                        "Con eso te entrego TMB/TDEE y recomendación calórica ajustada."
                    )
                    return Response({'output': out, 'quick_actions': qa}, status=200)

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
                        # Evitar anexar al attachment_text: sesga el prompt hacia n8n.

                    # Fast-path UX: inicio explícito debe devolver el resultado ya calculado,
                    # sin depender de respuesta posterior del modelo conversacional.
                    if explicit_metabolic_start and metabolic_text_for_output_override:
                        return Response(
                            _attach_wow_event_payload(
                                user,
                                {
                                    'output': metabolic_text_for_output_override,
                                    'qaf_metabolic': metabolic_result,
                                    'quick_actions': [
                                        {'label': 'Menú semanal', 'type': 'message', 'text': 'Menú semanal', 'payload': {'meal_plan_request': {'variety': 'normal', 'meals_per_day': 3}}},
                                        {'label': 'Tendencia 6 semanas', 'type': 'message', 'text': 'Tendencia 6 semanas', 'payload': {'body_trend_request': {}}},
                                        {'label': 'Estado de hoy', 'type': 'message', 'text': 'Estado de hoy', 'payload': {'lifestyle_request': {'days': 14}}},
                                    ],
                                },
                                event_key='qaf_metabolic_profile',
                                label='Perfil metabólico calculado',
                            ),
                            status=200,
                        )
        except Exception as ex:
            print(f"QAF metabolic profile warning: {ex}")


        # Construir prompt enriquecido (solo con texto real si se pudo)
        final_input = message or "Analisis de archivo adjunto"
        if vision_route_bias_guard:
            final_input = (
                "INSTRUCCION DE CONTEXTO: La imagen actual parece captura/pantalla o no es suficiente para activar una experiencia específica. "
                "No fuerces módulos de coach; prioriza conversación amplia, neutral y útil, y ofrece opciones solo si el usuario las pide.\n\n"
                + final_input
            )
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

        # Motor de cognición (QAF): decisión determinista para que n8n/LLM solo narre.
        # No cambia UX: solo enriquece payload y, en modo quantum o baja claridad, añade un resumen corto.
        qaf_cognition = None
        qaf_cognition_summary = ""
        protocol_cognitive_body = None
        protocol_cognitive_body_summary = ""
        try:
            if user is not None:
                from .qaf_cognition.engine import evaluate_cognition

                week_id_now = _week_id()
                user_profile = {
                    'goal_type': getattr(user, 'goal_type', None),
                    'activity_level': getattr(user, 'activity_level', None),
                }

                observations = {}
                try:
                    if isinstance(request.data, dict) and isinstance(request.data.get('qaf_context'), dict):
                        observations = dict(request.data.get('qaf_context') or {})
                except Exception:
                    observations = {}
                if isinstance(vision_parsed, dict):
                    observations.setdefault('vision', vision_parsed)

                qaf_cognition = evaluate_cognition(
                    user_profile=user_profile,
                    coach_state=getattr(user, 'coach_state', {}) or {},
                    coach_weekly_state=getattr(user, 'coach_weekly_state', {}) or {},
                    observations=observations,
                    message=message,
                    week_id=week_id_now,
                    locale='es-CO',
                )

                if isinstance(qaf_cognition, dict):
                    dec = qaf_cognition.get('decision') if isinstance(qaf_cognition.get('decision'), dict) else {}
                    mode = str(dec.get('mode') or '').strip().lower()
                    dtyp = str(dec.get('type') or '').strip().lower()
                    if hito5_inject_qaf and (mode in ('quantum',) or dtyp in ('ask_clarifying', 'needs_confirmation')):
                        actions = dec.get('next_3_actions') if isinstance(dec.get('next_3_actions'), list) else []
                        titles = [str(a.get('title') or '').strip() for a in actions if isinstance(a, dict) and a.get('title')]
                        titles = [t for t in titles if t][:3]
                        follow = dec.get('follow_up_questions') if isinstance(dec.get('follow_up_questions'), list) else []
                        prompts = [str(q.get('prompt') or '').strip() for q in follow if isinstance(q, dict) and q.get('prompt')]
                        prompts = [p for p in prompts if p][:2]

                        lines = []
                        if mode:
                            lines.append(f"modo: {mode}")
                        if titles:
                            lines.append("acciones_3: " + " | ".join(titles))
                        if prompts:
                            lines.append("pregunta: " + " | ".join(prompts))
                        qaf_cognition_summary = "\n".join(lines).strip()

                        if qaf_cognition_summary:
                            attachment_text = ((attachment_text or '').strip() + "\n\n" if (attachment_text or '').strip() else "") + (
                                "[QAF / COGNICIÓN]\n" + qaf_cognition_summary
                            )

                # Protocolo Cognitivo Corporal (determinista): consolida experiencias en estado evolutivo.
                try:
                    from .protocol_cognitive_body.engine import evaluate_protocol

                    proto = evaluate_protocol(
                        user_profile=user_profile,
                        coach_state=getattr(user, 'coach_state', {}) or {},
                        coach_weekly_state=getattr(user, 'coach_weekly_state', {}) or {},
                        qaf_cognition=qaf_cognition if isinstance(qaf_cognition, dict) else None,
                        observations=observations,
                        week_id=week_id_now,
                        locale='es-CO',
                    ).payload
                    protocol_cognitive_body = proto if isinstance(proto, dict) else None

                    # Persistir (coach_state + weekly)
                    if isinstance(protocol_cognitive_body, dict):
                        try:
                            cs2 = dict(getattr(user, 'coach_state', {}) or {})
                            cs2['protocol_cognitive_body'] = protocol_cognitive_body
                            user.coach_state = cs2
                            user.coach_state_updated_at = timezone.now()
                            user.save(update_fields=['coach_state', 'coach_state_updated_at'])
                        except Exception:
                            pass

                        try:
                            ws2 = dict(getattr(user, 'coach_weekly_state', {}) or {})
                            pw = ws2.get('protocol_cognitive_body') if isinstance(ws2.get('protocol_cognitive_body'), dict) else {}
                            pw2 = dict(pw)
                            pw2[week_id_now] = {'result': protocol_cognitive_body, 'updated_at': timezone.now().isoformat(), 'week_id': week_id_now}
                            ws2['protocol_cognitive_body'] = pw2
                            user.coach_weekly_state = ws2
                            user.coach_weekly_updated_at = timezone.now()
                            user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
                        except Exception:
                            pass

                        # Resumen corto para n8n
                        try:
                            summ = protocol_cognitive_body.get('summary') if isinstance(protocol_cognitive_body.get('summary'), dict) else {}
                            ev = str(summ.get('evolution_state') or '').strip()
                            foc = str(summ.get('primary_focus') or '').strip()
                            goal = str(summ.get('week_goal') or '').strip()
                            restr = summ.get('restrictions') if isinstance(summ.get('restrictions'), list) else []
                            restr = [str(x).strip() for x in restr if str(x).strip()][:2]
                            lines = []
                            if ev:
                                lines.append(f"estado: {ev}")
                            if foc:
                                lines.append(f"foco: {foc}")
                            if goal:
                                lines.append(f"meta_semana: {goal}")
                            if restr:
                                lines.append("restricciones: " + " | ".join(restr))
                            protocol_cognitive_body_summary = "\n".join(lines).strip()
                        except Exception:
                            protocol_cognitive_body_summary = ""
                except Exception:
                    protocol_cognitive_body = None
                    protocol_cognitive_body_summary = ""
        except Exception:
            qaf_cognition = None
            qaf_cognition_summary = ""
            protocol_cognitive_body = None
            protocol_cognitive_body_summary = ""

        payload = {
            "chatInput": final_input,
            "system_rules": {
                "professional_focus": professional_rule,
                "qaf_cognition_summary": qaf_cognition_summary,
                "protocol_cognitive_body_summary": protocol_cognitive_body_summary,
                "conversation_mode": "open" if vision_route_bias_guard else "guided",
                "avoid_experience_bias": bool(vision_route_bias_guard),
                "vision_experience_activated": vision_experience_activated,
            },
            "message": message,
            "sessionId": session_id,
            "username": username_for_payload,
            "auth_header": auth_header,
            "attachment": attachment_url_for_n8n,
            "attachment_text": attachment_text,
            "attachment_text_diagnostic": attachment_text_diagnostic,
            "qaf": qaf_result,
            "qaf_cognition": qaf_cognition if hito5_inject_qaf else None,
            "protocol_cognitive_body": protocol_cognitive_body if hito5_inject_qaf else None,
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

        try:
            payload.setdefault("release", {})
            payload["release"]["hito5"] = {
                "enabled": bool(release_ctx.get("enabled")),
                "variant": str(release_ctx.get("variant") or ""),
                "assigned": bool(release_ctx.get("assigned")),
                "bucket": release_ctx.get("bucket"),
                "canary_percent": release_ctx.get("canary_percent"),
                "rollback_force": bool(release_ctx.get("rollback_force")),
            }
        except Exception:
            pass

        try:
            if hito5_inject_qaf and isinstance(qaf_cognition, dict):
                decision = qaf_cognition.get("decision") if isinstance(qaf_cognition.get("decision"), dict) else {}
                flags = qaf_cognition.get("flags") if isinstance(qaf_cognition.get("flags"), dict) else {}
                needs_human = bool(flags.get("human_validation_required")) or str(decision.get("type") or "") in ("needs_confirmation", "ask_clarifying")
                if needs_human:
                    payload["human_in_the_loop"] = {
                        "required": True,
                        "reason": str(decision.get("type") or "needs_confirmation"),
                        "mode": str(decision.get("mode") or ""),
                    }
                    payload.setdefault("system_rules", {})
                    payload["system_rules"]["human_in_the_loop_required"] = True
                    payload["system_rules"]["llm_role"] = "narrate_only"
        except Exception:
            pass

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

        use_fused_runtime = _hito5_should_use_fused_runtime(release_ctx)
        _bench_event_note(request, hito5_fused_runtime=bool(use_fused_runtime))

        if use_fused_runtime:
            preferred_outputs = [
                qaf_text_for_output_override,
                metabolic_text_for_output_override,
                posture_text_for_output_override,
                lifestyle_text_for_output_override,
                motivation_text_for_output_override,
                progression_text_for_output_override,
                meal_plan_text_for_output_override,
                body_trend_text_for_output_override,
            ]

            output_text = _hito5_compose_fused_output(
                message=message,
                qaf_cognition=qaf_cognition if isinstance(qaf_cognition, dict) else None,
                protocol_cognitive_body=protocol_cognitive_body if isinstance(protocol_cognitive_body, dict) else None,
                preferred_outputs=preferred_outputs,
            )

            data = {
                "output": output_text,
                "source": "hito5_fused_local",
            }

            try:
                merged_quick_actions = []
                for block in [
                    quick_actions_out,
                    posture_quick_actions_out,
                    lifestyle_quick_actions_out,
                    motivation_quick_actions_out,
                    progression_quick_actions_out,
                ]:
                    if isinstance(block, list):
                        merged_quick_actions.extend([x for x in block if isinstance(x, dict)])
                if merged_quick_actions:
                    data["quick_actions"] = merged_quick_actions[:6]
            except Exception:
                pass

            for key, value in [
                ("qaf", qaf_result),
                ("qaf_metabolic", metabolic_result),
                ("qaf_posture", posture_result),
                ("qaf_lifestyle", lifestyle_result),
                ("qaf_motivation", motivation_result),
                ("qaf_progression", progression_result),
                ("qaf_meal_plan", meal_plan_result),
                ("qaf_body_trend", body_trend_result),
            ]:
                if isinstance(value, dict):
                    data[key] = value

            guardrail_applied = False
            try:
                if hito5_inject_qaf:
                    out2, applied = _hito5_apply_response_guardrail(
                        str(data.get("output") or ""),
                        qaf_cognition if isinstance(qaf_cognition, dict) else None,
                    )
                    data["output"] = out2
                    guardrail_applied = bool(applied)
                    if applied:
                        _bench_event_note(request, fallback_used=True, hito5_guardrail_applied=True)
            except Exception:
                pass

            try:
                data.setdefault("release", {})
                data["release"]["hito5"] = {
                    "enabled": bool(release_ctx.get("enabled")),
                    "variant": str(release_ctx.get("variant") or ""),
                    "assigned": bool(release_ctx.get("assigned")),
                    "rollback_force": bool(release_ctx.get("rollback_force")),
                    "runtime": "fused",
                }
            except Exception:
                pass

            _hito5_write_trace(
                request=request,
                user=user,
                session_id=session_id,
                release_ctx=release_ctx,
                qaf_cognition=qaf_cognition if isinstance(qaf_cognition, dict) else None,
                protocol_cognitive_body=protocol_cognitive_body if isinstance(protocol_cognitive_body, dict) else None,
                status_code=200,
                guardrail_applied=guardrail_applied,
                error="",
            )
            return Response(data)

        # 3. Enviar a n8n
        # Timeout corto por si n8n tarda
        response = _bench_n8n_post(request, n8n_url, payload, timeout=60)
        
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

            guardrail_applied = False
            try:
                if isinstance(data, dict) and hito5_inject_qaf:
                    out = str(data.get("output") or "")
                    out2, applied = _hito5_apply_response_guardrail(out, qaf_cognition if isinstance(qaf_cognition, dict) else None)
                    data["output"] = out2
                    guardrail_applied = bool(applied)
                    if applied:
                        _bench_event_note(request, fallback_used=True, hito5_guardrail_applied=True)
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
                    # Guardrail UX: no preprender salida metabólica si el usuario no la pidió.
                    if isinstance(out_text, str) and metabolic_text_for_output_override and user_asked_metabolic:
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
                    if posture_requested and posture_text_for_output_override:
                        data['output'] = posture_text_for_output_override
                    elif isinstance(out_text, str) and posture_text_for_output_override:
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
                    if lifestyle_requested and lifestyle_text_for_output_override:
                        data['output'] = lifestyle_text_for_output_override
                    elif isinstance(out_text, str) and lifestyle_text_for_output_override:
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
                    if motivation_requested and motivation_text_for_output_override:
                        data['output'] = motivation_text_for_output_override
                    elif isinstance(out_text, str) and motivation_text_for_output_override:
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
            try:
                if isinstance(data, dict):
                    data.setdefault("release", {})
                    data["release"]["hito5"] = {
                        "enabled": bool(release_ctx.get("enabled")),
                        "variant": str(release_ctx.get("variant") or ""),
                        "assigned": bool(release_ctx.get("assigned")),
                        "rollback_force": bool(release_ctx.get("rollback_force")),
                        "runtime": "n8n",
                    }
            except Exception:
                pass

            _hito5_write_trace(
                request=request,
                user=user,
                session_id=session_id,
                release_ctx=release_ctx,
                qaf_cognition=qaf_cognition if isinstance(qaf_cognition, dict) else None,
                protocol_cognitive_body=protocol_cognitive_body if isinstance(protocol_cognitive_body, dict) else None,
                status_code=200,
                guardrail_applied=guardrail_applied,
                error="",
            )
            return Response(data)
        else:
            # Intentar obtener mensaje de error de n8n
            try:
                err_body = response.json()
                err_msg = err_body.get('message') or response.text
            except:
                err_msg = response.text
                
            print(f"Error n8n body: {err_msg}")
            _hito5_write_trace(
                request=request,
                user=user,
                session_id=session_id,
                release_ctx=release_ctx,
                qaf_cognition=qaf_cognition if isinstance(qaf_cognition, dict) else None,
                protocol_cognitive_body=protocol_cognitive_body if isinstance(protocol_cognitive_body, dict) else None,
                status_code=response.status_code,
                guardrail_applied=False,
                error=str(err_msg or "")[:400],
            )
            return Response({'error': f"n8n Error ({response.status_code}): {err_msg}"}, status=502)

    except Exception as e:
        print(f"Error chat_n8n: {e}")
        return Response({'error': str(e)}, status=500)


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_cognition_evaluate(request):
    """Motor de cognición QAF (determinista) para orquestación (n8n) y coherencia global.

    Entrada (JSON):
      {
        "week_id"?: "YYYY-Www",
        "message"?: str,
        "observations"?: object,  # señales opcionales (por ejemplo outputs de experimentos en la interacción)
        "locale"?: str
      }

    Requiere usuario autenticado (JWT).
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    week_id = str(payload.get('week_id') or _week_id())
    locale = (payload.get('locale') or '').strip() or 'es-CO'
    message = str(payload.get('message') or payload.get('text') or '').strip()

    observations = payload.get('observations')
    # Compat: algunos callers envían `qaf_context` como contenedor genérico
    if observations is None and isinstance(payload.get('qaf_context'), dict):
        observations = payload.get('qaf_context')
    observations = observations if isinstance(observations, dict) else {}

    user_profile = {
        'goal_type': getattr(user, 'goal_type', None),
        'activity_level': getattr(user, 'activity_level', None),
    }

    from .qaf_cognition.engine import evaluate_cognition

    result = evaluate_cognition(
        user_profile=user_profile,
        coach_state=getattr(user, 'coach_state', {}) or {},
        coach_weekly_state=getattr(user, 'coach_weekly_state', {}) or {},
        observations=observations,
        message=message,
        week_id=week_id,
        locale=locale,
    )

    return Response(_attach_wow_event_payload(user, result, event_key='qaf_cognition', label='Motor de cognición activado'))


@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def hito5_release_status(request):
    """Estado operativo de Hito 5: entrenamiento, release y KPIs runtime."""
    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    now = timezone.now()
    since_24h = now - timedelta(hours=24)

    release_cfg = {
        "enabled": _hito5_env_bool("HITO5_RELEASE_ENABLED", False),
        "canary_percent": _hito5_env_float("HITO5_CANARY_PERCENT", 10.0),
        "rollback_force": _hito5_env_bool("HITO5_ROLLBACK_FORCE", False),
        "fused_mode_enabled": _hito5_env_bool("HITO5_FUSED_MODE_ENABLED", False),
        "fused_mode_canary_only": _hito5_env_bool("HITO5_FUSED_MODE_CANARY_ONLY", True),
    }

    train_manifest = None
    alignment_report = None
    try:
        base = Path(__file__).resolve().parents[1] / "llmops" / "hito4" / "output"
        manifest_path = base / "manifest.json"
        report_path = base / "alignment_report.json"
        if manifest_path.exists():
            train_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report_path.exists():
            alignment_report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        train_manifest = None
        alignment_report = None

    runtime_qs = AuditLog.objects.filter(action="llm.hito5.inference", occurred_at__gte=since_24h)
    total = runtime_qs.count()
    success = runtime_qs.filter(after_json__status_code=200).count()
    guardrail_hits = runtime_qs.filter(after_json__guardrail_applied=True).count()
    canary_hits = runtime_qs.filter(before_json__release__variant="canary").count()

    decision_counts = {}
    for row in runtime_qs.values_list("before_json", flat=True):
        if not isinstance(row, dict):
            continue
        decision_type = str(row.get("qaf_decision_type") or "").strip() or "unknown"
        decision_counts[decision_type] = int(decision_counts.get(decision_type, 0)) + 1

    status_payload = {
        "success": True,
        "hito": 5,
        "release": release_cfg,
        "training": {
            "manifest": train_manifest,
            "alignment_report": alignment_report,
            "alignment_ok": bool(isinstance(alignment_report, dict) and alignment_report.get("pass") is True),
        },
        "runtime_24h": {
            "requests": total,
            "success": success,
            "error": max(0, total - success),
            "success_rate": round((float(success) / float(total)) if total else 0.0, 4),
            "canary_requests": canary_hits,
            "guardrail_hits": guardrail_hits,
            "decision_type_counts": decision_counts,
        },
        "notes": [
            "QAF engine -> policy -> LLM narrador activo cuando el release lo permite.",
            "Canary y rollback se controlan por variables de entorno HITO5_*.",
            "Trazabilidad por request en AuditLog accion llm.hito5.inference.",
        ],
    }

    return Response(status_payload)


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
    return Response(_attach_wow_event_payload(user, result, event_key='qaf_meal_coherence', label='Coherencia nutricional evaluada'))


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

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res.payload}, event_key='qaf_metabolic_profile', label='Perfil metabólico actualizado'))


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

    is_applied = False
    try:
        ws = getattr(user, 'coach_weekly_state', {}) or {}
        is_applied = str(ws.get('meal_plan_active_week_id') or '') == str(week_id)
    except Exception:
        is_applied = False
    quick_actions = build_quick_actions_for_menu(variety_level=variety, is_applied=bool(is_applied))
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

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': result, 'text': text, 'quick_actions': quick_actions}, event_key='qaf_meal_plan', label='Menú inteligente generado'))


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
    return Response(_attach_wow_event_payload(user, {'success': True, 'week_id': week_id}, event_key='qaf_meal_plan_apply', label='Plan semanal aplicado'))


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
    is_applied = False
    try:
        ws = getattr(user, 'coach_weekly_state', {}) or {}
        is_applied = str(ws.get('meal_plan_active_week_id') or '') == str(week_id)
    except Exception:
        is_applied = False
    quick_actions = build_quick_actions_for_menu(
        variety_level=str(((mutated.get('inputs') or {}).get('variety')) or 'normal'),
        is_applied=bool(is_applied),
    )
    return Response(_attach_wow_event_payload(user, {'success': True, 'result': mutated, 'text': text, 'quick_actions': quick_actions}, event_key='qaf_meal_plan_mutate', label='Plan semanal ajustado'))


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
    if not _is_premium_active(user):
        return Response(
            _qaf_premium_gate_payload(
                'exp-005_body_trend',
                'Tendencia corporal (6 semanas)',
                'Preview: te mostramos una lectura inicial y el tipo de escenario recomendado. Para simulaciones completas y ajustes por ingesta, activa Premium.',
                request=request,
                user=user,
                source='api_qaf_endpoint',
            ),
            status=200,
        )

    payload = request.data if isinstance(request.data, dict) else {}
    scenario = None
    try:
        scenario = str(payload.get('scenario') or '').strip().lower() or None
    except Exception:
        scenario = None
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
    text = render_professional_summary(res.payload, preferred_scenario=scenario)

    has_intake = (kcal_in_avg is not None)
    show_sim_actions = True
    if has_intake and scenario in ('follow_plan', 'minus_200', 'plus_200'):
        try:
            ws0 = getattr(user, 'coach_weekly_state', {}) or {}
            ws1 = dict(ws0)
            bt_sim = ws1.get('body_trend_sim_count')
            bt_sim = bt_sim if isinstance(bt_sim, dict) else {}
            bt_sim2 = dict(bt_sim)
            row = bt_sim2.get(week_id_now)
            row = row if isinstance(row, dict) else {}
            cnt = int(row.get('count') or 0) + 1
            bt_sim2[week_id_now] = {'count': int(cnt), 'updated_at': timezone.now().isoformat()}
            ws1['body_trend_sim_count'] = bt_sim2
            user.coach_weekly_state = ws1
            user.coach_weekly_updated_at = timezone.now()
            user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
            if cnt >= 2:
                show_sim_actions = False
        except Exception:
            pass

    quick_actions = []
    if (not has_intake) or show_sim_actions:
        quick_actions = build_quick_actions_for_trend(has_intake=bool(has_intake))

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

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res.payload, 'text': text, 'quick_actions': quick_actions}, event_key='qaf_body_trend', label='Tendencia corporal analizada'))


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
    if not _is_premium_active(user):
        return Response(
            _qaf_premium_gate_payload(
                'exp-006_posture',
                'Postura correctiva',
                'Preview: puedes recibir una guía general de captura y criterios básicos. El análisis postural completo y rutina correctiva detallada es Premium.',
                request=request,
                user=user,
                source='api_qaf_endpoint',
            ),
            status=200,
        )

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
    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res, 'text': text}, event_key='qaf_posture', label='Postura evaluada'))


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
    if not _is_premium_active(user):
        return Response(
            _qaf_premium_gate_payload(
                'exp-007_lifestyle',
                'Estado de hoy (Lifestyle)',
                'Preview: vista rápida del estado general. Para lectura DHSS completa, patrones y micro-hábitos personalizados, activa Premium.',
                request=request,
                user=user,
                source='api_qaf_endpoint',
            ),
            status=200,
        )

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

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res, 'text': text, 'daily_metrics': daily_metrics[-7:]}, event_key='qaf_lifestyle', label='Lifestyle actualizado'))


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
    if not _is_premium_active(user):
        return Response(
            _qaf_premium_gate_payload(
                'exp-008_motivation',
                'Motivación psicológica',
                'Preview: mensaje breve de impulso para hoy. El perfil motivacional adaptativo y retos personalizados en continuidad requieren Premium.',
                request=request,
                user=user,
                source='api_qaf_endpoint',
            ),
            status=200,
        )

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

    # UX: nombre amigable para saludo
    try:
        display = (getattr(user, 'full_name', '') or '').strip() or (getattr(user, 'username', '') or '').strip()
        if display and isinstance(res, dict):
            res = {**res, 'user_display_name': display}
    except Exception:
        pass
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

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res, 'text': text}, event_key='qaf_motivation', label='Motivación personalizada'))


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
    if not _is_premium_active(user):
        return Response(
            _qaf_premium_gate_payload(
                'exp-009_progression',
                'Evolución de entrenamiento',
                'Preview: recomendación inicial del siguiente paso. La progresión inteligente completa con ajuste por historial y señales requiere Premium.',
                request=request,
                user=user,
                source='api_qaf_endpoint',
            ),
            status=200,
        )

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

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res, 'text': text}, event_key='qaf_progression', label='Progresión de entrenamiento'))


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_muscle_measure(request):
    """Exp-010: medición muscular (solo fotografía) usando keypoints 2D.

    Importante:
    - Este endpoint NO hace pose estimation.
    - Recibe `poses[view_id].keypoints` ya calculados en cliente.
    - Acepta 1..4 vistas (opcionales).
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    locale = (payload.get('locale') or '').strip() or 'es-CO'

    week_id_now = _week_id()
    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

    baseline = None
    try:
        mm = weekly_state.get('muscle_measure') if isinstance(weekly_state.get('muscle_measure'), dict) else {}
        prev_keys = [k for k in mm.keys() if isinstance(k, str) and k != week_id_now]
        if prev_keys:
            prev_key = sorted(prev_keys)[-1]
            prev_row = mm.get(prev_key)
            if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                baseline = prev_row.get('result')
    except Exception:
        baseline = None

    from .qaf_muscle_measure.engine import evaluate_muscle_measure, render_professional_summary

    focus = payload.get('focus') if isinstance(payload.get('focus'), str) else None

    height_cm = None
    try:
        height_cm = float(getattr(user, 'height', None)) if getattr(user, 'height', None) is not None else None
    except Exception:
        height_cm = None

    res = evaluate_muscle_measure({
        'poses': payload.get('poses') if isinstance(payload.get('poses'), dict) else {},
        'baseline': baseline if isinstance(baseline, dict) else None,
        'focus': focus,
        'height_cm': height_cm,
        'locale': locale,
    }).payload

    # UX: nombre amigable para saludo
    try:
        display = (getattr(user, 'full_name', '') or '').strip() or (getattr(user, 'username', '') or '').strip()
        if display and isinstance(res, dict):
            res = {**res, 'user_display_name': display}
    except Exception:
        pass

    text = render_professional_summary(res)

    # Persistir (por semana)
    try:
        ws2 = dict(weekly_state)
        mm = ws2.get('muscle_measure') if isinstance(ws2.get('muscle_measure'), dict) else {}
        mm2 = dict(mm)
        mm2[week_id_now] = {
            'result': res,
            'updated_at': timezone.now().isoformat(),
            'week_id': week_id_now,
        }
        ws2['muscle_measure'] = mm2
        user.coach_weekly_state = ws2
        user.coach_weekly_updated_at = timezone.now()
        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
    except Exception:
        pass

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res, 'text': text}, event_key='qaf_muscle_measure', label='Medición muscular'))


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_shape_presence(request):
    """Exp-012: Shape & Presence Intelligence™ (MVP).

    Importante:
    - Este endpoint NO hace pose estimation.
    - Recibe `poses[view_id].keypoints` ya calculados en cliente.
    - Acepta 1..2 vistas (front_relaxed recomendado; side_right_relaxed opcional).
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}

    week_id_now = _week_id()
    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

    baseline = None
    try:
        sp = weekly_state.get('shape_presence') if isinstance(weekly_state.get('shape_presence'), dict) else {}
        prev_keys = [k for k in sp.keys() if isinstance(k, str) and k != week_id_now]
        if prev_keys:
            prev_key = sorted(prev_keys)[-1]
            prev_row = sp.get(prev_key)
            if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                baseline = prev_row.get('result')
    except Exception:
        baseline = None

    from .qaf_shape_presence.engine import evaluate_shape_presence, render_professional_summary

    res = evaluate_shape_presence(
        {
            'poses': payload.get('poses') if isinstance(payload.get('poses'), dict) else {},
            'baseline': baseline if isinstance(baseline, dict) else None,
            'locale': (payload.get('locale') or '').strip() or 'es-CO',
        }
    ).payload

    text = render_professional_summary(res)

    # Persistir (por semana)
    try:
        ws2 = dict(weekly_state)
        sp = ws2.get('shape_presence') if isinstance(ws2.get('shape_presence'), dict) else {}
        sp2 = dict(sp)
        sp2[week_id_now] = {
            'result': res,
            'updated_at': timezone.now().isoformat(),
            'week_id': week_id_now,
        }
        ws2['shape_presence'] = sp2
        user.coach_weekly_state = ws2
        user.coach_weekly_updated_at = timezone.now()
        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
    except Exception:
        pass

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res, 'text': text}, event_key='qaf_shape_presence', label='Shape & Presence actualizado'))


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_posture_proportion(request):
    """Exp-013: Arquitectura Corporal (antes Postura & Proporción) (unificado).

    Importante:
    - Este endpoint NO hace pose estimation.
    - Recibe `poses[view_id].keypoints` ya calculados en cliente.
    - Requiere 2 vistas: front_relaxed y side_right_relaxed.
    - Acepta back_relaxed como opcional.
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}

    week_id_now = _week_id()
    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}

    baseline = None
    try:
        pp = weekly_state.get('posture_proportion') if isinstance(weekly_state.get('posture_proportion'), dict) else {}
        prev_keys = [k for k in pp.keys() if isinstance(k, str) and k != week_id_now]
        if prev_keys:
            prev_key = sorted(prev_keys)[-1]
            prev_row = pp.get(prev_key)
            if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                baseline = prev_row.get('result')
    except Exception:
        baseline = None

    from .qaf_posture_proportion.engine import evaluate_posture_proportion, render_professional_summary

    res = evaluate_posture_proportion(
        {
            'poses': payload.get('poses') if isinstance(payload.get('poses'), dict) else {},
            'baseline': baseline if isinstance(baseline, dict) else None,
            'locale': (payload.get('locale') or '').strip() or 'es-CO',
        }
    ).payload

    text = render_professional_summary(res)

    # Persistir (por semana)
    try:
        ws2 = dict(weekly_state)
        pp = ws2.get('posture_proportion') if isinstance(ws2.get('posture_proportion'), dict) else {}
        pp2 = dict(pp)
        pp2[week_id_now] = {
            'result': res,
            'updated_at': timezone.now().isoformat(),
            'week_id': week_id_now,
        }
        ws2['posture_proportion'] = pp2
        user.coach_weekly_state = ws2
        user.coach_weekly_updated_at = timezone.now()
        user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
    except Exception:
        pass

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res, 'text': text}, event_key='qaf_posture_proportion', label='Arquitectura corporal evaluada'))


@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def qaf_skin_health(request):
    """Exp-011: Vitalidad de la Piel (energía visible + salud de piel).

    Importante:
    - No es diagnóstico médico.
    - Requiere al menos una imagen (attachment ya subida por el usuario).

    Entrada (JSON):
      {"attachment_url": "...", "context"?: {...}, "persist"?: bool}
    """

    if request.method == 'OPTIONS':
        return Response(status=status.HTTP_200_OK)

    user = _resolve_request_user(request)
    if not user:
        return Response({'error': 'Autenticacion requerida'}, status=status.HTTP_401_UNAUTHORIZED)

    payload = request.data if isinstance(request.data, dict) else {}
    attachment_url = str(payload.get('attachment_url') or '').strip()
    persist = str(payload.get('persist') or '').strip().lower() in ('1', 'true', 'yes', 'on')
    if payload.get('persist') is None:
        persist = True

    if not attachment_url:
        return Response({'success': False, 'error': 'attachment_url requerido'}, status=400)

    # Descargar bytes solo si el blob pertenece al usuario.
    image_bytes = None
    image_content_type = None
    try:
        normalized_url = _normalize_attachment_url(str(attachment_url))
        container_name, blob_name = _extract_blob_ref_from_url(normalized_url)
        if container_name == _chat_attachment_container() and blob_name:
            blob_name = _resolve_blob_name(container_name, blob_name) or blob_name
            safe_username = user.username.replace('/', '_')
            allowed = bool(blob_name.startswith(f"{safe_username}/")) or _is_signed_chat_attachment_url(normalized_url)
            if allowed:
                max_bytes = int(os.getenv('CHAT_VISION_MAX_BYTES', str(4 * 1024 * 1024)) or (4 * 1024 * 1024))
                resp = requests.get(normalized_url, timeout=20)
                if resp.status_code == 200 and resp.content and len(resp.content) <= max_bytes:
                    image_bytes = resp.content
                    image_content_type = (resp.headers.get('Content-Type') or 'image/jpeg')
    except Exception:
        image_bytes = None

    if not image_bytes:
        return Response({'success': False, 'error': 'No pude descargar la imagen (permiso/tamaño).'}, status=400)

    # Contexto opcional (hábitos). Preferimos payload.context, pero podemos enriquecer desde coach_state.
    ctx = payload.get('context') if isinstance(payload.get('context'), dict) else {}
    try:
        cs = getattr(user, 'coach_state', {}) or {}

        # Auto-reporte específico de Skin (si existe): prioriza sobre inferencias.
        try:
            sc = cs.get('skin_context') if isinstance(cs.get('skin_context'), dict) else {}
            if isinstance(sc, dict):
                for k in ('water_liters', 'stress_1_5', 'movement_1_5', 'sun_minutes', 'steps', 'sleep_minutes'):
                    if ctx.get(k) is None and sc.get(k) is not None:
                        ctx[k] = sc.get(k)
        except Exception:
            pass

        lifestyle_last = (cs.get('lifestyle_last') or {}).get('result') if isinstance(cs.get('lifestyle_last'), dict) else None
        if isinstance(lifestyle_last, dict):
            sig = lifestyle_last.get('signals') if isinstance(lifestyle_last.get('signals'), dict) else {}
            if ctx.get('sleep_minutes') is None:
                ctx['sleep_minutes'] = (sig.get('sleep') or {}).get('value')
            if ctx.get('steps') is None:
                ctx['steps'] = (sig.get('steps') or {}).get('value')
            if ctx.get('stress_1_5') is None:
                try:
                    stress_score01 = (sig.get('stress_inv') or {}).get('score01')
                    if stress_score01 is not None:
                        s = float(stress_score01)
                        s = max(0.0, min(1.0, s))
                        ctx['stress_1_5'] = round((1.0 - s) * 4.0 + 1.0, 1)
                except Exception:
                    pass
    except Exception:
        pass

    # baseline semanal
    week_id_now = _week_id()
    weekly_state = getattr(user, 'coach_weekly_state', {}) or {}
    baseline = None
    try:
        sh = weekly_state.get('skin_health') if isinstance(weekly_state.get('skin_health'), dict) else {}
        prev_keys = [k for k in sh.keys() if isinstance(k, str) and k != week_id_now]
        if prev_keys:
            prev_key = sorted(prev_keys)[-1]
            prev_row = sh.get(prev_key)
            if isinstance(prev_row, dict) and isinstance(prev_row.get('result'), dict):
                baseline = prev_row.get('result')
    except Exception:
        baseline = None

    from .qaf_skin_health.engine import evaluate_skin_health, render_professional_summary

    res = evaluate_skin_health(
        image_bytes=image_bytes,
        content_type=image_content_type,
        context=ctx,
        baseline=baseline if isinstance(baseline, dict) else None,
    ).payload

    try:
        if isinstance(res, dict):
            res = dict(res)
            res['user_display_name'] = (getattr(user, 'full_name', None) or getattr(user, 'username', '') or '').strip()
    except Exception:
        pass
    text = render_professional_summary(res)

    if persist:
        try:
            ws2 = dict(weekly_state)
            sh = ws2.get('skin_health') if isinstance(ws2.get('skin_health'), dict) else {}
            sh2 = dict(sh)
            sh2[week_id_now] = {
                'result': res,
                'updated_at': timezone.now().isoformat(),
                'week_id': week_id_now,
            }
            ws2['skin_health'] = sh2
            user.coach_weekly_state = ws2
            user.coach_weekly_updated_at = timezone.now()
            user.save(update_fields=['coach_weekly_state', 'coach_weekly_updated_at'])
        except Exception:
            pass

    qas = []
    try:
        if isinstance(res, dict) and str(res.get('decision') or '').strip().lower() != 'accepted':
            qas = [
                {'label': 'Tomar foto', 'type': 'open_camera'},
                {'label': 'Adjuntar foto', 'type': 'open_attach'},
            ]
    except Exception:
        qas = []

    return Response(_attach_wow_event_payload(user, {'success': True, 'result': res, 'text': text, 'quick_actions': qas}, event_key='qaf_skin_health', label='Vitalidad de la piel analizada'))


