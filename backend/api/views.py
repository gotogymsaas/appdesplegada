from rest_framework.decorators import api_view
from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework.response import Response
from rest_framework import status
import re

from devices.models import DeviceConnection, FitnessSync

import requests
import json
import traceback
import hashlib
import ipaddress
from django.contrib.auth import authenticate
from .models import User, HappinessRecord, IFQuestion, IFAnswer, UserDocument, ContactMessage, PushToken, TermsAcceptance
from .if_questions import IF_QUESTIONS
from .serializers import UserSerializer
# Importar el servicio ML
from .serializers import UserSerializer
# Importar el servicio ML
from datetime import date, datetime, timedelta
from django.utils import timezone
from django.db.models import Avg
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
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from django.conf import settings
from django.core.mail import EmailMessage
from django.utils.text import get_valid_filename
from zoneinfo import ZoneInfo

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
    from azure.storage.blob import generate_blob_sas, BlobSasPermissions
except Exception:
    BlobServiceClient = None
    ContentSettings = None
    generate_blob_sas = None
    BlobSasPermissions = None


class IsAuthenticatedOrOptions(IsAuthenticated):
    def has_permission(self, request, view):
        if request.method == "OPTIONS":
            return True
        return super().has_permission(request, view)

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


def _clean_extracted_text(text):
    if not text:
        return ""
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return cleaned.strip()


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
    recent_syncs = FitnessSync.objects.filter(user=user).order_by('-created_at')[:50]
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
            "age": user.age,
            "weight": user.weight,
            "height": user.height,
            "profession": getattr(user, "profession", None),
            "favorite_exercise_time": getattr(user, "favorite_exercise_time", None),
            "favorite_sport": getattr(user, "favorite_sport", None),
            "age_range": _range_bucket(user.age, 5),
            "weight_range": _range_bucket(user.weight, 5, lower=30, upper=200),
            "height_range": _range_bucket(user.height, 5, lower=120, upper=230),
            "happiness_index": user.happiness_index,
            "scores": user.scores or {},
            "current_streak": user.current_streak,
            "badges": user.badges,
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
        print("Intento de login:", data)

        username = data.get('username')
        password = data.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
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
        else:
            if not User.objects.filter(username=username).exists():
                error_msg = 'Usuario no encontrado'
            else:
                error_msg = 'Contrasena incorrecta'
                
            return Response({
                'success': False,
                'error': error_msg
            }, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        print('Error en login:', str(e))
        return Response({'error': 'Error interno del servidor'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

    def _send_via_smtp():
        if not settings.EMAIL_HOST or not settings.EMAIL_HOST_USER or not settings.EMAIL_HOST_PASSWORD:
            return Response({'error': 'SMTP no configurado'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        try:
            msg = EmailMessage(
                subject=subject_line,
                body=plain_text,
                from_email=settings.CONTACT_EMAIL_FROM,
                to=[recipient],
                reply_to=[email],
            )
            msg.content_subtype = "plain"
            msg.send(fail_silently=False)
            return Response({'success': True, 'message': 'Mensaje enviado'}, status=status.HTTP_200_OK)
        except Exception as e:
            print("Error SMTP:", str(e))
            return Response(
                {
                    'success': True,
                    'message': 'Mensaje recibido',
                },
                status=status.HTTP_200_OK
            )

    def _send_confirmation_via_smtp():
        msg = EmailMessage(
            subject=confirmation_subject,
            body=confirmation_plain,
            from_email=settings.CONTACT_EMAIL_FROM,
            to=[email],
        )
        msg.content_subtype = "plain"
        msg.send(fail_silently=False)

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

    if provider == "smtp":
        response = _send_via_smtp()
        if response.status_code == status.HTTP_200_OK:
            try:
                _send_confirmation_via_smtp()
            except Exception as confirm_err:
                print("Error enviando confirmacion:", str(confirm_err))
            _store_message()
        return response

    if not settings.ACS_EMAIL_CONNECTION_STRING or EmailClient is None:
        return Response(
            {'error': 'Servicio de correo no configurado'},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    try:
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
            client = EmailClient(endpoint, AzureKeyCredential(access_key))
        else:
            client = EmailClient.from_connection_string(conn)

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
            _store_message()
            return Response(
                {
                    'success': True,
                    'message': 'Mensaje recibido',
                },
                status=status.HTTP_200_OK
            )
        except Exception as send_err:
            print("Error enviando correo de soporte:", str(send_err))
            _store_message()
            return Response(
                {
                    'success': True,
                    'message': 'Mensaje recibido',
                },
                status=status.HTTP_200_OK
            )
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
        _store_message()
        return Response({'success': True, 'message': 'Mensaje enviado'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("Error enviando correo:", str(e))
        _store_message()
        return Response(
            {
                'success': True,
                'message': 'Mensaje recibido',
            },
            status=status.HTTP_200_OK
        )


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

    ok, info = _send_fcm(tokens, title, body, data={"source": "test"})
    if not ok:
        return Response({'success': False, 'error': str(info)}, status=status.HTTP_400_BAD_REQUEST)

    return Response({'success': True, 'result': info}, status=status.HTTP_200_OK)

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
        users = User.objects.all().order_by('-date_joined')
        serializer = UserSerializer(users, many=True, context={"request": request})
        return Response(serializer.data)
    except Exception as e:
        print("Error fetching users:", str(e))
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def delete_user(request, user_id):
    try:
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)
            
        if user.is_superuser:
            return Response({'error': 'No se puede eliminar un superusuario'}, status=status.HTTP_403_FORBIDDEN)
            
        user.delete()
        return Response({'success': True, 'message': 'Usuario eliminado correctamente'})
    except Exception as e:
        return Response({'error': 'Error al eliminar usuario'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def create_user_admin(request):
    try:
        data = request.data
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        plan = data.get('plan', 'Gratis')

        if not username or not email or not password:
            return Response({'error': 'Todos los campos son obligatorios'}, status=status.HTTP_400_BAD_REQUEST)

        if User.objects.filter(username=username).exists():
            return Response({'error': 'El usuario ya existe'}, status=status.HTTP_400_BAD_REQUEST)

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password,
            plan=plan
        )
        
        serializer = UserSerializer(user)
        return Response({
            'success': True, 
            'message': 'Usuario creado exitosamente',
            'user': serializer.data
        }, status=status.HTTP_201_CREATED)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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

    total = IFQuestion.objects.filter(active=True).count()
    answered = IFAnswer.objects.filter(user=user, week_id=week_id).values('question').distinct().count()

    payload = {
        'success': True,
        'week_id': week_id,
        'progress': {'answered': answered, 'total': total},
        'message': 'Respuesta guardada'
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
                user.height = float(request.data['height']) if str(request.data['height']).strip() != '' else None
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

def update_user_streak(user):
    today = date.today()
    if user.last_streak_date == today:
        return # Already updated today
    
    if user.last_streak_date == today - timedelta(days=1):
        user.current_streak += 1
    else:
        user.current_streak = 1
    
    user.last_streak_date = today
    user.save()
    check_badges(user)

def check_badges(user):
    badges = set(user.badges) # Use set to avoid duplicates
    
    # Badge: Iniciado (First step)
    badges.add('iniciado')
    
    # Badge: On Fire (Streak >= 3)
    if user.current_streak >= 3:
        badges.add('on_fire')
        
    # Badge: Zen Master (Happiness >= 8.0)
    if user.happiness_index and user.happiness_index >= 8.0:
        badges.add('zen_master')
        
    # Save if changed
    new_badges = list(badges)
    if len(new_badges) != len(user.badges):
        user.badges = new_badges
    new_badges = list(badges)
    if len(new_badges) != len(user.badges):
        user.badges = new_badges
        user.save()

@api_view(['PUT', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def update_user_admin(request, user_id):
    try:
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({'error': 'Usuario no encontrado'}, status=status.HTTP_404_NOT_FOUND)

        data = request.data
        if 'username' in data: user.username = data['username']
        if 'email' in data: user.email = data['email']
        if 'plan' in data: user.plan = data['plan']
        
        user.save()
        return Response({'success': True, 'message': 'Usuario actualizado'})
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET', 'OPTIONS'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticatedOrOptions])
def get_global_history(request):
    try:
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

@api_view(['POST'])
def upload_medical_record(request):
    try:
        username = (request.data.get('username') or '').strip()
        file_obj = request.FILES.get('file')
        doc_type = (request.data.get('doc_type') or request.POST.get('doc_type') or '').strip()
        
        if not username or not file_obj:
            return Response({'error': 'Username and File required'}, status=400)

        if doc_type not in ('nutrition_plan', 'training_plan', 'medical_history'):
            return Response({'error': 'doc_type inválido'}, status=400)

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            return Response({'error': 'User not found'}, status=404)
            
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
        attachment_text = request.data.get('attachment_text') # Nuevo: Texto extraído

        if not message and not attachment_url:
            return Response({'error': 'Mensaje o adjunto vacío'}, status=400)

        # 2. Configuración de n8n
        n8n_url = getattr(settings, "N8N_WEBHOOK_URL", "").strip() or "http://172.200.202.47/webhook/general-agent-gotogym-v2"
        
        # Construir prompt enriquecido
        final_input = message or "Analisis de archivo adjunto"
        if attachment_text:
            final_input += f"\n\n--- DOCUMENTO ADJUNTO ---\n{attachment_text}\n-----------------------"

        professional_rule = (
            os.getenv("CHAT_PROFESSIONAL_RULE", "")
            .strip()
            or "Toma decisiones con el foco en ser lo mas profesional posible."
        )
        if professional_rule:
            final_input = f"INSTRUCCION DEL SISTEMA: {professional_rule}\n\n{final_input}"

        fitness_payload = None
        profile_payload = None
        if_snapshot = None
        integrations_payload = None
        documents_payload = None

        user = _resolve_request_user(request)
        if user:
            session_id = _safe_session_id(user)
        elif not session_id or session_id == 'invitado':
            session_id = f"guest_{uuid.uuid4().hex}"

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
                    FitnessSync.objects.filter(user=user)
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

        payload = {
            "chatInput": final_input,
            "system_rules": {
                "professional_focus": professional_rule,
            },
            "message": message,
            "sessionId": session_id,
            "attachment": attachment_url,
            "fitness": fitness_payload,
            "profile": profile_payload,
            "if_snapshot": if_snapshot,
            "integrations": integrations_payload,
            "documents": documents_payload,
        }

        # 3. Enviar a n8n
        # Timeout corto por si n8n tarda
        response = requests.post(n8n_url, json=payload, timeout=60)
        
        # 4. Procesar respuesta
        if response.status_code == 200:
            try:
                # Si n8n devuelve JSON, lo pasamos directo
                data = response.json()
            except:
                # Si devuelve texto plano, lo envolvemos
                data = {'output': response.text}
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
