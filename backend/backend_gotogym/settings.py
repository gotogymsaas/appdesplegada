import os
from pathlib import Path
from urllib.parse import urlparse
from django.core.exceptions import ImproperlyConfigured
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent.parent

env_path = BASE_DIR / ".env"
if load_dotenv is not None:
    load_dotenv(env_path)
elif env_path.exists():
    # Fallback simple para cargar variables sin python-dotenv
    with env_path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_list(name, default=None):
    value = os.getenv(name, "")
    if not value:
        return default or []
    return [item.strip() for item in value.split(",") if item.strip()]


def _database_from_url(url):
    try:
        parsed = urlparse(url)
    except Exception:
        return None

    scheme = (parsed.scheme or "").lower()
    if scheme in ("postgres", "postgresql"):
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": (parsed.path or "").lstrip("/"),
            "USER": parsed.username or "",
            "PASSWORD": parsed.password or "",
            "HOST": parsed.hostname or "",
            "PORT": str(parsed.port or 5432),
            "CONN_MAX_AGE": int(os.getenv("DB_CONN_MAX_AGE", "60")),
        }
    return None


DEBUG = _env_bool("DEBUG", False)

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if DEBUG:
        SECRET_KEY = "django-insecure-dev-only"
    else:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY no está definido en el entorno.")

ALLOWED_HOSTS = _env_list("ALLOWED_HOSTS")
if DEBUG and not ALLOWED_HOSTS:
    ALLOWED_HOSTS = ["localhost", "127.0.0.1"]

# ======================
# APLICACIONES
# ======================
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Externas
    'rest_framework',
    'rest_framework.authtoken',
    'rest_framework_simplejwt',
    'corsheaders',

    # Storage
    'storages',

    # Tu app
    'api',
    'devices',
]

# ======================
# MIDDLEWARE - ORDEN CORREGIDO
# ======================
MIDDLEWARE = [
    'api.middleware.CorsMiddleware',  # PRIMERO - Middleware personalizado
    'corsheaders.middleware.CorsMiddleware',  # SEGUNDO - CorsHeaders
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'backend_gotogym.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'backend_gotogym.wsgi.application'

# ======================
# BASE DE DATOS
# ======================
_db = None
database_url = os.getenv("DATABASE_URL")
if database_url:
    _db = _database_from_url(database_url)
else:
    pg_name = os.getenv("POSTGRES_DB")
    pg_user = os.getenv("POSTGRES_USER")
    pg_pass = os.getenv("POSTGRES_PASSWORD")
    pg_host = os.getenv("POSTGRES_HOST")
    if pg_name and pg_user and pg_pass and pg_host:
        _db = {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": pg_name,
            "USER": pg_user,
            "PASSWORD": pg_pass,
            "HOST": pg_host,
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
            "CONN_MAX_AGE": int(os.getenv("DB_CONN_MAX_AGE", "60")),
        }

if not _db:
    _db = {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }

DATABASES = {"default": _db}

# ======================
# CONTRASEÑAS
# ======================
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# ======================
# INTERNACIONALIZACIÓN
# ======================
LANGUAGE_CODE = 'es-es'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# ======================
# CORS CONFIGURACIÓN COMPLETA
# ======================
CORS_ALLOW_ALL_ORIGINS = _env_bool("CORS_ALLOW_ALL_ORIGINS", False)
CORS_ALLOW_CREDENTIALS = True

# Lista explícita de orígenes permitidos
CORS_ALLOWED_ORIGINS = _env_list("CORS_ALLOWED_ORIGINS")
if DEBUG and not CORS_ALLOWED_ORIGINS:
    CORS_ALLOWED_ORIGINS = [
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://192.168.1.9:5500",
    ]

# Headers permitidos
CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# Métodos permitidos
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

# ======================
# SEGURIDAD / PRODUCCIÓN
# ======================
CSRF_TRUSTED_ORIGINS = _env_list("CSRF_TRUSTED_ORIGINS")

SECURE_SSL_REDIRECT = _env_bool("SECURE_SSL_REDIRECT", not DEBUG)
SESSION_COOKIE_SECURE = _env_bool("SESSION_COOKIE_SECURE", not DEBUG)
CSRF_COOKIE_SECURE = _env_bool("CSRF_COOKIE_SECURE", not DEBUG)

SECURE_HSTS_SECONDS = int(
    os.getenv("SECURE_HSTS_SECONDS", "0" if DEBUG else "31536000")
)
SECURE_HSTS_INCLUDE_SUBDOMAINS = _env_bool(
    "SECURE_HSTS_INCLUDE_SUBDOMAINS", not DEBUG
)
SECURE_HSTS_PRELOAD = _env_bool("SECURE_HSTS_PRELOAD", not DEBUG)

SECURE_CONTENT_TYPE_NOSNIFF = _env_bool("SECURE_CONTENT_TYPE_NOSNIFF", True)
SECURE_REFERRER_POLICY = os.getenv("SECURE_REFERRER_POLICY", "same-origin")
X_FRAME_OPTIONS = os.getenv("X_FRAME_OPTIONS", "DENY")

if _env_bool("SECURE_PROXY_SSL_HEADER", not DEBUG):
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "Lax")
CSRF_COOKIE_SAMESITE = os.getenv("CSRF_COOKIE_SAMESITE", "Lax")

# ======================
# MODELO DE USUARIO PERSONALIZADO
# ======================
AUTH_USER_MODEL = 'api.User'

# ======================
# REST FRAMEWORK (API JWT)
# ======================
REST_FRAMEWORK = {
  'DEFAULT_PERMISSION_CLASSES': [
     'rest_framework.permissions.AllowAny',
   ]
}
# ======================
# CONFIGURACIÓN DE AUTENTICACIÓN
# ======================
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
]

ENABLE_PASSWORD_VALIDATORS = _env_bool("ENABLE_PASSWORD_VALIDATORS", True)
if not ENABLE_PASSWORD_VALIDATORS:
    AUTH_PASSWORD_VALIDATORS = []

# MEDIA FILES
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip()
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "").strip()
AZURE_STORAGE_CONTAINER = os.getenv("AZURE_STORAGE_CONTAINER", "media").strip()
AZURE_STORAGE_CUSTOM_DOMAIN = os.getenv("AZURE_STORAGE_CUSTOM_DOMAIN", "").strip()

if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY:
    STORAGES = {
        "default": {
            "BACKEND": "backend_gotogym.storage.MediaAzureStorage",
        }
    }
    DEFAULT_FILE_STORAGE = "backend_gotogym.storage.MediaAzureStorage"
    AZURE_QUERYSTRING_AUTH = False
    if AZURE_STORAGE_CUSTOM_DOMAIN:
        MEDIA_URL = f"https://{AZURE_STORAGE_CUSTOM_DOMAIN}/{AZURE_STORAGE_CONTAINER}/"
    else:
        MEDIA_URL = f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/{AZURE_STORAGE_CONTAINER}/"


# ============================
# GOOGLE FIT OAUTH (CONSOLIDADO)
# ============================

GOOGLE_FIT = {
    "WEB": {
        "CLIENT_ID": os.getenv(
            "GF_WEB_CLIENT_ID",
            "16190961867-j27bvojfhann46tqf8p1ba75b5imfdgb.apps.googleusercontent.com"
        ),
        "CLIENT_SECRET": os.getenv(
            "GF_WEB_CLIENT_SECRET",
            "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"
        ),
        "REDIRECT_URI": os.getenv(
            "GF_WEB_REDIRECT_URI",
            "http://127.0.0.1:8000/oauth/google_fit/callback/"
        ),
    },
    "ANDROID": {
        "CLIENT_ID": os.getenv("GF_ANDROID_CLIENT_ID"),
    },
    "IOS": {
        "CLIENT_ID": os.getenv("GF_IOS_CLIENT_ID"),
    },
    "TOKEN_URL": "https://oauth2.googleapis.com/token",
    "AUTH_URL": "https://accounts.google.com/o/oauth2/v2/auth",
    "SCOPE": "https://www.googleapis.com/auth/fitness.activity.read https://www.googleapis.com/auth/fitness.sleep.read",
}

# ============================
# FITBIT OAUTH
# ============================
FITBIT = {
    "CLIENT_ID": os.getenv("FITBIT_CLIENT_ID", ""),
    "CLIENT_SECRET": os.getenv("FITBIT_CLIENT_SECRET", ""),
    "REDIRECT_URI": os.getenv(
        "FITBIT_REDIRECT_URI",
        "http://127.0.0.1:8000/oauth/fitbit/callback/",
    ),
    "SCOPE": os.getenv(
        "FITBIT_SCOPE",
        "activity sleep heartrate profile",
    ),
    "AUTH_URL": "https://www.fitbit.com/oauth2/authorize",
    "TOKEN_URL": "https://api.fitbit.com/oauth2/token",
}

# ============================
# GARMIN OAUTH (requiere Garmin Health API / partner)
# ============================
GARMIN = {
    "CLIENT_ID": os.getenv("GARMIN_CLIENT_ID", ""),
    "CLIENT_SECRET": os.getenv("GARMIN_CLIENT_SECRET", ""),
    "REDIRECT_URI": os.getenv(
        "GARMIN_REDIRECT_URI",
        "http://127.0.0.1:8000/oauth/garmin/callback/",
    ),
    "SCOPE": os.getenv("GARMIN_SCOPE", ""),
    "AUTH_URL": os.getenv("GARMIN_AUTH_URL", ""),
    "TOKEN_URL": os.getenv("GARMIN_TOKEN_URL", ""),
    "API_BASE": os.getenv("GARMIN_API_BASE", ""),
    "ENDPOINTS": {
        "steps": os.getenv("GARMIN_STEPS_URL", ""),
        "sleep": os.getenv("GARMIN_SLEEP_URL", ""),
        "heart": os.getenv("GARMIN_HEART_URL", ""),
    },
}

# ============================
# AZURE COMMUNICATION SERVICES (EMAIL)
# ============================
ACS_EMAIL_CONNECTION_STRING = os.getenv("ACS_EMAIL_CONNECTION_STRING", "").strip()
CONTACT_EMAIL_FROM = os.getenv("CONTACT_EMAIL_FROM", "DoNotReply@gotogym.store").strip()
CONTACT_EMAIL_TO = os.getenv("CONTACT_EMAIL_TO", "support@gotogym.store").strip()
CONTACT_EMAIL_SUBJECT_PREFIX = os.getenv("CONTACT_EMAIL_SUBJECT_PREFIX", "[GoToGym Contact]").strip()

# ============================
# PUSH NOTIFICATIONS (FCM)
# ============================
FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY", "").strip()
# ============================
# SMTP (Microsoft 365 u otro)
# ============================
_contact_provider = os.getenv("CONTACT_EMAIL_PROVIDER", "acs").strip().lower()
if _contact_provider not in ("acs", "smtp"):
    _contact_provider = "acs"
# Si hay connection string de ACS, priorizar ACS para evitar caidas a SMTP
if os.getenv("ACS_EMAIL_CONNECTION_STRING", "").strip():
    _contact_provider = "acs"
CONTACT_EMAIL_PROVIDER = _contact_provider
EMAIL_BACKEND = os.getenv("EMAIL_BACKEND", "django.core.mail.backends.smtp.EmailBackend")
EMAIL_HOST = os.getenv("EMAIL_HOST", "")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USE_TLS = os.getenv("EMAIL_USE_TLS", "true").lower() in ("1", "true", "yes")
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "")
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "")
DEFAULT_FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL", CONTACT_EMAIL_FROM)


