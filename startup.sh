#!/bin/sh
set -e

# App Service zip deploy path
if [ -d "/home/site/wwwroot/backend" ]; then
	APP_ROOT="/home/site/wwwroot"
	BACKEND_DIR="/home/site/wwwroot/backend"
elif [ -d "/app/backend" ]; then
	APP_ROOT="/app"
	BACKEND_DIR="/app/backend"
else
	echo "Backend directory not found."
	exit 1
fi

cd "$BACKEND_DIR"

INSTALL_OCR_SYSTEM_DEPS="${INSTALL_OCR_SYSTEM_DEPS:-${CHAT_ATTACHMENT_INSTALL_OCR_DEPS:-}}"
INSTALL_PDF_OCR_SYSTEM_DEPS="${INSTALL_PDF_OCR_SYSTEM_DEPS:-${CHAT_ATTACHMENT_INSTALL_PDF_OCR_DEPS:-}}"

if [ "$INSTALL_OCR_SYSTEM_DEPS" = "1" ] || [ "$INSTALL_OCR_SYSTEM_DEPS" = "true" ]; then
	if ! command -v tesseract >/dev/null 2>&1; then
		if command -v apt-get >/dev/null 2>&1; then
			echo "Installing OCR system dependency: tesseract."
			apt-get update \
				&& apt-get install -y --no-install-recommends tesseract-ocr \
				&& rm -rf /var/lib/apt/lists/*
		else
			echo "apt-get not available; tesseract not installed."
		fi
	fi

	if [ "$INSTALL_PDF_OCR_SYSTEM_DEPS" = "1" ] || [ "$INSTALL_PDF_OCR_SYSTEM_DEPS" = "true" ]; then
		if ! command -v pdftoppm >/dev/null 2>&1; then
			if command -v apt-get >/dev/null 2>&1; then
				echo "Installing PDF OCR system dependency: poppler-utils (pdftoppm)."
				apt-get update \
					&& apt-get install -y --no-install-recommends poppler-utils \
					&& rm -rf /var/lib/apt/lists/*
			else
				echo "apt-get not available; poppler-utils not installed."
			fi
		fi
	fi
else
	if ! command -v tesseract >/dev/null 2>&1; then
		echo "OCR system dependency not found (tesseract). Skipping install for faster startup. Set INSTALL_OCR_SYSTEM_DEPS=true to enable."
	fi
fi

# Prefer a persistent virtualenv under /home (App Service)
PYTHON_BIN="python"
VENV_DIR="${APP_ROOT}/antenv"
VENV_PY="${VENV_DIR}/bin/python"

if [ -x "$VENV_PY" ]; then
	PYTHON_BIN="$VENV_PY"
else
	echo "Creating persistent virtualenv at: $VENV_DIR"
	python -m venv "$VENV_DIR" || true
	if [ -x "$VENV_PY" ]; then
		PYTHON_BIN="$VENV_PY"
	fi
fi

# Install deps only if key runtime deps are missing
if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
required = ("django", "rest_framework")
missing = [m for m in required if importlib.util.find_spec(m) is None]
raise SystemExit(0 if not missing else 1)
PY
then
	REQ_FILE=""
	if [ -f "$BACKEND_DIR/requirements.runtime.txt" ]; then
		REQ_FILE="$BACKEND_DIR/requirements.runtime.txt"
	elif [ -f "$BACKEND_DIR/requirements.txt" ]; then
		REQ_FILE="$BACKEND_DIR/requirements.txt"
	elif [ -f "$APP_ROOT/requirements.txt" ]; then
		REQ_FILE="$APP_ROOT/requirements.txt"
	fi

	if [ -z "$REQ_FILE" ]; then
		echo "requirements.txt not found in expected paths."
		exit 1
	fi

	echo "Installing dependencies from: $REQ_FILE"
	"$PYTHON_BIN" -m pip install --no-cache-dir -r "$REQ_FILE"
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("storages") else 1)
PY
then
	echo "Installing missing runtime storage dependency: django-storages"
	"$PYTHON_BIN" -m pip install django-storages azure-storage-blob
fi

if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pywebpush") else 1)
PY
then
	echo "Installing missing runtime dependency: pywebpush"
	"$PYTHON_BIN" -m pip install pywebpush
fi

"$PYTHON_BIN" manage.py migrate --noinput
"$PYTHON_BIN" manage.py collectstatic --noinput || true

APP_PORT="${PORT:-${WEBSITES_PORT:-8000}}"
exec "$PYTHON_BIN" -m gunicorn --bind "0.0.0.0:${APP_PORT}" --workers 3 --timeout 120 backend_gotogym.wsgi:application
