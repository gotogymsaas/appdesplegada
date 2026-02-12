#!/bin/sh
set -e

# App Service zip deploy path
if [ -d "/home/site/wwwroot/backend" ]; then
	APP_ROOT="/home/site/wwwroot"
	cd /home/site/wwwroot/backend
elif [ -d "/app/backend" ]; then
	APP_ROOT="/app"
	cd /app/backend
else
	echo "Backend directory not found."
	exit 1
fi

# Prefer App Service venv if present
PYTHON_BIN="python"
if [ -x "/home/site/wwwroot/antenv/bin/python" ]; then
	if /home/site/wwwroot/antenv/bin/python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("django") else 1)
PY
	then
		PYTHON_BIN="/home/site/wwwroot/antenv/bin/python"
	else
		echo "Detected broken antenv (django missing). Falling back to system python."
	fi
fi

# Install deps only if Django is missing
if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("django") else 1)
PY
then
	"$PYTHON_BIN" -m pip install -r "$APP_ROOT/requirements.txt"
fi

"$PYTHON_BIN" manage.py migrate --noinput
"$PYTHON_BIN" manage.py collectstatic --noinput || true

exec "$PYTHON_BIN" -m gunicorn --bind 0.0.0.0:8000 --workers 3 --timeout 120 backend_gotogym.wsgi:application
