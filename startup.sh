#!/bin/sh
set -e

cd /app/backend

python manage.py migrate --noinput
python manage.py collectstatic --noinput || true

exec gunicorn --bind 0.0.0.0:8000 --workers 3 --timeout 120 backend_gotogym.wsgi:application
