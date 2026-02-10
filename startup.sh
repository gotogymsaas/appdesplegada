#!/bin/sh
set -e

# App Service zip deploy path
if [ -d "/home/site/wwwroot/backend" ]; then
	cd /home/site/wwwroot/backend
elif [ -d "/app/backend" ]; then
	cd /app/backend
else
	echo "Backend directory not found."
	exit 1
fi

python manage.py migrate --noinput
python manage.py collectstatic --noinput || true

exec gunicorn --bind 0.0.0.0:8000 --workers 3 --timeout 120 backend_gotogym.wsgi:application
