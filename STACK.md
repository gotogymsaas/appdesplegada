# GoToGym Stack

## Lenguajes

- Python
- JavaScript (vanilla)
- HTML/CSS

## Frameworks y librerías

- Backend.
- Django
- Django REST Framework
- SimpleJWT
- django-cors-headers
- ML/processing: numpy, pandas, scikit-learn, xgboost, joblib, pytesseract, pypdf, pdf2image, pillow, pillow-heif
- Email: azure-communication-email
- Env: python-dotenv

- Mobile/Web.
- Capacitor (`@capacitor/core`, `@capacitor/android`, `@capacitor/ios`)
- Push: `@capacitor/push-notifications`
- Integraciones salud: `capacitor-google-fit`, `capacitor-healthkit`

## Paquetes Node detectados (package.json)

- name: `gotogym_project_janathan`
- version: `1.0.0`
- dependencies: `@capacitor/android`, `@capacitor/cli`, `@capacitor/core`, `@capacitor/ios`, `@capacitor/push-notifications`, `capacitor-google-fit`, `capacitor-healthkit`.
- scripts: no hay `build` definido actualmente.

## Runtime detectado (local)

- Python 3.13.3 (según `backend/venv_gotogym/pyvenv.cfg`).
- Node.js v23.10.0 / npm 10.9.2 (según entorno local).

## Gestores de paquetes

- `pip` (Python)
- `npm` (Node)

## Base de datos

- Desarrollo: SQLite (`backend/db.sqlite3`)
- Producción: PostgreSQL (configurable por `DATABASE_URL` o `POSTGRES_*`)

## Servicios externos

- Azure Communication Services (Email)
- Firebase Cloud Messaging (Push)
- MercadoPago (suscripciones/webhook)
- Google Fit / HealthKit

## Checklist de preparación (pre-deploy)

1. Definir variables de entorno de producción (SECRET_KEY, DB, dominios, CORS).
2. Configurar base de datos PostgreSQL y migraciones.
3. Configurar almacenamiento de media (Blob Storage o similar).
4. Configurar HTTPS y dominio en el hosting.
5. Verificar que `contact/`, `login/` y `register/` funcionan con SSL.
6. Validar push tokens en Android/iOS (FCM/APNs).
7. Opcional: dockerizar backend y definir `Dockerfile`/`docker-compose` para entornos estables.
8. Ejecutar pruebas básicas o smoke tests antes del despliegue.

## Comandos locales para verificar dependencias

- Node.
`node -v`
`npm -v`
`npm install`
`npm run build` (si agregas script de build)

- Python.
`python -V`
`python -m venv backend/venv_gotogym`
`backend/venv_gotogym/bin/pip install -r backend/requirements.txt`
`backend/venv_gotogym/bin/python backend/manage.py check`
`backend/venv_gotogym/bin/python backend/manage.py migrate`

- Capacitor.
`npx cap sync android`
`npx cap sync ios`
`android/app/google-services.json` debe existir para Android (Firebase).

- Backend local.
`backend/venv_gotogym/bin/python backend/manage.py runserver 0.0.0.0:8000`

- Frontend local.
`cd frontend && python -m http.server 5500 --bind 0.0.0.0`

## Preguntas abiertas

- Capacidad esperada (usuarios concurrentes, picos) para dimensionar recursos.
- Dominio final del API (para `ALLOWED_HOSTS` y `CORS_ALLOWED_ORIGINS`).
- Estrategia de almacenamiento de media (Azure Blob, S3, local).
