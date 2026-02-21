#!/bin/bash
set -e

# Usage:
#   export RESOURCE_GROUP=gotogym-prod-rg
#   export APP_NAME=gotogym-api
#   export DATABASE_URL='postgresql://user:pass@host:5432/gotogym_db'
#   export DJANGO_SECRET_KEY='replace-me'
#   export ALLOWED_HOSTS='gotogym-api.azurewebsites.net,api.gotogym.store'
#   export CORS_ALLOWED_ORIGINS='https://www.gotogym.store,https://gotogym.store,https://app.gotogym.store'
#   export ACS_EMAIL_CONNECTION_STRING='endpoint=...'
#   export CONTACT_EMAIL_FROM='DoNotReply@gotogym.store'
#   export CONTACT_EMAIL_TO='support@gotogym.store'
#   export FCM_SERVER_KEY='replace-me'
#   export MERCADOPAGO_ACCESS_TOKEN='replace-me'
#   export GF_WEB_CLIENT_ID='replace-me'
#   export GF_WEB_CLIENT_SECRET='replace-me'
#   export GF_WEB_REDIRECT_URI='https://api.gotogym.com/oauth/google_fit/callback/'

if [ -z "$RESOURCE_GROUP" ] || [ -z "$APP_NAME" ]; then
  echo "RESOURCE_GROUP and APP_NAME are required."
  exit 1
fi

az webapp config appsettings set \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_NAME" \
  --settings \
    "DEBUG=false" \
    "DJANGO_SECRET_KEY=${DJANGO_SECRET_KEY}" \
    "ALLOWED_HOSTS=${ALLOWED_HOSTS}" \
    "CORS_ALLOWED_ORIGINS=${CORS_ALLOWED_ORIGINS}" \
    "CSRF_TRUSTED_ORIGINS=${CSRF_TRUSTED_ORIGINS}" \
    "DATABASE_URL=${DATABASE_URL}" \
    "SCM_DO_BUILD_DURING_DEPLOYMENT=true" \
    "WEBSITES_PORT=8000" \
    "CHAT_ATTACHMENT_OCR=true" \
    "CHAT_ATTACHMENT_INSTALL_OCR_DEPS=true" \
    "CHAT_ATTACHMENT_INSTALL_PDF_OCR_DEPS=true" \
    "INTERNAL_OCR_HEALTH_TOKEN=${INTERNAL_OCR_HEALTH_TOKEN}" \
    "ACS_EMAIL_CONNECTION_STRING=${ACS_EMAIL_CONNECTION_STRING}" \
    "CONTACT_EMAIL_FROM=${CONTACT_EMAIL_FROM}" \
    "CONTACT_EMAIL_TO=${CONTACT_EMAIL_TO}" \
    "FCM_SERVER_KEY=${FCM_SERVER_KEY}" \
    "MERCADOPAGO_ACCESS_TOKEN=${MERCADOPAGO_ACCESS_TOKEN}" \
    "GF_WEB_CLIENT_ID=${GF_WEB_CLIENT_ID}" \
    "GF_WEB_CLIENT_SECRET=${GF_WEB_CLIENT_SECRET}" \
    "GF_WEB_REDIRECT_URI=${GF_WEB_REDIRECT_URI}" \
    "FITBIT_CLIENT_ID=${FITBIT_CLIENT_ID}" \
    "FITBIT_CLIENT_SECRET=${FITBIT_CLIENT_SECRET}" \
    "FITBIT_REDIRECT_URI=${FITBIT_REDIRECT_URI}" \
    "GARMIN_CLIENT_ID=${GARMIN_CLIENT_ID}" \
    "GARMIN_CLIENT_SECRET=${GARMIN_CLIENT_SECRET}" \
    "GARMIN_REDIRECT_URI=${GARMIN_REDIRECT_URI}"

az webapp config set \
  --resource-group "$RESOURCE_GROUP" \
  --name "$APP_NAME" \
  --startup-file "bash /home/site/wwwroot/startup.sh"
