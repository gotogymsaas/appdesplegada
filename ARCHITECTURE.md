# GoToGym Architecture

## High-Level Diagram

```mermaid
flowchart LR
  subgraph Clients
    Web[Web/PWA (frontend/)]
    Mobile[Mobile iOS/Android (Capacitor)]
  end

  Web -->|HTTPS| API[Django REST API]
  Mobile -->|HTTPS| API

  API -->|ORM| DB[(PostgreSQL in prod)
SQLite in dev]
  API --> Storage[(Media/Documents Storage)]
  API --> Auth[JWT Auth]

  API --> ACS[Azure Communication Services Email]
  API --> FCM[Firebase Cloud Messaging]
  API --> MP[MercadoPago Webhooks]
  API --> N8N[n8n Webhook]
  API --> Fit[Google Fit / HealthKit]
```

## Componentes

- Frontend web estático en `frontend/` con HTML/CSS/JS.
- App móvil Capacitor en `android/` y `ios/` usando el mismo `frontend/` como WebView.
- Backend Django en `backend/` con API REST (DRF) y JWT.
- Base de datos SQLite en desarrollo y PostgreSQL en producción.
- Almacenamiento de archivos (media/docs) que en producción debe ser Blob Storage o similar.
- Servicios externos.
- Email via Azure Communication Services.
- Push notifications via Firebase Cloud Messaging.
- Pagos y suscripciones via MercadoPago.
- Integraciones de bienestar (Google Fit / HealthKit).

## Notas de tráfico esperado

- Tráfico principal es lectura con picos en login, onboarding y consultas de perfil.
- Escrituras frecuentes en contacto, historial de felicidad, documentos, y tokens push.
- Webhooks de MercadoPago deben ser públicos, pero protegidos con validación de token.
- Chat y OCR pueden producir picos de CPU durante procesamiento de archivos.
- Estimación inicial: 1-2 vCPU y 2-4 GB RAM para la API en fase temprana, escalable según crecimiento.
