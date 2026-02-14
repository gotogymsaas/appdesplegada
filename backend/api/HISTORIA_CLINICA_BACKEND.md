# Backend listo para Historia Clínica

Este documento describe lo que **ya existe** en backend para soportar Historia Clínica, aunque la pantalla frontend todavía esté en modo informativo.

## Estado actual
- El backend ya soporta `doc_type=medical_history` en el mismo flujo de documentos de salud.
- El almacenamiento está preparado para contenedor dedicado de Azure Blob.
- La extracción de texto (cuando aplica) ya está integrada y se guarda en BD.

## Modelo de datos
En [backend/api/models.py](backend/api/models.py#L87-L106):
- Entidad: `UserDocument`.
- Tipos soportados:
  - `nutrition_plan`
  - `training_plan`
  - `medical_history`
- Campos relevantes:
  - `file_name`
  - `file_url`
  - `extracted_text`
  - `updated_at`

## Endpoints disponibles
Definidos en [backend/api/urls.py](backend/api/urls.py#L21-L23):
- `POST /api/upload_medical/`
- `GET /api/user_documents/?username=<u>&doc_type=<t>`
- `POST /api/user_documents/delete/`

### 1) Subida de documento
Implementación: [backend/api/views.py](backend/api/views.py#L1624-L1732)
- Recibe `username`, `file`, `doc_type`.
- Valida `doc_type` permitido (`medical_history` incluido).
- Selecciona contenedor por tipo documental (mapa de contenedores).
- Sube el archivo a Azure Blob cuando hay credenciales configuradas.
- Mantiene fallback local si Blob no está habilitado.
- Extrae texto para PDF/imagen y guarda en `extracted_text`.
- Si reemplaza documento, elimina blob anterior.

Respuesta relevante:
- `file_url` (URL firmada para lectura)
- `file_url_raw` (URL base guardada)
- `doc_type`
- `extracted_text`

### 2) Consulta de documento
Implementación: [backend/api/views.py](backend/api/views.py#L1739-L1772)
- Retorna el documento más reciente por `doc_type`.
- Devuelve URL firmada (`file_url`) + URL base (`file_url_raw`).

### 3) Eliminación de documento
Implementación: [backend/api/views.py](backend/api/views.py#L1775-L1796)
- Elimina registro en BD.
- Intenta eliminar también el blob asociado.

## Contenedor de Historia Clínica
Configuración por variables de entorno en [env.example](env.example#L91-L94):
- `AZURE_CONTAINER_MEDICAL=historiaclinica`

Mapeo técnico en [backend/api/views.py](backend/api/views.py#L209-L214):
- `medical_history -> historiaclinica`

## Seguridad de acceso a archivo
En [backend/api/views.py](backend/api/views.py#L248-L273):
- Se genera SAS temporal para lectura (`file_url`).
- Tiempo de expiración configurable con `AZURE_SAS_EXPIRATION`.

## OCR / extracción de texto
En [backend/api/views.py](backend/api/views.py#L1665-L1695):
- PDF: extracción por `pypdf` + fallback OCR.
- Imágenes: OCR con `pytesseract`.
- El texto se persiste en `UserDocument.extracted_text`.

Dependencias Python declaradas en [requirements.txt](requirements.txt#L13-L19).

## Lo pendiente (cuando se trabaje la pantalla)
- Habilitar UI de carga/visualización/eliminación en pantalla de Historia Clínica.
- Mantener patrón visual actual del proyecto (mismo estilo de tarjetas/botones/toasts).
- Conectar la UI con:
  - `doc_type=medical_history`
  - `POST /api/upload_medical/`
  - `GET /api/user_documents/`
  - `POST /api/user_documents/delete/`

## Nota operativa
Para que OCR funcione plenamente en servidor, además de paquetes Python, se requieren binarios del sistema (`tesseract` y, para algunos PDFs escaneados, `poppler`).
