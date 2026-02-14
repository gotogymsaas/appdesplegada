# Programa de implementación: contenedores independientes de salud

## Objetivo
Separar almacenamiento documental por dominio sensible en Azure Blob:
- Nutrición: `nutricion`
- Entrenamiento: `entrenamiento`
- Historia clínica: `historiaclinica`

## Hito 1 — Gobierno y cumplimiento
- Confirmar clasificación de datos por `doc_type` con Legal.
- Definir retención y borrado por tipo documental.
- Definir política de acceso mínimo y auditoría.

**Estado:** Implementado en código el mapeo técnico por `doc_type`.

## Hito 2 — Infraestructura Blob segura
- Crear/validar contenedores separados.
- Configurar acceso privado por defecto y HTTPS obligatorio.
- Establecer caducidad de SAS corta para lectura.

**Estado:** Implementado en backend soporte de URL firmada por documento (`AZURE_SAS_EXPIRATION`).

## Hito 3 — Backend de documentos
- Subir archivo al contenedor según `doc_type`.
- Mantener extracción de texto (PDF + OCR) y persistencia en `UserDocument`.
- Eliminar blob previo al reemplazar/eliminar documento.

**Estado:** Implementado en `upload_medical`, `user_documents`, `user_documents_delete`.

## Hito 4 — UX in-app y continuidad visual
- Mantener estilos, fuentes y componentes existentes.
- Visualizar documento dentro de la app (sin pestaña externa).
- Mantener controles de eliminar/reemplazar.

**Estado:** Implementado para Nutrición y Entrenamiento con visor modal interno.

## Hito 5 — Validación y despliegue controlado
- Probar flujo completo por tipo: subir, ver, reemplazar, eliminar.
- Verificar OCR en entorno real (dependencias de sistema: tesseract/poppler).
- Activar monitoreo y checklist de rollback.

**Pendiente operativo:** ejecución de pruebas en entorno desplegado y validación de permisos de contenedores.
