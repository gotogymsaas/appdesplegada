# Checklist de mejoras — Admin Dashboard (27-02-2026)

## 1) Menú hamburguesa consistente con Perfil
- [x] Replicar ubicación/jerarquía visual del menú de Perfil en páginas Admin.
- [x] Mantener navegación entre Dashboard, Usuarios, Notificaciones y Experiencias.
- [x] Mantener acción Cerrar Sesión dentro del menú.

## 2) Buenas prácticas de diseño en parte superior
- [x] Agregar bloque superior con principios de diseño operativo (claridad, consistencia, feedback, mínimo riesgo).
- [x] Reutilizar componentes/tokens existentes (`theme.css`, `adminDashboard.css`).

## 3) Ingresos conectados a Mercado Pago
- [x] Corregir cálculo MRR para contemplar suscripciones `authorized` y `active`.
- [x] Exponer metadata de fuente/estado en `overview` para trazabilidad.
- [x] Verificar render en dashboard con fallback claro si Mercado Pago falla.

## 4) Página de referencia de experiencias + endpoints
- [x] Crear página Admin de Experiencias con inventario de endpoints (13 experiencias).
- [x] Incluir tabla: código experiencia, nombre, endpoint principal, endpoint de soporte.
- [x] Agregar navegación desde menú hamburguesa.

## 5) Conexión del diagrama de 13 experiencias
- [x] Conectar gráfico de líneas con catálogo de experiencias/endpoints.
- [x] Asegurar correspondencia `exp-001` ... `exp-013` en frontend y backend.

## 6) Costos Azure (fase investigativa)
- [x] Identificar recursos Azure usados por la arquitectura actual.
- [x] Levantar método de estimación por servicio (App Service, DB, Storage, ACS, etc.).
- [~] Definir estructura de datos para panel de costos (pendiente de credenciales y consulta real de costos por recurso).

## 7) Validación
- [x] Verificar errores de archivos modificados.
- [ ] Verificar disponibilidad en la página desplegada de producción.
