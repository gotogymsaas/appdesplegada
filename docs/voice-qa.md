# QA Voz (Web + App)

## Web (PWA)
- Permiso microfono: permitir, negar, volver a intentar.
- Web Speech API: texto reconocido se envia a /api/chat/.
- Fallback STT: forzar fallo en Web Speech y validar /api/stt/.
- Estados UI: grabando, cancelar, reintentar.
- Mensajes: sin audio, sin permisos, transcripcion vacia.

## Android (Capacitor)
- Permiso microfono: permitir/denegar.
- SpeechRecognition nativo: texto reconocido se envia a /api/chat/.
- Cancelar y reintentar durante grabacion.
- Con y sin conexion.

## iOS (Capacitor)
- Permiso microfono y reconocimiento de voz.
- SpeechRecognition nativo: texto reconocido se envia a /api/chat/.
- Cancelar y reintentar.
- Con y sin conexion.

## Rendimiento
- Latencia total objetivo: <= 2.5s.
- Errores de red: mensaje claro y opcion de reintentar.

## Rollout
- Habilitar en 10% usuarios (si hay feature flag).
- Monitorear errores de STT y tiempos de respuesta.
- Escalar a 50% y 100% en 24-48h si estable.
