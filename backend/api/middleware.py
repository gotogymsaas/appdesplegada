import os
import json
import logging
import time
import uuid

from django.http import HttpResponse


bench_logger = logging.getLogger("gtg.bench")


class AzureInternalHostMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Azure App Service startup probe can hit the app using a link-local IP
        # as Host (e.g. 169.254.x.x:8000). Django will raise DisallowedHost
        # unless it's in ALLOWED_HOSTS.
        #
        # We normalize this specific internal-probe host to WEBSITE_HOSTNAME.
        try:
            host = (request.META.get('HTTP_HOST') or '').strip()
            host_no_port = host.split(':', 1)[0]
            website_hostname = (os.getenv('WEBSITE_HOSTNAME', '') or '').strip()
            if website_hostname and host_no_port.startswith('169.254.'):
                request.META['HTTP_HOST'] = website_hostname
        except Exception:
            pass

        return self.get_response(request)


class CorsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Responder preflight sin depender de las vistas.
        if request.method == 'OPTIONS':
            response = HttpResponse(status=204)
        else:
            try:
                response = self.get_response(request)
            except Exception:
                # Importante: devolver un 500 con CORS para que el navegador no oculte el error.
                response = HttpResponse('Internal Server Error', status=500)

        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'

        # Si el browser pidió headers específicos (preflight), los reflejamos.
        requested_headers = request.headers.get('Access-Control-Request-Headers')
        response['Access-Control-Allow-Headers'] = requested_headers or 'Content-Type, Authorization'
        response['Vary'] = 'Origin'

        return response


class BenchmarkChatMiddleware:
    """Benchmark/telemetría mínima para construir comparativos % (costos/latencia/fallback).

    - Solo se activa para `POST /api/chat/`.
    - Emite 1 evento JSON por request (log line).
    - El view puede enriquecer el evento escribiendo en `request._bench_event`.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        try:
            req_path = str(getattr(request, 'path', '') or '')
            if request.method == 'POST' and (req_path == '/api/chat/' or req_path.startswith('/api/qaf/')):
                setattr(request, '_ops_started_perf', time.perf_counter())
        except Exception:
            pass

        # Feature-flag para no inundar logs en prod si no se requiere.
        enabled = (os.getenv("BENCHMARK_CHAT_LOGS", "") or "").strip().lower() in (
            "1",
            "true",
            "yes",
            "y",
            "on",
        )
        if not enabled:
            return self.get_response(request)

        # Solo /api/chat/ POST.
        try:
            if request.method != "POST":
                return self.get_response(request)
            if (request.path or "") != "/api/chat/":
                return self.get_response(request)
        except Exception:
            return self.get_response(request)

        start = time.perf_counter()

        # request_id estable para correlación
        request_id = (request.headers.get("X-Request-Id") or "").strip() or uuid.uuid4().hex
        setattr(request, "_bench_request_id", request_id)

        # Evento base; el view puede ampliar/ajustar.
        bench_event = {
            "event": "chat_benchmark",
            "request_id": request_id,
            "path": request.path,
            "method": request.method,
            "flow": "other",
            "qaf_fastpath": False,
            "n8n_called": False,
            "fallback_used": False,
            "tokens_in": None,
            "tokens_out": None,
            "latency_ms_total": None,
            "latency_ms_qaf": None,
            "latency_ms_n8n": 0,
            "attachment_bytes": None,
        }
        setattr(request, "_bench_event", bench_event)

        try:
            response = self.get_response(request)
        except Exception as exc:
            # Si algo rompe aquí, loguear igual.
            response = HttpResponse("Internal Server Error", status=500)
            try:
                bench_event["error"] = f"{type(exc).__name__}: {exc}"
            except Exception:
                pass

        end = time.perf_counter()
        total_ms = int(round((end - start) * 1000.0))
        bench_event["latency_ms_total"] = total_ms

        # Derivar latencia_qaf aproximada cuando solo sabemos total + n8n.
        try:
            n8n_ms = int(bench_event.get("latency_ms_n8n") or 0)
        except Exception:
            n8n_ms = 0
        bench_event["latency_ms_n8n"] = n8n_ms
        bench_event["latency_ms_qaf"] = max(0, total_ms - n8n_ms)

        # Si el view no marcó fastpath, inferirlo (éxito + sin n8n + flow conocido).
        try:
            status_code = int(getattr(response, "status_code", 0) or 0)
            bench_event["status_code"] = status_code
            if (not bench_event.get("qaf_fastpath")) and status_code and status_code < 400:
                flow = str(bench_event.get("flow") or "")
                if (not bench_event.get("n8n_called")) and flow.startswith("exp-"):
                    bench_event["qaf_fastpath"] = True
        except Exception:
            pass

        # Exponer request_id al cliente para correlación.
        try:
            response["X-Request-Id"] = request_id
        except Exception:
            pass

        # Emitir log JSON en 1 línea.
        try:
            bench_logger.info(json.dumps(bench_event, ensure_ascii=False, separators=(",", ":")))
        except Exception:
            # Nunca romper la request por logging.
            pass

        return response
