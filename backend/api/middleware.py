import os


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
        response = self.get_response(request)
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        response['Access-Control-Allow-Methods'] = 'GET, POST, PUT, PATCH, DELETE, OPTIONS'
        return response