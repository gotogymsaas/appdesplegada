from django.shortcuts import redirect
from django.http import HttpResponse
from django.utils import timezone

from devices.models import DeviceConnection


def oauth_authorize(request, provider):
    
    """
    Simulación de pantalla OAuth del proveedor.
    Aquí luego irá Google / Fitbit / Garmin real.
    """
    html = f"""
    <html>
      <head>
        <title>Autorizar {provider}</title>
        <style>
          body {{
            font-family: sans-serif;
            background: #111;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
          }}
          button {{
            font-size: 18px;
            padding: 12px 24px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
          }}
        </style>
      </head>
      <body>
        <form method="get" action="/oauth/{provider}/callback/">
          <button type="submit">Autorizar {provider}</button>
        </form>
      </body>
    </html>
    """
    return HttpResponse(html)


def oauth_callback(request, provider):
    """
    Simulación del callback OAuth exitoso.
    """
    if not request.user.is_authenticated:
        return HttpResponse("Usuario no autenticado", status=401)

    obj, _ = DeviceConnection.objects.get_or_create(
        user=request.user,
        provider=provider
    )

    # Stub de tokens (luego serán reales)
    obj.status = "connected"
    obj.access_token = "stub_access_token"
    obj.refresh_token = "stub_refresh_token"
    obj.token_expires_at = timezone.now() + timezone.timedelta(hours=1)
    obj.last_sync_at = timezone.now()
    obj.save()

    # Volvemos al frontend (pantalla Dispositivos)
    return redirect("http://127.0.0.1:5500/pages/devices/index.html")


