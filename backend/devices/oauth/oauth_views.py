# backend/devices/oauth_views.py

from django.shortcuts import redirect
from django.utils import timezone
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication

from devices.models import DeviceConnection

from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def oauth_authorize(request, provider):
    """
    OAuth stub REAL.
    Simula autorización exitosa del proveedor.
    """

    obj, _ = DeviceConnection.objects.get_or_create(
        user=request.user,
        provider=provider
    )

    # Simulación de tokens OAuth
    obj.status = "connected"
    obj.access_token = f"stub-access-token-{provider}"
    obj.refresh_token = f"stub-refresh-token-{provider}"
    obj.token_expires_at = timezone.now() + timezone.timedelta(days=30)
    obj.last_sync_at = timezone.now()
    obj.save()

    # Redirige al frontend (ajusta si usas otra URL)
    return redirect("/devices.html?oauth=success")


@api_view(["GET"])
def oauth_callback(request, provider):
    return Response({
        "ok": True,
        "provider": provider,
        "message": "OAuth callback recibido (stub)",
    })

