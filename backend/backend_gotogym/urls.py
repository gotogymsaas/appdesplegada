from django.contrib import admin
from django.urls import path, include, re_path
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/', include('devices.urls')),
    path('oauth/', include('devices.oauth.urls')),
]

from django.conf import settings
from django.views.static import serve

if settings.MEDIA_URL and settings.MEDIA_ROOT:
    media_prefix = settings.MEDIA_URL.lstrip('/')
    urlpatterns += [
        re_path(rf'^{media_prefix}(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    ]

