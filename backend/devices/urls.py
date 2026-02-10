from django.urls import path
from . import views
from devices.oauth.oauth_views import oauth_authorize, oauth_callback


urlpatterns = [
    path('devices/', views.devices_list),
    path('devices/<str:provider>/connect/', views.device_connect),
    path('devices/<str:provider>/disconnect/', views.device_disconnect),
    path('devices/<str:provider>/sync/', views.device_sync),

    # OAuth stub (REAL)
    path('oauth/<str:provider>/authorize/', oauth_authorize),
    path('oauth/<str:provider>/callback/', oauth_callback),
    

]

