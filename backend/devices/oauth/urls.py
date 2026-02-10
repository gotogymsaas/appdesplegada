from django.urls import path
from devices.oauth.oauth_views import oauth_authorize, oauth_callback
from devices.oauth.views_google_fit import google_fit_authorize, google_fit_callback
from devices.oauth.views_fitbit import fitbit_authorize, fitbit_callback
from devices.oauth.views_garmin import garmin_authorize, garmin_callback

urlpatterns = [
    # ✅ Google Fit REAL (redirige a Google)
    path('google_fit/authorize/', google_fit_authorize),
    path('google_fit/callback/', google_fit_callback),
    # ✅ Fitbit REAL
    path('fitbit/authorize/', fitbit_authorize),
    path('fitbit/callback/', fitbit_callback),
    # ✅ Garmin REAL (requiere credenciales de partner)
    path('garmin/authorize/', garmin_authorize),
    path('garmin/callback/', garmin_callback),

    # genéricos (otros providers)
    path("<str:provider>/authorize/", oauth_authorize),
    path("<str:provider>/callback/", oauth_callback),
]
