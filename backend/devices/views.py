from django.shortcuts import render
from django.utils import timezone
import requests
from django.conf import settings
import base64
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.authentication import JWTAuthentication


from .models import DeviceConnection, FitnessSync
from api.models import User
from .serializers import DeviceConnectionSerializer

PROVIDERS = [
    {"provider": "apple_health", "label": "Apple Health"},
    {"provider": "google_fit", "label": "Google Fit"},
    {"provider": "fitbit", "label": "Fitbit"},
    {"provider": "garmin", "label": "Garmin"},
]
PROVIDER_KEYS = {p["provider"] for p in PROVIDERS}

def _validate_provider(provider):
    if provider not in PROVIDER_KEYS:
        return False
    return True

def _clamp_score(value, min_val=0, max_val=10):
    try:
        return max(min_val, min(max_val, int(round(value))))
    except Exception:
        return min_val

def _score_from_steps(steps):
    # 0 steps -> 0, 12k steps -> 10
    return _clamp_score((steps / 12000.0) * 10.0)

def _score_from_sleep_minutes(minutes):
    # 0h -> 0, 8h -> 10
    hours = minutes / 60.0
    return _clamp_score((hours / 8.0) * 10.0)

def _refresh_google_fit_token(conn):
    if not conn.refresh_token:
        return False, {"error": "Falta refresh_token"}

    payload = {
        "client_id": settings.GOOGLE_FIT["WEB"]["CLIENT_ID"],
        "client_secret": settings.GOOGLE_FIT["WEB"]["CLIENT_SECRET"],
        "refresh_token": conn.refresh_token,
        "grant_type": "refresh_token",
    }

    r = requests.post(
        settings.GOOGLE_FIT["TOKEN_URL"],
        data=payload,
        timeout=15,
    )
    data = r.json()
    if r.status_code != 200:
        return False, {"error": "Refresh failed", "google": data}

    conn.access_token = data.get("access_token", conn.access_token)
    expires_in = int(data.get("expires_in", 0) or 0)
    if expires_in:
        conn.token_expires_at = timezone.now() + timezone.timedelta(seconds=expires_in)
    conn.save()
    return True, {"ok": True}

def _refresh_fitbit_token(conn):
    if not conn.refresh_token:
        return False, {"error": "Falta refresh_token"}

    client_id = settings.FITBIT["CLIENT_ID"]
    client_secret = settings.FITBIT["CLIENT_SECRET"]
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    headers = {
        "Authorization": f"Basic {basic}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": conn.refresh_token,
    }

    r = requests.post(settings.FITBIT["TOKEN_URL"], data=payload, headers=headers, timeout=15)
    data = r.json()
    if r.status_code != 200:
        return False, {"error": "Refresh failed", "fitbit": data}

    conn.access_token = data.get("access_token", conn.access_token)
    if data.get("refresh_token"):
        conn.refresh_token = data["refresh_token"]
    expires_in = int(data.get("expires_in", 0) or 0)
    if expires_in:
        conn.token_expires_at = timezone.now() + timezone.timedelta(seconds=expires_in)
    conn.save()
    return True, {"ok": True}

def _refresh_garmin_token(conn):
    if not conn.refresh_token:
        return False, {"error": "Falta refresh_token"}

    client_id = settings.GARMIN.get("CLIENT_ID")
    client_secret = settings.GARMIN.get("CLIENT_SECRET")
    token_url = settings.GARMIN.get("TOKEN_URL")
    if not client_id or not client_secret or not token_url:
        return False, {"error": "Configuración Garmin incompleta"}

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": conn.refresh_token,
    }

    r = requests.post(token_url, data=payload, timeout=15)
    data = r.json()
    if r.status_code != 200:
        return False, {"error": "Refresh failed", "garmin": data}

    conn.access_token = data.get("access_token", conn.access_token)
    if data.get("refresh_token"):
        conn.refresh_token = data["refresh_token"]
    expires_in = int(data.get("expires_in", 0) or 0)
    if expires_in:
        conn.token_expires_at = timezone.now() + timezone.timedelta(seconds=expires_in)
    conn.save()
    return True, {"ok": True}

@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def devices_list(request):
    # Solo considerar realmente conectados en la lista de "Conectados"
    qs = DeviceConnection.objects.filter(user=request.user, status="connected")
    return Response({
        "providers": PROVIDERS,
        "connected": DeviceConnectionSerializer(qs, many=True).data
    })

@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def device_connect(request, provider):

    if not _validate_provider(provider):
        return Response({"detail": "Proveedor inválido"}, status=400)

    obj, _ = DeviceConnection.objects.get_or_create(user=request.user, provider=provider)

    # TODO: aquí irá OAuth real. Por ahora devolvemos una URL simulada.
    obj.status = "pending"
    obj.save()

    return Response({
        "device": DeviceConnectionSerializer(obj).data,
        "redirect_url": f"/oauth/{provider}/authorize/"
    })


@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def device_disconnect(request, provider):
    obj, _ = DeviceConnection.objects.get_or_create(user=request.user, provider=provider)
    obj.status = "disconnected"
    obj.access_token = ""
    obj.refresh_token = ""
    obj.token_expires_at = None
    obj.save()
    return Response(DeviceConnectionSerializer(obj).data)

@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def device_sync(request, provider):
    user = request.user

    obj, _ = DeviceConnection.objects.get_or_create(user=request.user, provider=provider)

    if provider == "google_fit":
        if obj.status != "connected" or not obj.access_token:
            return Response(
                {"ok": False, "error": "Google Fit no conectado"},
                status=400,
            )

        if obj.token_expires_at and obj.token_expires_at <= timezone.now() + timezone.timedelta(seconds=30):
            ok, info = _refresh_google_fit_token(obj)
            if not ok:
                return Response({"ok": False, **info}, status=400)

        # Google Fit dataset aggregation for steps + calories + distance + sleep + heart rate
        now = timezone.now()
        start = obj.last_sync_at or (now - timezone.timedelta(hours=24))

        payload = {
            "aggregateBy": [
                {"dataTypeName": "com.google.step_count.delta"},
                {"dataTypeName": "com.google.calories.expended"},
                {"dataTypeName": "com.google.distance.delta"},
                {"dataTypeName": "com.google.sleep.segment"},
                {"dataTypeName": "com.google.heart_rate.bpm"},
            ],
            "bucketByTime": {"durationMillis": 86400000},
            "startTimeMillis": int(start.timestamp() * 1000),
            "endTimeMillis": int(now.timestamp() * 1000),
        }

        headers = {
            "Authorization": f"Bearer {obj.access_token}",
        }

        try:
            r = requests.post(
                "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate",
                json=payload,
                headers=headers,
                timeout=15,
            )
            if r.status_code != 200:
                return Response(
                    {"ok": False, "error": "Google Fit error", "google": r.json()},
                    status=400,
                )

            data = r.json()
            total_steps = 0
            total_calories = 0.0
            total_distance_m = 0.0
            sleep_duration_ms = 0
            hr_values = []

            for bucket in data.get("bucket", []):
                for dataset in bucket.get("dataset", []):
                    dtype = dataset.get("dataTypeName") or dataset.get("dataSourceId", "")
                    for point in dataset.get("point", []):
                        values = point.get("value") or []
                        if not values:
                            continue

                        v_int = values[0].get("intVal")
                        v_fp = values[0].get("fpVal")
                        v = v_int if v_int is not None else v_fp
                        if v is None:
                            v = 0

                        if "step_count" in dtype:
                            total_steps += int(v)
                        elif "calories" in dtype:
                            total_calories += float(v)
                        elif "distance" in dtype:
                            total_distance_m += float(v)
                        elif "heart_rate" in dtype:
                            hr_values.append(float(v))
                        elif "sleep" in dtype:
                            try:
                                start_ns = int(point.get("startTimeNanos", 0))
                                end_ns = int(point.get("endTimeNanos", 0))
                                if end_ns > start_ns:
                                    sleep_duration_ms += (end_ns - start_ns) / 1_000_000
                            except Exception:
                                pass

            avg_hr = round(sum(hr_values) / len(hr_values), 2) if hr_values else None
            metrics = {
                "steps": total_steps,
                "calories": round(total_calories, 2),
                "distance_m": round(total_distance_m, 2),
                "sleep_minutes": round(sleep_duration_ms / 60000.0, 2),
                "avg_heart_rate_bpm": avg_hr,
                "start_time": start.isoformat(),
                "end_time": now.isoformat(),
            }

            FitnessSync.objects.create(
                user=request.user,
                provider="google_fit",
                start_time=start,
                end_time=now,
                metrics=metrics,
                raw=data,
            )

            # Update user "concentrador" scores
            try:
                user = User.objects.get(id=request.user.id)
                current_scores = user.scores if user.scores else {}
                current_scores["s_steps"] = _score_from_steps(total_steps)
                current_scores["s_sleep"] = _score_from_sleep_minutes(metrics["sleep_minutes"])
                user.scores = current_scores
                user.save()
            except User.DoesNotExist:
                pass

            obj.last_sync_at = now
            obj.save()
            return Response(
                {
                    "ok": True,
                    "provider": "google_fit",
                    "metrics": metrics,
                    "scores": {
                        "s_steps": _score_from_steps(total_steps),
                        "s_sleep": _score_from_sleep_minutes(metrics["sleep_minutes"]),
                    },
                    "device": DeviceConnectionSerializer(obj).data,
                }
            )
        except requests.RequestException as exc:
            return Response(
                {"ok": False, "error": "Google Fit request failed", "detail": str(exc)},
                status=500,
            )

    if provider == "fitbit":
        if obj.status != "connected" or not obj.access_token:
            return Response(
                {"ok": False, "error": "Fitbit no conectado"},
                status=400,
            )

        if obj.token_expires_at and obj.token_expires_at <= timezone.now() + timezone.timedelta(seconds=30):
            ok, info = _refresh_fitbit_token(obj)
            if not ok:
                return Response({"ok": False, **info}, status=400)

        now = timezone.now()
        date_str = now.date().isoformat()

        headers = {
            "Authorization": f"Bearer {obj.access_token}",
        }

        def _get(url):
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                raise ValueError(r.json())
            return r.json()

        try:
            steps = 0
            calories = 0.0
            distance = 0.0
            sleep_minutes = 0
            avg_hr = None

            steps_data = _get(f"https://api.fitbit.com/1/user/-/activities/steps/date/{date_str}/1d.json")
            steps_arr = steps_data.get("activities-steps", [])
            if steps_arr:
                steps = int(float(steps_arr[-1].get("value", 0) or 0))

            cal_data = _get(f"https://api.fitbit.com/1/user/-/activities/calories/date/{date_str}/1d.json")
            cal_arr = cal_data.get("activities-calories", [])
            if cal_arr:
                calories = float(cal_arr[-1].get("value", 0) or 0)

            dist_data = _get(f"https://api.fitbit.com/1/user/-/activities/distance/date/{date_str}/1d.json")
            dist_arr = dist_data.get("activities-distance", [])
            if dist_arr:
                distance = float(dist_arr[-1].get("value", 0) or 0)

            sleep_data = _get(f"https://api.fitbit.com/1/user/-/sleep/date/{date_str}.json")
            sleep_summary = sleep_data.get("summary", {})
            sleep_minutes = int(sleep_summary.get("totalMinutesAsleep", 0) or 0)

            hr_data = _get(f"https://api.fitbit.com/1/user/-/activities/heart/date/{date_str}/1d.json")
            hr_arr = hr_data.get("activities-heart", [])
            if hr_arr:
                hr_val = hr_arr[-1].get("value", {})
                avg_hr = hr_val.get("restingHeartRate")

            metrics = {
                "steps": steps,
                "calories": round(calories, 2),
                "distance_km": round(distance, 3),
                "sleep_minutes": sleep_minutes,
                "resting_heart_rate_bpm": avg_hr,
                "date": date_str,
            }

            FitnessSync.objects.create(
                user=request.user,
                provider="fitbit",
                start_time=now,
                end_time=now,
                metrics=metrics,
                raw={
                    "steps": steps_data,
                    "calories": cal_data,
                    "distance": dist_data,
                    "sleep": sleep_data,
                    "heart": hr_data,
                },
            )

            try:
                user = User.objects.get(id=request.user.id)
                current_scores = user.scores if user.scores else {}
                current_scores["s_steps"] = _score_from_steps(steps)
                current_scores["s_sleep"] = _score_from_sleep_minutes(sleep_minutes)
                user.scores = current_scores
                user.save()
            except User.DoesNotExist:
                pass

            obj.last_sync_at = now
            obj.save()
            return Response(
                {
                    "ok": True,
                    "provider": "fitbit",
                    "metrics": metrics,
                    "scores": {
                        "s_steps": _score_from_steps(steps),
                        "s_sleep": _score_from_sleep_minutes(sleep_minutes),
                    },
                    "device": DeviceConnectionSerializer(obj).data,
                }
            )
        except ValueError as exc:
            return Response(
                {"ok": False, "error": "Fitbit error", "fitbit": exc.args[0]},
                status=400,
            )
        except requests.RequestException as exc:
            return Response(
                {"ok": False, "error": "Fitbit request failed", "detail": str(exc)},
                status=500,
            )

    if provider == "garmin":
        if obj.status != "connected" or not obj.access_token:
            return Response(
                {"ok": False, "error": "Garmin no conectado"},
                status=400,
            )

        if obj.token_expires_at and obj.token_expires_at <= timezone.now() + timezone.timedelta(seconds=30):
            ok, info = _refresh_garmin_token(obj)
            if not ok:
                return Response({"ok": False, **info}, status=400)

        endpoints = settings.GARMIN.get("ENDPOINTS", {})
        steps_url = endpoints.get("steps")
        sleep_url = endpoints.get("sleep")
        heart_url = endpoints.get("heart")

        if not steps_url and not sleep_url and not heart_url:
            return Response(
                {
                    "ok": False,
                    "error": "Garmin API no configurada (ENDPOINTS vacíos).",
                },
                status=400,
            )

        headers = {"Authorization": f"Bearer {obj.access_token}"}

        def _get(url):
            if not url:
                return None
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                raise ValueError(r.json())
            return r.json()

        try:
            steps_data = _get(steps_url)
            sleep_data = _get(sleep_url)
            heart_data = _get(heart_url)

            metrics = {
                "steps": steps_data,
                "sleep": sleep_data,
                "heart": heart_data,
            }

            FitnessSync.objects.create(
                user=request.user,
                provider="garmin",
                start_time=timezone.now(),
                end_time=timezone.now(),
                metrics=metrics,
                raw=metrics,
            )

            obj.last_sync_at = timezone.now()
            obj.save()
            return Response(
                {
                    "ok": True,
                    "provider": "garmin",
                    "metrics": metrics,
                    "device": DeviceConnectionSerializer(obj).data,
                }
            )
        except ValueError as exc:
            return Response(
                {"ok": False, "error": "Garmin error", "garmin": exc.args[0]},
                status=400,
            )
        except requests.RequestException as exc:
            return Response(
                {"ok": False, "error": "Garmin request failed", "detail": str(exc)},
                status=500,
            )

    # Default: simulated sync for other providers
    obj.last_sync_at = timezone.now()
    obj.save()
    return Response({"ok": True, "device": DeviceConnectionSerializer(obj).data})
