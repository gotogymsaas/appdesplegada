from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from api.models import User

from .models import DeviceConnection, FitnessSync


@dataclass(frozen=True)
class SyncResult:
    ok: bool
    status: int
    payload: dict[str, Any]


def _is_premium_or_trial(user: User) -> bool:
    if getattr(user, "plan", "Gratis") == "Premium":
        return True
    if getattr(user, "trial_active", False) and getattr(user, "trial_ends_at", None):
        try:
            return timezone.now() < user.trial_ends_at
        except Exception:
            return False
    return False


def _clamp_score(value: Any, min_val: int = 0, max_val: int = 10) -> int:
    try:
        return max(min_val, min(max_val, int(round(float(value)))))
    except Exception:
        return min_val


def _score_from_steps(steps: Any) -> int:
    try:
        steps_num = float(steps or 0)
    except Exception:
        steps_num = 0
    return _clamp_score((steps_num / 12000.0) * 10.0)


def _score_from_sleep_minutes(minutes: Any) -> int:
    try:
        minutes_num = float(minutes or 0)
    except Exception:
        minutes_num = 0
    hours = minutes_num / 60.0
    return _clamp_score((hours / 8.0) * 10.0)


def _refresh_google_fit_token(conn: DeviceConnection) -> tuple[bool, dict[str, Any]]:
    if not conn.refresh_token:
        return False, {"error": "Falta refresh_token"}

    payload = {
        "client_id": settings.GOOGLE_FIT["WEB"]["CLIENT_ID"],
        "client_secret": settings.GOOGLE_FIT["WEB"]["CLIENT_SECRET"],
        "refresh_token": conn.refresh_token,
        "grant_type": "refresh_token",
    }

    r = requests.post(settings.GOOGLE_FIT["TOKEN_URL"], data=payload, timeout=15)
    data = r.json()
    if r.status_code != 200:
        return False, {"error": "Refresh failed", "google": data}

    conn.access_token = data.get("access_token", conn.access_token)
    expires_in = int(data.get("expires_in", 0) or 0)
    if expires_in:
        conn.token_expires_at = timezone.now() + timezone.timedelta(seconds=expires_in)
    conn.save(update_fields=["access_token", "token_expires_at", "updated_at"])
    return True, {"ok": True}


def _refresh_fitbit_token(conn: DeviceConnection) -> tuple[bool, dict[str, Any]]:
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
    conn.save(update_fields=["access_token", "refresh_token", "token_expires_at", "updated_at"])
    return True, {"ok": True}


def sync_device(user: User, provider: str) -> SyncResult:
    if not _is_premium_or_trial(user):
        return SyncResult(
            ok=False,
            status=402,
            payload={"ok": False, "error": "Requiere Premium para sincronizar"},
        )

    # Evitar concurrencia: el auto-sync (scheduler) y el botón manual pueden ejecutarse al mismo tiempo.
    # En Fitbit, el refresh_token puede rotar, causando fallos intermitentes si hay 2 refresh simultáneos.
    # Bloqueamos la fila por (user, provider) durante todo el proceso.
    with transaction.atomic():
        conn = (
            DeviceConnection.objects.select_for_update()
            .filter(user=user, provider=provider)
            .first()
        )
        if not conn:
            conn = DeviceConnection.objects.create(user=user, provider=provider)

    if provider == "google_fit":
        if conn.status != "connected" or not conn.access_token:
            return SyncResult(False, 400, {"ok": False, "error": "Google Fit no conectado"})

        if conn.token_expires_at and conn.token_expires_at <= timezone.now() + timezone.timedelta(seconds=30):
            ok, info = _refresh_google_fit_token(conn)
            if not ok:
                return SyncResult(False, 400, {"ok": False, **info})

        now = timezone.now()
        start = conn.last_sync_at or (now - timezone.timedelta(hours=24))

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

        headers = {"Authorization": f"Bearer {conn.access_token}"}

        try:
            r = requests.post(
                "https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate",
                json=payload,
                headers=headers,
                timeout=15,
            )
            if r.status_code != 200:
                return SyncResult(
                    False,
                    400,
                    {"ok": False, "error": "Google Fit error", "google": r.json()},
                )

            data = r.json()
            total_steps = 0
            total_calories = 0.0
            total_distance_m = 0.0
            sleep_duration_ms = 0
            hr_values: list[float] = []

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
                user=user,
                provider="google_fit",
                start_time=start,
                end_time=now,
                metrics=metrics,
                raw=data,
            )

            try:
                db_user = User.objects.get(id=user.id)
                current_scores = db_user.scores if db_user.scores else {}
                current_scores["s_steps"] = _score_from_steps(total_steps)
                current_scores["s_sleep"] = _score_from_sleep_minutes(metrics["sleep_minutes"])
                db_user.scores = current_scores
                db_user.save(update_fields=["scores"])
            except User.DoesNotExist:
                pass

            conn.last_sync_at = now
            conn.save(update_fields=["last_sync_at", "updated_at"])

            return SyncResult(
                True,
                200,
                {
                    "ok": True,
                    "provider": "google_fit",
                    "metrics": metrics,
                    "scores": {
                        "s_steps": _score_from_steps(total_steps),
                        "s_sleep": _score_from_sleep_minutes(metrics["sleep_minutes"]),
                    },
                    "device": {
                        "provider": conn.provider,
                        "status": conn.status,
                        "last_sync_at": conn.last_sync_at.isoformat() if conn.last_sync_at else None,
                        "updated_at": conn.updated_at.isoformat() if conn.updated_at else None,
                    },
                },
            )
        except requests.RequestException as exc:
            return SyncResult(False, 500, {"ok": False, "error": "Google Fit request failed", "detail": str(exc)})

    if provider == "fitbit":
        if conn.status != "connected" or not conn.access_token:
            return SyncResult(False, 400, {"ok": False, "error": "Fitbit no conectado"})

        if conn.token_expires_at and conn.token_expires_at <= timezone.now() + timezone.timedelta(seconds=30):
            ok, info = _refresh_fitbit_token(conn)
            if not ok:
                return SyncResult(False, 400, {"ok": False, **info})

        now = timezone.now()
        tzinfo = timezone.get_current_timezone()
        now_local = timezone.localtime(now, tzinfo)
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        date_str = now_local.date().isoformat()
        tz_name = getattr(tzinfo, "key", str(tzinfo))

        headers = {"Authorization": f"Bearer {conn.access_token}"}

        def _get(url: str) -> dict[str, Any]:
            """GET con 1 reintento automático si Fitbit devuelve 401.

            Fitbit puede invalidar access tokens antes de token_expires_at (revocación, rotación, etc.).
            En ese caso, intentamos refresh y reintentamos una vez.
            """

            def _parse_body(resp: requests.Response):
                try:
                    return resp.json()
                except Exception:
                    return {"raw": (resp.text or "")[:500]}

            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                return _parse_body(r)

            # Rate limit
            if r.status_code == 429:
                raise ValueError({"http": 429, "error": "rate_limited", "fitbit": _parse_body(r)})

            # Token inválido/revocado: refresh y retry
            if r.status_code == 401 and conn.refresh_token:
                ok, info = _refresh_fitbit_token(conn)
                if ok:
                    headers["Authorization"] = f"Bearer {conn.access_token}"
                    r2 = requests.get(url, headers=headers, timeout=15)
                    if r2.status_code == 200:
                        return _parse_body(r2)
                    raise ValueError({"http": r2.status_code, "fitbit": _parse_body(r2)})
                raise ValueError({"http": 401, "error": "refresh_failed", **info})

            raise ValueError({"http": r.status_code, "fitbit": _parse_body(r)})

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

            missing_fields: list[str] = []
            if steps == 0:
                missing_fields.append("steps")
            if sleep_minutes == 0:
                missing_fields.append("sleep_minutes")
            is_partial = bool(missing_fields)

            metrics = {
                "steps": steps,
                "calories": round(calories, 2),
                "distance_km": round(distance, 3),
                "sleep_minutes": sleep_minutes,
                "resting_heart_rate_bpm": avg_hr,
                "date": date_str,
                "start_time": start_local.isoformat(),
                "end_time": now_local.isoformat(),
                "timezone": tz_name,
                "data_quality": "partial" if is_partial else "ok",
                "missing_fields": missing_fields,
            }

            FitnessSync.objects.create(
                user=user,
                provider="fitbit",
                start_time=start_local,
                end_time=now_local,
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
                db_user = User.objects.get(id=user.id)
                current_scores = db_user.scores if db_user.scores else {}
                current_scores["s_steps"] = _score_from_steps(steps)
                current_scores["s_sleep"] = _score_from_sleep_minutes(sleep_minutes)
                db_user.scores = current_scores
                db_user.save(update_fields=["scores"])
            except User.DoesNotExist:
                pass

            conn.last_sync_at = now
            conn.save(update_fields=["last_sync_at", "updated_at"])

            return SyncResult(
                True,
                200,
                {
                    "ok": True,
                    "provider": "fitbit",
                    "metrics": metrics,
                    "scores": {
                        "s_steps": _score_from_steps(steps),
                        "s_sleep": _score_from_sleep_minutes(sleep_minutes),
                    },
                    "device": {
                        "provider": conn.provider,
                        "status": conn.status,
                        "last_sync_at": conn.last_sync_at.isoformat() if conn.last_sync_at else None,
                        "updated_at": conn.updated_at.isoformat() if conn.updated_at else None,
                    },
                },
            )
        except ValueError as exc:
            details = exc.args[0] if exc.args else {}
            http = details.get("http") if isinstance(details, dict) else None
            if http == 429:
                return SyncResult(False, 429, {"ok": False, "error": "Fitbit rate limit. Intenta en 1-2 minutos.", "fitbit": details})
            # Mensaje más accionable si parece auth
            if http == 401:
                return SyncResult(False, 400, {"ok": False, "error": "Fitbit sesión expirada. Reconecta Fitbit e intenta de nuevo.", "fitbit": details})
            return SyncResult(False, 400, {"ok": False, "error": "Fitbit error", "fitbit": details})
        except requests.RequestException as exc:
            return SyncResult(False, 500, {"ok": False, "error": "Fitbit request failed", "detail": str(exc)})

    # Otros providers: por ahora sólo marca sync
    now = timezone.now()
    conn.last_sync_at = now
    conn.save(update_fields=["last_sync_at", "updated_at"])
    FitnessSync.objects.create(
        user=user,
        provider=provider,
        start_time=now,
        end_time=now,
        metrics={"note": "provider_not_implemented"},
        raw={},
    )
    return SyncResult(False, 200, {"ok": False, "provider": provider, "note": "provider_not_implemented"})
