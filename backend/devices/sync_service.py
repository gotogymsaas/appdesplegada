from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from api.models import User

from .models import DeviceConnection, FitnessSync, WhoopRawRecord


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


def _refresh_whoop_token(conn: DeviceConnection) -> tuple[bool, dict[str, Any]]:
    if not conn.refresh_token:
        return False, {"error": "Falta refresh_token"}

    cfg = getattr(settings, "WHOOP", {})
    client_id = (cfg.get("CLIENT_ID") or "").strip()
    client_secret = (cfg.get("CLIENT_SECRET") or "").strip()
    token_url = (cfg.get("TOKEN_URL") or "").strip()
    if not client_id or not client_secret or not token_url:
        return False, {"error": "Configuración WHOOP incompleta"}

    payload = {
        "grant_type": "refresh_token",
        "refresh_token": conn.refresh_token,
        "client_id": client_id,
        "client_secret": client_secret,
    }

    r = requests.post(token_url, data=payload, timeout=20)
    data = r.json() if r.content else {}
    if r.status_code != 200:
        return False, {"error": "Refresh failed", "whoop": data}

    conn.access_token = data.get("access_token", conn.access_token)
    if data.get("refresh_token"):
        conn.refresh_token = data["refresh_token"]
    expires_in = int(data.get("expires_in", 0) or 0)
    if expires_in:
        conn.token_expires_at = timezone.now() + timezone.timedelta(seconds=expires_in)
    conn.save(update_fields=["access_token", "refresh_token", "token_expires_at", "updated_at"])
    return True, {"ok": True}


def _whoop_api_base() -> str:
    base = getattr(settings, "WHOOP", {}).get("API_BASE") or "https://api.prod.whoop.com/developer"
    return base.rstrip("/")


def _whoop_get(url_path: str, access_token: str, *, params: dict[str, Any] | None = None) -> requests.Response:
    url = f"{_whoop_api_base()}{url_path}"
    headers = {"Authorization": f"Bearer {access_token}"}
    return requests.get(url, headers=headers, params=params or {}, timeout=25)


def _parse_dt(value: Any):
    if not value or not isinstance(value, str):
        return None
    try:
        # Django puede parsear ISO con 'Z' usando fromisoformat si se normaliza.
        cleaned = value.replace("Z", "+00:00")
        return timezone.datetime.fromisoformat(cleaned)
    except Exception:
        return None


def _whoop_upsert_records(user: User, resource_type: str, records: list[dict[str, Any]]) -> int:
    written = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue

        rid = rec.get("id")
        if rid is None and resource_type == "recovery":
            # Recovery no tiene id uuid; cycle_id es el identificador estable en el schema.
            rid = rec.get("cycle_id")
        if rid is None and resource_type in ("profile", "body"):
            rid = rec.get("user_id") or "singleton"
        if rid is None:
            continue

        defaults = {
            "payload": rec,
            "timezone_offset": (rec.get("timezone_offset") or "")[:8],
            "score_state": (rec.get("score_state") or "")[:20],
            "start_time": _parse_dt(rec.get("start")),
            "end_time": _parse_dt(rec.get("end")),
            "created_at_remote": _parse_dt(rec.get("created_at")),
            "updated_at_remote": _parse_dt(rec.get("updated_at")),
        }
        WhoopRawRecord.objects.update_or_create(
            user=user,
            resource_type=resource_type,
            resource_id=str(rid),
            defaults=defaults,
        )
        written += 1
    return written


def _whoop_fetch_collection(user: User, access_token: str, *, resource_type: str, path: str, start: str | None, end: str | None) -> dict[str, Any]:
    total = 0
    next_token: str | None = None
    pages = 0

    while True:
        params: dict[str, Any] = {"limit": 25}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        if next_token:
            params["nextToken"] = next_token

        resp = _whoop_get(path, access_token, params=params)
        if resp.status_code == 401:
            return {"ok": False, "error": "whoop_unauthorized", "http": 401}
        if resp.status_code == 429:
            return {"ok": False, "error": "whoop_rate_limited", "http": 429}
        if resp.status_code != 200:
            try:
                return {"ok": False, "error": "whoop_http_error", "http": resp.status_code, "whoop": resp.json()}
            except Exception:
                return {"ok": False, "error": "whoop_http_error", "http": resp.status_code, "whoop": (resp.text or "")[:500]}

        data = resp.json() if resp.content else {}
        records = data.get("records") or []
        if isinstance(records, list) and records:
            total += _whoop_upsert_records(user, resource_type, records)

        next_token = data.get("next_token") or data.get("nextToken")
        pages += 1
        if not next_token:
            break

        # corte de seguridad para evitar loops accidentales
        if pages >= 50:
            break

    return {"ok": True, "written": total, "pages": pages}


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

    if provider == "whoop":
        if conn.status != "connected" or not conn.access_token:
            return SyncResult(False, 400, {"ok": False, "error": "WHOOP no conectado"})

        # refresh si expira pronto
        if conn.token_expires_at and conn.token_expires_at <= timezone.now() + timezone.timedelta(seconds=30):
            ok, info = _refresh_whoop_token(conn)
            if not ok:
                return SyncResult(False, 400, {"ok": False, **info})

        # Ventana: incremental por last_sync_at; si no hay, último 30 días
        now = timezone.now()
        start_dt = conn.last_sync_at or (now - timezone.timedelta(days=30))
        start = start_dt.isoformat()
        end = now.isoformat()

        # 1) profile/basic y body measurements (no paginado, raw “singleton”)
        profile_resp = _whoop_get("/v2/user/profile/basic", conn.access_token)
        if profile_resp.status_code == 200:
            profile = profile_resp.json() if profile_resp.content else {}
            if isinstance(profile, dict) and profile:
                _whoop_upsert_records(user, "profile", [profile])
        elif profile_resp.status_code == 401:
            return SyncResult(False, 401, {"ok": False, "error": "whoop_unauthorized"})

        body_resp = _whoop_get("/v2/user/measurement/body", conn.access_token)
        if body_resp.status_code == 200:
            body = body_resp.json() if body_resp.content else {}
            if isinstance(body, dict) and body:
                _whoop_upsert_records(user, "body", [body])
        elif body_resp.status_code == 401:
            return SyncResult(False, 401, {"ok": False, "error": "whoop_unauthorized"})

        # 2) colecciones principales
        cycles = _whoop_fetch_collection(user, conn.access_token, resource_type="cycle", path="/v2/cycle", start=start, end=end)
        sleeps = _whoop_fetch_collection(user, conn.access_token, resource_type="sleep", path="/v2/activity/sleep", start=start, end=end)
        recoveries = _whoop_fetch_collection(user, conn.access_token, resource_type="recovery", path="/v2/recovery", start=start, end=end)
        workouts = _whoop_fetch_collection(user, conn.access_token, resource_type="workout", path="/v2/activity/workout", start=start, end=end)

        if not all(x.get("ok") for x in (cycles, sleeps, recoveries, workouts)):
            return SyncResult(
                False,
                400,
                {
                    "ok": False,
                    "error": "whoop_sync_failed",
                    "cycles": cycles,
                    "sleeps": sleeps,
                    "recoveries": recoveries,
                    "workouts": workouts,
                },
            )

        # registrar un resumen en FitnessSync (métricas normalizadas vendrán después)
        FitnessSync.objects.create(
            user=user,
            provider="whoop",
            start_time=start_dt,
            end_time=now,
            metrics={
                "cycles_written": cycles.get("written", 0),
                "sleeps_written": sleeps.get("written", 0),
                "recoveries_written": recoveries.get("written", 0),
                "workouts_written": workouts.get("written", 0),
            },
            raw={
                "cycles": cycles,
                "sleeps": sleeps,
                "recoveries": recoveries,
                "workouts": workouts,
            },
        )

        conn.last_sync_at = now
        conn.save(update_fields=["last_sync_at", "updated_at"])

        return SyncResult(
            True,
            200,
            {
                "ok": True,
                "provider": "whoop",
                "window": {"start": start, "end": end},
                "ingest": {
                    "cycles": cycles,
                    "sleeps": sleeps,
                    "recoveries": recoveries,
                    "workouts": workouts,
                },
                "device": {
                    "provider": conn.provider,
                    "status": conn.status,
                    "last_sync_at": conn.last_sync_at.isoformat() if conn.last_sync_at else None,
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
