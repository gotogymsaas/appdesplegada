from __future__ import annotations

from dataclasses import asdict
from typing import Any

from django.db import transaction
from django.utils import timezone
from zoneinfo import ZoneInfo

from api.models import User

from .models import DeviceConnection, SyncRequest, UserSyncCheckpoint
from .state_service import coach_message_for_now, compute_coach_state
from .sync_service import sync_device


SCHEDULE_KEYS = {
    "daily_reset": {"hour": 0, "minute": 5},
    "sleep_analysis": {"hour": 5, "minute": 30},
    "midday_review": {"hour": 12, "minute": 0},
    "evening_review": {"hour": 18, "minute": 0},
}


def _user_tz(user: User) -> ZoneInfo:
    name = (getattr(user, "timezone", "") or "").strip() or "UTC"
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


def _within_window(now_local: timezone.datetime, hour: int, minute: int, window_minutes: int) -> bool:
    start = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    end = start + timezone.timedelta(minutes=window_minutes)
    return start <= now_local < end


def _checkpoint_due(user: User, key: str, now_local: timezone.datetime) -> bool:
    cp = UserSyncCheckpoint.objects.filter(user=user, key=key).first()
    if not cp or not cp.last_run_at:
        return True
    last_local = timezone.localtime(cp.last_run_at, now_local.tzinfo)

    # daily tasks only once per local day
    return last_local.date() != now_local.date()


def _mark_checkpoint(user: User, key: str, now_utc: timezone.datetime) -> None:
    UserSyncCheckpoint.objects.update_or_create(
        user=user,
        key=key,
        defaults={"last_run_at": now_utc},
    )


def _connected_providers(user: User) -> list[str]:
    return list(
        DeviceConnection.objects.filter(user=user, status="connected")
        .values_list("provider", flat=True)
        .distinct()
    )


def _run_sync_for_user(user: User, providers: list[str], reason: str) -> dict[str, Any]:
    results: dict[str, Any] = {"reason": reason, "providers": {}}
    for provider in providers:
        res = sync_device(user, provider)
        results["providers"][provider] = res.payload

        # Capturar timezone desde métricas si viene
        try:
            tz_name = (
                res.payload.get("metrics", {}).get("timezone")
                if isinstance(res.payload, dict)
                else None
            )
            if tz_name and not (user.timezone or "").strip():
                user.timezone = str(tz_name)
                user.save(update_fields=["timezone"])
        except Exception:
            pass

    state = compute_coach_state(user)
    state_dict = {
        "morning_state": state.morning_state,
        "energy_mode": state.energy_mode,
        "rationale": state.rationale,
        "message": coach_message_for_now(state),
        "last_sync_at": state.last_sync_at,
        "metrics": state.metrics,
    }

    user.coach_state = state_dict
    user.coach_state_updated_at = timezone.now()
    user.save(update_fields=["coach_state", "coach_state_updated_at"])

    results["coach_state"] = state_dict
    return results


def enqueue_sync_request(user: User, provider: str = "", reason: str = "", priority: int = 5) -> None:
    SyncRequest.objects.create(
        user=user,
        provider=provider or "",
        reason=reason or "",
        priority=int(priority or 5),
        status="pending",
    )


def run_due_sync(*, window_minutes: int = 15, max_users: int = 50, max_requests: int = 50) -> dict[str, Any]:
    now_utc = timezone.now()

    processed_users: list[str] = []
    processed_requests: list[int] = []

    # 1) Procesar requests por evento (prioridad alta)
    pending = (
        SyncRequest.objects.filter(status="pending")
        .order_by("priority", "requested_at")
        .select_related("user")[:max_requests]
    )

    for req in pending:
        with transaction.atomic():
            locked = (
                SyncRequest.objects.select_for_update(skip_locked=True)
                .filter(id=req.id, status="pending")
                .first()
            )
            if not locked:
                continue
            locked.status = "running"
            locked.started_at = now_utc
            locked.save(update_fields=["status", "started_at"])

        user = locked.user
        providers = [locked.provider] if locked.provider else _connected_providers(user)
        if not providers:
            locked.status = "skipped"
            locked.finished_at = timezone.now()
            locked.result = {"ok": False, "error": "no_connected_providers"}
            locked.save(update_fields=["status", "finished_at", "result"])
            processed_requests.append(locked.id)
            continue

        try:
            result = _run_sync_for_user(user, providers, reason=f"event:{locked.reason or 'generic'}")
            locked.status = "done"
            locked.finished_at = timezone.now()
            locked.result = result
            locked.save(update_fields=["status", "finished_at", "result"])
            processed_requests.append(locked.id)
        except Exception as exc:
            locked.status = "error"
            locked.finished_at = timezone.now()
            locked.error = str(exc)
            locked.save(update_fields=["status", "finished_at", "error"])
            processed_requests.append(locked.id)

    # 2) Procesar tareas por horarios + invisible 3h
    users = (
        User.objects.all()
        .order_by("id")
        .only("id", "username", "plan", "trial_active", "trial_ends_at", "timezone", "coach_state")
    )

    count = 0
    for user in users:
        if count >= max_users:
            break

        providers = _connected_providers(user)
        if not providers:
            continue

        tz = _user_tz(user)
        now_local = timezone.localtime(now_utc, tz)

        ran_any = False
        for key, t in SCHEDULE_KEYS.items():
            if not _within_window(now_local, t["hour"], t["minute"], window_minutes):
                continue
            if not _checkpoint_due(user, key, now_local):
                continue

            _run_sync_for_user(user, providers, reason=f"scheduled:{key}")
            _mark_checkpoint(user, key, now_utc)
            ran_any = True

        # Ritual semanal (lunes 05:30 local)
        if now_local.weekday() == 0 and _within_window(now_local, 5, 30, window_minutes):
            key = "weekly_qaf"
            if _checkpoint_due(user, key, now_local):
                result = _run_sync_for_user(user, providers, reason="scheduled:weekly_qaf")
                # Weekly summary mínimo
                try:
                    user.coach_weekly_state = {
                        "week_id": now_local.strftime("%G-W%V"),
                        "generated_at": timezone.now().isoformat(),
                        "mode": user.coach_state.get("energy_mode") if isinstance(user.coach_state, dict) else "normal",
                        "note": "Ritual semanal generado (MVP).",
                    }
                    user.coach_weekly_updated_at = timezone.now()
                    user.save(update_fields=["coach_weekly_state", "coach_weekly_updated_at"])
                    result["coach_weekly_state"] = user.coach_weekly_state
                except Exception:
                    pass
                _mark_checkpoint(user, key, now_utc)
                ran_any = True

        # Invisible cada 3h (o cada 2h en modo protección)
        if not ran_any:
            key = "invisible_3h"
            cp = UserSyncCheckpoint.objects.filter(user=user, key=key).first()
            due = True
            interval_hours = 3
            try:
                if isinstance(user.coach_state, dict) and user.coach_state.get("energy_mode") == "proteccion":
                    interval_hours = 2
            except Exception:
                pass
            if cp and cp.last_run_at:
                due = now_utc - cp.last_run_at >= timezone.timedelta(hours=interval_hours)
            if due:
                _run_sync_for_user(user, providers, reason="scheduled:invisible_3h")
                _mark_checkpoint(user, key, now_utc)

        processed_users.append(user.username)
        count += 1

    return {
        "ok": True,
        "now": now_utc.isoformat(),
        "processed_users": processed_users,
        "processed_requests": processed_requests,
    }
