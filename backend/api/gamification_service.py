from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from django.db import transaction
from django.utils import timezone

from api.models import HappinessRecord, IFAnswer, User, UserDocument
from devices.models import FitnessSync


WEEKLY_TARGET_REGISTERS = 5
WOW_DAILY_REWARD_POINTS = 20
WOW_EVENT_REWARD_POINTS = 5


STREAK_TIERS: list[tuple[int, str]] = [
    (0, "Inicio"),
    (3, "Consistencia Activa"),
    (7, "Ritmo Estable"),
    (21, "Arquitecto de Hábito"),
    (90, "Maestría Biológica"),
]


IDENTITY_STATES: list[tuple[str, str]] = [
    ("explorador", "Explorador de Datos"),
    ("arquitecto", "Arquitecto de Hábitos"),
    ("consistente", "Consistente Estratégico"),
    ("biointeligente", "BioInteligente Activo"),
    ("maestro", "Maestro de Evolución"),
]


QAF_EXPERIENCES: list[dict[str, Any]] = [
    {"code": "exp-001_calories", "label": "Calorías Inteligentes", "event_keys": ["qaf_calories", "exp-001_calories"]},
    {"code": "exp-002_meal_coherence", "label": "Coherencia Nutricional", "event_keys": ["qaf_meal_coherence"]},
    {"code": "exp-003_metabolic_profile", "label": "Perfil Metabólico", "event_keys": ["qaf_metabolic_profile"]},
    {
        "code": "exp-004_meal_plan",
        "label": "Menú Semanal",
        "event_keys": ["qaf_meal_plan", "qaf_meal_plan_apply", "qaf_meal_plan_mutate"],
    },
    {"code": "exp-005_body_trend", "label": "Tendencia 6 Semanas", "event_keys": ["qaf_body_trend"]},
    {"code": "exp-006_posture", "label": "Corrección de Postura", "event_keys": ["qaf_posture"]},
    {"code": "exp-007_lifestyle", "label": "Estado de Hoy", "event_keys": ["qaf_lifestyle"]},
    {"code": "exp-008_motivation", "label": "Motivación", "event_keys": ["qaf_motivation"]},
    {"code": "exp-009_progression", "label": "Evolución de Entrenamiento", "event_keys": ["qaf_progression"]},
    {"code": "exp-010_muscle_measure", "label": "Progreso Muscular", "event_keys": ["qaf_muscle_measure"]},
    {"code": "exp-011_skin_health", "label": "Vitalidad de la Piel", "event_keys": ["qaf_skin_health"]},
    {"code": "exp-012_shape_presence", "label": "Alta Costura Inteligente", "event_keys": ["qaf_shape_presence"]},
    {"code": "exp-013_body_architecture", "label": "Arquitectura Corporal", "event_keys": ["qaf_posture_proportion"]},
]


def _today() -> date:
    return timezone.localdate()


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def _parse_iso_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except Exception:
            return None
    return None


def _week_start(d: date) -> date:
    # Monday as week start
    return d - timedelta(days=d.weekday())


def _week_end(d: date) -> date:
    # inclusive end Sunday
    return _week_start(d) + timedelta(days=6)


def _streak_tier(streak: int) -> dict[str, Any]:
    streak_num = int(streak or 0)
    chosen_threshold = 0
    chosen_label = STREAK_TIERS[0][1]
    for threshold, label in STREAK_TIERS:
        if streak_num >= threshold:
            chosen_threshold = threshold
            chosen_label = label
    next_tier = None
    for threshold, label in STREAK_TIERS:
        if threshold > chosen_threshold:
            next_tier = {"threshold": threshold, "label": label}
            break
    return {
        "current": {"threshold": chosen_threshold, "label": chosen_label},
        "next": next_tier,
    }


def _get_coach_state(user: User) -> dict[str, Any]:
    return user.coach_state if isinstance(getattr(user, "coach_state", None), dict) else {}


def _get_weekly_state(user: User) -> dict[str, Any]:
    return user.coach_weekly_state if isinstance(getattr(user, "coach_weekly_state", None), dict) else {}


def _set_user_json_state(user: User, *, coach_state: dict[str, Any] | None = None, weekly_state: dict[str, Any] | None = None) -> None:
    update_fields: list[str] = []
    if coach_state is not None:
        user.coach_state = coach_state
        update_fields.append("coach_state")
    if weekly_state is not None:
        user.coach_weekly_state = weekly_state
        update_fields.append("coach_weekly_state")
    if update_fields:
        user.save(update_fields=update_fields)


def check_badges(user: User) -> None:
    existing = user.badges or []
    badges = set(existing)

    badges.add("iniciado")

    if (user.current_streak or 0) >= 3:
        badges.add("on_fire")

    if user.happiness_index is not None and user.happiness_index >= 8.0:
        badges.add("zen_master")

    if set(existing) != badges:
        user.badges = sorted(badges)
        user.save(update_fields=["badges"])


def update_user_streak(user: User, *, activity_date: date | None = None, source: str | None = None) -> dict[str, Any]:
    """Update streak with 1-day grace.

    Rules:
    - One credit per day (if already credited today, noop)
    - Consecutive day increments
    - If user skipped exactly 1 day (gap of 2), allow a grace keep ONCE per rolling 7 days
    """

    d = activity_date or _today()
    last = user.last_streak_date
    if last == d:
        return {"changed": False, "date": _iso(d), "source": source or ""}

    coach_state = _get_coach_state(user)
    last_grace_used = _parse_iso_date(coach_state.get("streak_grace_used_on"))

    def grace_available() -> bool:
        if last_grace_used is None:
            return True
        # One grace per rolling 7 days
        return last_grace_used <= (d - timedelta(days=7))

    changed = False
    if last == (d - timedelta(days=1)):
        user.current_streak = int(user.current_streak or 0) + 1
        changed = True
    elif last == (d - timedelta(days=2)) and grace_available():
        user.current_streak = int(user.current_streak or 0) + 1
        coach_state["streak_grace_used_on"] = _iso(d)
        changed = True
    else:
        user.current_streak = 1
        changed = True

    user.last_streak_date = d

    # Persist streak + optional coach_state atomically
    with transaction.atomic():
        update_fields = ["current_streak", "last_streak_date"]
        if coach_state != _get_coach_state(user):
            update_fields.append("coach_state")
            user.coach_state = coach_state
        user.save(update_fields=update_fields)

    check_badges(user)
    return {"changed": changed, "date": _iso(d), "source": source or ""}


def _fitness_sync_is_real(metrics: Any) -> bool:
    if not isinstance(metrics, dict):
        return False
    if metrics.get("note") == "provider_not_implemented":
        return False
    # Prefer explicit quality if present
    if metrics.get("data_quality") == "ok":
        return True
    # Otherwise infer from any meaningful value
    keys = ("steps", "sleep_minutes", "calories", "distance_m", "distance_km")
    for k in keys:
        try:
            if float(metrics.get(k) or 0) > 0:
                return True
        except Exception:
            continue
    return False


def weekly_register_dates(user: User, *, week_of: date | None = None) -> dict[str, Any]:
    now_d = week_of or _today()
    start = _week_start(now_d)
    end = _week_end(now_d)

    if_dates = set(
        IFAnswer.objects.filter(user=user, answered_date__gte=start, answered_date__lte=end)
        .values_list("answered_date", flat=True)
    )

    # Dates from HappinessRecord (DateTimeField)
    start_dt_hr = timezone.make_aware(datetime.combine(start, time.min))
    end_dt_hr = timezone.make_aware(datetime.combine(end + timedelta(days=1), time.min))
    hr_dates = set(
        HappinessRecord.objects.filter(user=user, date__gte=start_dt_hr, date__lt=end_dt_hr).dates("date", "day")
    )

    # Dates from FitnessSync (created_at)
    start_dt = timezone.make_aware(datetime.combine(start, time.min))
    end_dt = timezone.make_aware(datetime.combine(end + timedelta(days=1), time.min))
    sync_dates: set[date] = set()
    for sync in FitnessSync.objects.filter(user=user, created_at__gte=start_dt, created_at__lt=end_dt).only("created_at", "metrics"):
        if _fitness_sync_is_real(getattr(sync, "metrics", None)):
            sync_dates.add(timezone.localdate(sync.created_at))

    all_dates = sorted(set(if_dates) | set(hr_dates) | sync_dates)
    return {
        "week_start": _iso(start),
        "week_end": _iso(end),
        "dates": [d.isoformat() for d in all_dates],
        "count": len(all_dates),
    }


def _compute_if_delta(user: User, *, start: date) -> float:
    # Baseline: latest record before week start
    start_dt = timezone.make_aware(datetime.combine(start, time.min))
    baseline = (
        HappinessRecord.objects.filter(user=user, date__lt=start_dt)
        .order_by("-date")
        .values_list("value", flat=True)
        .first()
    )
    current = (
        HappinessRecord.objects.filter(user=user)
        .order_by("-date")
        .values_list("value", flat=True)
        .first()
    )
    try:
        if baseline is None or current is None:
            return 0.0
        return float(current) - float(baseline)
    except Exception:
        return 0.0


def ensure_weekly_missions(user: User, *, week_of: date | None = None) -> dict[str, Any]:
    d = week_of or _today()
    start = _week_start(d)
    week_id = start.strftime("%G-W%V")
    weekly_state = _get_weekly_state(user)
    existing = weekly_state.get("missions") if isinstance(weekly_state.get("missions"), dict) else {}
    existing_week_id = existing.get("week_id") if isinstance(existing, dict) else None

    if existing_week_id != week_id:
        weekly_state["missions"] = {
            "week_id": week_id,
            "week_start": _iso(start),
            "created_at": timezone.now().isoformat(),
            "items": [
                {"key": "checkins", "label": "Completa 4 check-ins", "target": 4},
                {"key": "sync", "label": "Sincroniza 3 días", "target": 3},
                {"key": "if_delta", "label": "Mejora +0.5 tu IF", "target": 0.5},
            ],
        }
        _set_user_json_state(user, weekly_state=weekly_state)

    return weekly_state.get("missions") if isinstance(weekly_state.get("missions"), dict) else {}


def resolve_identity_state(user: User, *, weekly: dict[str, Any] | None = None) -> dict[str, Any]:
    streak = int(user.current_streak or 0)
    weekly_done = int((weekly or {}).get("weekly_progress", {}).get("done") or 0)
    sync_done = int((weekly or {}).get("missions_progress", {}).get("sync") or 0)

    # Deterministic rules (simple + explainable)
    key = "explorador"
    if streak >= 90 and weekly_done >= WEEKLY_TARGET_REGISTERS:
        key = "maestro"
    elif streak >= 21:
        key = "arquitecto"
    elif streak >= 7:
        key = "consistente"
    elif sync_done >= 3:
        key = "biointeligente"

    label = next((lbl for k, lbl in IDENTITY_STATES if k == key), IDENTITY_STATES[0][1])
    return {"key": key, "label": label}


def _resolve_qaf_experiences_progress(
    *,
    wow_events_by_day: dict[str, Any],
    week_start: date,
    week_end: date,
) -> dict[str, Any]:
    all_events: set[str] = set()
    week_events: set[str] = set()

    for day_key, events in wow_events_by_day.items():
        if not isinstance(events, list):
            continue
        d = _parse_iso_date(day_key)
        normalized = {str(x).strip().lower() for x in events if str(x).strip()}
        all_events |= normalized
        if d is not None and week_start <= d <= week_end:
            week_events |= normalized

    items: list[dict[str, Any]] = []
    completed_total = 0
    completed_week = 0

    for exp in QAF_EXPERIENCES:
        keys = [str(k).strip().lower() for k in (exp.get("event_keys") or []) if str(k).strip()]
        done_all = any(k in all_events for k in keys)
        done_week = any(k in week_events for k in keys)
        if done_all:
            completed_total += 1
        if done_week:
            completed_week += 1
        items.append({
            "code": exp.get("code"),
            "label": exp.get("label"),
            "completed": done_all,
            "completed_week": done_week,
        })

    return {
        "summary": {
            "completed": completed_total,
            "total": len(QAF_EXPERIENCES),
            "completed_week": completed_week,
        },
        "items": items,
    }


def build_gamification_status(user: User, *, as_of: date | None = None) -> dict[str, Any]:
    d = as_of or _today()

    registers = weekly_register_dates(user, week_of=d)
    weekly_progress = {
        "done": min(int(registers["count"]), WEEKLY_TARGET_REGISTERS),
        "target": WEEKLY_TARGET_REGISTERS,
    }

    missions_cfg = ensure_weekly_missions(user, week_of=d)

    start = _parse_iso_date(missions_cfg.get("week_start")) or _week_start(d)
    if_dates = set(
        IFAnswer.objects.filter(user=user, answered_date__gte=start, answered_date__lte=_week_end(start))
        .values_list("answered_date", flat=True)
    )

    # sync days for missions
    start_dt = timezone.make_aware(datetime.combine(start, time.min))
    end_dt = timezone.make_aware(datetime.combine(_week_end(start) + timedelta(days=1), time.min))
    sync_days: set[date] = set()
    for sync in FitnessSync.objects.filter(user=user, created_at__gte=start_dt, created_at__lt=end_dt).only("created_at", "metrics"):
        if _fitness_sync_is_real(getattr(sync, "metrics", None)):
            sync_days.add(timezone.localdate(sync.created_at))

    if_delta = _compute_if_delta(user, start=start)
    if_delta_progress = round(max(0.0, float(if_delta)), 2)

    missions_progress = {
        "checkins": len(if_dates),
        "sync": len(sync_days),
        "if_delta": if_delta_progress,
    }

    # Produce missions with progress/completion
    missions_items_out: list[dict[str, Any]] = []
    for item in (missions_cfg.get("items") or []):
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        target = item.get("target")
        progress = missions_progress.get(str(key), 0)
        completed = False
        try:
            if isinstance(target, (int, float)):
                completed = float(progress) >= float(target)
        except Exception:
            completed = False
        missions_items_out.append({
            "key": key,
            "label": item.get("label"),
            "target": target,
            "progress": progress,
            "completed": completed,
        })

    identity = resolve_identity_state(user, weekly={"weekly_progress": weekly_progress, "missions_progress": missions_progress})

    # Documentos clave (para activar recomendaciones más coherentes)
    docs_qs = UserDocument.objects.filter(user=user, doc_type__in=["nutrition_plan", "training_plan"]).only(
        "doc_type",
        "updated_at",
    )
    docs_map: dict[str, dict[str, Any]] = {
        "nutrition_plan": {"uploaded": False, "updated_at": None},
        "training_plan": {"uploaded": False, "updated_at": None},
    }
    for doc in docs_qs:
        try:
            docs_map[str(doc.doc_type)] = {
                "uploaded": True,
                "updated_at": doc.updated_at.isoformat() if getattr(doc, "updated_at", None) else None,
            }
        except Exception:
            continue

    # Persist identity in coach_state if changed
    coach_state = _get_coach_state(user)
    prev_key = coach_state.get("identity_state")
    identity_changed = prev_key != identity["key"]
    if identity_changed:
        coach_state["identity_state"] = identity["key"]
        coach_state["identity_changed_at"] = timezone.now().isoformat()
        _set_user_json_state(user, coach_state=coach_state)

    wow_state = coach_state.get("wow") if isinstance(coach_state.get("wow"), dict) else {}
    wow_events_by_day = wow_state.get("events_by_day") if isinstance(wow_state.get("events_by_day"), dict) else {}
    today_iso = _iso(d) or ""
    today_events = wow_events_by_day.get(today_iso) if today_iso else []
    if not isinstance(today_events, list):
        today_events = []
    points_total = int(wow_state.get("points_total") or 0)
    last_daily_claim_on = wow_state.get("last_daily_claim_on") if isinstance(wow_state.get("last_daily_claim_on"), str) else None
    claimed_today = bool(last_daily_claim_on and last_daily_claim_on == today_iso)
    week_start_d = _parse_iso_date(registers.get("week_start")) or _week_start(d)
    week_end_d = _parse_iso_date(registers.get("week_end")) or _week_end(d)
    qaf_experiences = _resolve_qaf_experiences_progress(
        wow_events_by_day=wow_events_by_day,
        week_start=week_start_d,
        week_end=week_end_d,
    )

    return {
        "as_of": _iso(d),
        "streak": int(user.current_streak or 0),
        "streak_tier": _streak_tier(int(user.current_streak or 0)),
        "weekly": {
            "week_start": registers["week_start"],
            "week_end": registers["week_end"],
            "register_dates": registers["dates"],
            "progress": weekly_progress,
        },
        "missions": {
            "week_id": missions_cfg.get("week_id"),
            "items": missions_items_out,
            "completed_all": all(bool(i.get("completed")) for i in missions_items_out) if missions_items_out else False,
        },
        "identity": {
            **identity,
            "changed": identity_changed,
        },
        "wow": {
            "points_total": points_total,
            "today_events_count": len(today_events),
            "daily_reward": {
                "amount": WOW_DAILY_REWARD_POINTS,
                "claimed_today": claimed_today,
                "last_claim_on": last_daily_claim_on,
            },
        },
        "experiences": qaf_experiences,
        "documents": docs_map,
    }


def _today_iso() -> str:
    return _today().isoformat()


def _get_wow_state(user: User) -> dict[str, Any]:
    coach_state = _get_coach_state(user)
    wow = coach_state.get("wow") if isinstance(coach_state.get("wow"), dict) else {}
    if not isinstance(wow.get("events_by_day"), dict):
        wow["events_by_day"] = {}
    return wow


def _set_wow_state(user: User, wow_state: dict[str, Any]) -> None:
    coach_state = _get_coach_state(user)
    coach_state["wow"] = wow_state
    _set_user_json_state(user, coach_state=coach_state)


def _prune_wow_events(events_by_day: dict[str, Any], *, keep_days: int = 400) -> dict[str, Any]:
    today_d = _today()
    pruned: dict[str, Any] = {}
    for k, v in events_by_day.items():
        d = _parse_iso_date(k)
        if d is None:
            continue
        if d < (today_d - timedelta(days=keep_days)):
            continue
        if isinstance(v, list):
            pruned[k] = [str(x) for x in v]
    return pruned


def claim_daily_wow_reward(user: User) -> dict[str, Any]:
    today_iso = _today_iso()
    wow = _get_wow_state(user)
    last_claim = wow.get("last_daily_claim_on")
    points_total = int(wow.get("points_total") or 0)

    claimed = str(last_claim or "") != today_iso
    delta = WOW_DAILY_REWARD_POINTS if claimed else 0

    if claimed:
        wow["last_daily_claim_on"] = today_iso
        wow["points_total"] = points_total + delta
        wow["daily_claims"] = int(wow.get("daily_claims") or 0) + 1
        _set_wow_state(user, wow)

    return {
        "ok": True,
        "claimed": claimed,
        "reward": delta,
        "points_total": int((wow.get("points_total") if claimed else points_total) or 0),
        "claimed_on": today_iso,
        "gamification": build_gamification_status(user),
    }


def award_wow_event(user: User, *, event_key: str, label: str | None = None) -> dict[str, Any]:
    key = str(event_key or "").strip().lower()
    if not key:
        return {
            "ok": False,
            "awarded": False,
            "error": "event_key_required",
            "gamification": build_gamification_status(user),
        }

    today_iso = _today_iso()
    wow = _get_wow_state(user)
    events_by_day = wow.get("events_by_day") if isinstance(wow.get("events_by_day"), dict) else {}
    events_by_day = _prune_wow_events(events_by_day)

    day_events = events_by_day.get(today_iso)
    if not isinstance(day_events, list):
        day_events = []

    awarded = key not in day_events
    delta = WOW_EVENT_REWARD_POINTS if awarded else 0

    if awarded:
        day_events.append(key)
        events_by_day[today_iso] = day_events
        wow["events_by_day"] = events_by_day
        wow["points_total"] = int(wow.get("points_total") or 0) + delta
        wow["events_total"] = int(wow.get("events_total") or 0) + 1
        wow["last_event"] = {
            "key": key,
            "label": label or key,
            "awarded_on": timezone.now().isoformat(),
        }
        _set_wow_state(user, wow)

    return {
        "ok": True,
        "awarded": awarded,
        "event_key": key,
        "label": label or key,
        "reward": delta,
        "points_total": int(wow.get("points_total") or 0),
        "gamification": build_gamification_status(user),
    }
