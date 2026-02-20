from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LOCALE = "es-CO"


@dataclass(frozen=True)
class CalorieItem:
    item_id: str
    kcal_per_100g: float | None
    kcal_per_unit: float | None
    unit_grams: float | None
    default_serving_grams: float | None


def load_calorie_db(csv_path: Path) -> dict[str, CalorieItem]:
    items: dict[str, CalorieItem] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                continue

            def _to_float(value: str | None) -> float | None:
                raw = (value or "").strip()
                if not raw:
                    return None
                try:
                    return float(raw)
                except Exception:
                    return None

            items[item_id] = CalorieItem(
                item_id=item_id,
                kcal_per_100g=_to_float(row.get("kcal_per_100g")),
                kcal_per_unit=_to_float(row.get("kcal_per_unit")),
                unit_grams=_to_float(row.get("unit_grams")),
                default_serving_grams=_to_float(row.get("default_serving_grams")),
            )
    return items


_STOPWORDS = {
    "con",
    "de",
    "del",
    "la",
    "el",
    "los",
    "las",
    "a",
    "al",
    "y",
    "en",
    "para",
    "por",
    "una",
    "un",
}


@dataclass(frozen=True)
class NormalizedItem:
    item_id: str
    confidence: float
    method: str  # exact|fuzzy|heuristic
    matched_alias: str | None = None


def _normalize_key(value: str) -> str:
    key = re.sub(r"\s+", " ", (value or "").strip().lower())
    key = re.sub(r"[^a-záéíóúñü0-9/ ]", "", key)
    key = key.strip()
    return key


def _tokenize(value: str) -> list[str]:
    key = _normalize_key(value)
    tokens = [t for t in key.split(" ") if t and t not in _STOPWORDS]
    return tokens


def load_aliases(csv_path: Path) -> dict[str, str]:
    """Carga alias -> item_id desde CSV.

    Columnas esperadas: alias,item_id,(locale,priority opcionales)
    """

    aliases: dict[str, str] = {}
    if not csv_path.exists():
        return aliases
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alias = _normalize_key(row.get("alias") or "")
            item_id = (row.get("item_id") or "").strip()
            if alias and item_id:
                aliases[alias] = item_id
    return aliases


def default_alias_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "aliases.csv"


def default_soft_memory_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "soft_memory.json"


def _best_fuzzy_match(query: str, candidates: list[str]) -> tuple[str | None, float]:
    if not query or not candidates:
        return None, 0.0
    # difflib devuelve 0..1, suficiente para MVP.
    matches = difflib.get_close_matches(query, candidates, n=1, cutoff=0.0)
    if not matches:
        return None, 0.0
    best = matches[0]
    score = difflib.SequenceMatcher(a=query, b=best).ratio()
    return best, float(score)


def normalize_item(raw: str, *, aliases: dict[str, str] | None = None, fuzzy_cutoff: float = 0.86) -> NormalizedItem | None:
    """Normaliza un texto libre a item_id.

    - exact: match exacto contra alias
    - fuzzy: fuzzy match contra alias
    - heuristic: reglas mínimas (contiene arroz/pollo)
    """

    key = _normalize_key(raw)
    if not key:
        return None

    aliases = aliases or {}
    if key in aliases:
        return NormalizedItem(item_id=aliases[key], confidence=1.0, method="exact", matched_alias=key)

    # fuzzy: intentar sobre el texto completo
    if aliases:
        best, score = _best_fuzzy_match(key, list(aliases.keys()))
        if best and score >= fuzzy_cutoff:
            # penalizamos un poco el fuzzy vs exact
            conf = max(0.0, min(0.95, score))
            return NormalizedItem(item_id=aliases[best], confidence=conf, method="fuzzy", matched_alias=best)

    # token heuristics: ayuda con compuestos tipo "arroz con pollo"
    tokens = _tokenize(key)
    if any("arroz" == t or t.startswith("arroz") for t in tokens):
        return NormalizedItem(item_id="arroz_blanco_cocido", confidence=0.65, method="heuristic")
    if any("pollo" == t or t.startswith("pollo") for t in tokens):
        return NormalizedItem(item_id="pollo_pechuga_asada", confidence=0.65, method="heuristic")

    return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _portion_confidence(method: str) -> float:
    return {
        "explicit_grams": 1.0,
        "explicit_units": 0.8,
        "explicit_ml": 0.9,
        "cup": 0.7,
        "slice": 0.75,
        "qualitative": 0.55,
        "default": 0.5,
        "unknown": 0.2,
    }.get(method, 0.2)


def build_portion_candidates(
    item_id: str,
    *,
    calorie_db: dict[str, CalorieItem],
    locale: str = DEFAULT_LOCALE,
    memory_hint_grams: float | None = None,
    center_grams: float | None = None,
    center_source: str | None = None,  # explicit|inferred|default|memory
) -> list[dict[str, Any]]:
    """Devuelve exactamente 3 opciones de porción para UI.

    Cada opción: {label, grams, confidence_hint}
    - Si existe memoria reciente: opción #1 = repetir memoria.
    - Si existe unit_grams: ofrece 1/2/3 unidades.
    - Si existe default_serving_grams: opción central.
    - Si no hay nada: heurísticas por tipo.
    """

    ci = calorie_db.get(item_id)
    candidates: list[tuple[str, float, float]] = []

    def _add(label: str, grams: float, conf: float):
        grams = float(grams)
        if grams <= 0:
            return
        # evitar duplicados por gramos
        if any(abs(g - grams) < 1e-6 for _l, g, _c in candidates):
            return
        candidates.append((str(label), grams, float(max(0.0, min(1.0, conf)))))

    # 0) Si tenemos una porción explícita/inferida con confianza alta, estrechamos candidates alrededor.
    if center_grams and center_grams > 0 and (center_source in {"explicit", "inferred"}):
        g = float(center_grams)
        _add("Menos", g * 0.9, 0.85)
        _add("Estimado", g, 0.9)
        _add("Más", g * 1.1, 0.85)

        # Para evitar que heurísticas posteriores ensanchen el rango, retornamos exactamente estas 3.
        return [
            {"label": label, "grams": round(float(grams), 2), "confidence_hint": round(float(conf), 4)}
            for (label, grams, conf) in candidates[:3]
        ]

    # 1) Memoria suave (si no hay centro explícito)
    if (not candidates) and memory_hint_grams and memory_hint_grams > 0:
        _add("Repetir última porción", float(memory_hint_grams), 0.9)

    # 2) Unidades
    if ci and ci.unit_grams:
        ug = float(ci.unit_grams)
        # labels simples en español
        _add("1 unidad", 1.0 * ug, 0.85)
        _add("2 unidades", 2.0 * ug, 0.8)
        _add("3 unidades", 3.0 * ug, 0.75)

    # 3) Default serving como centro
    if ci and ci.default_serving_grams:
        center = float(ci.default_serving_grams)
        _add("Porción típica", center, 0.75)
        _add("Porción pequeña", max(10.0, center * 0.75), 0.65)
        _add("Porción grande", center * 1.25, 0.65)

    # 4) Heurísticas por item_id (MVP)
    if item_id == "arroz_blanco_cocido":
        _add("1/2 taza", 80.0, 0.7)
        _add("1 taza", 160.0, 0.75)
        _add("1.5 tazas", 240.0, 0.7)
    elif item_id == "pollo_pechuga_asada":
        _add("100 g", 100.0, 0.7)
        _add("150 g", 150.0, 0.75)
        _add("200 g", 200.0, 0.7)
    elif item_id == "pan_rebanada":
        # 1/2/3 rebanadas (30g por rebanada en la DB)
        base = float(ci.unit_grams) if (ci and ci.unit_grams) else 30.0
        _add("1 rebanada", 1.0 * base, 0.75)
        _add("2 rebanadas", 2.0 * base, 0.7)
        _add("3 rebanadas", 3.0 * base, 0.65)
    elif item_id == "leche_entera":
        # Aproximación: 1ml ~ 1g
        _add("150 ml", 150.0, 0.65)
        _add("250 ml", 250.0, 0.7)
        _add("350 ml", 350.0, 0.65)
    elif item_id == "ensalada_mixta":
        _add("100 g", 100.0, 0.6)
        _add("200 g", 200.0, 0.65)
        _add("300 g", 300.0, 0.6)

    # Asegurar exactamente 3 opciones.
    # Estrategia: si hay >3, elegir las 3 más útiles: priorizar memoria (si existe) + 2 cercanas al centro.
    if not candidates:
        # fallback ultra genérico
        _add("Porción pequeña", 100.0, 0.5)
        _add("Porción media", 200.0, 0.55)
        _add("Porción grande", 300.0, 0.5)

    # Orden: mantener "Repetir" primero si existe, luego por gramos asc.
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (0 if x[0].lower().startswith("repetir") else 1, x[1]),
    )

    # Reducir a 3 manteniendo diversidad
    if len(candidates_sorted) > 3:
        # si hay repetir, lo preservamos
        keep: list[tuple[str, float, float]] = []
        repeat = [c for c in candidates_sorted if c[0].lower().startswith("repetir")]
        if repeat:
            keep.append(repeat[0])
        # completar con medianas por gramos
        rest = [c for c in candidates_sorted if c not in keep]
        rest_by_g = sorted(rest, key=lambda x: x[1])
        if rest_by_g:
            keep.append(rest_by_g[len(rest_by_g) // 2])
        rest2 = [c for c in rest_by_g if c not in keep]
        if rest2:
            # elegir la más cercana a la que falta (si solo hay 2, toma la más cercana a la mediana)
            keep.append(rest2[len(rest2) // 2])
        candidates_sorted = keep[:3]

    # Si todavía faltan, rellenar duplicando escala alrededor del último
    while len(candidates_sorted) < 3:
        last = candidates_sorted[-1]
        candidates_sorted.append(("Otra opción", last[1] * 1.25, max(0.4, last[2] - 0.1)))

    return [
        {"label": label, "grams": round(float(grams), 2), "confidence_hint": round(float(conf), 4)}
        for (label, grams, conf) in candidates_sorted[:3]
    ]


def parse_portion_grams(item_id: str, portion_text: str | None, calorie_db: dict[str, CalorieItem]) -> tuple[float | None, str]:
    """Heurística MVP para porciones.

    Soporta:
    - "150g" / "150 g"
    - "2 unidades" (si hay unit_grams)
    - "1 taza" (default 160g para arroz cocido si aplica)
    """

    text = (portion_text or "").strip().lower()
    if not text:
        return None, "unknown"

    # 150g
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*g\b", text)
    if m:
        value = m.group(1).replace(",", ".")
        grams = _safe_float(value)
        if grams and grams > 0:
            return grams, "explicit_grams"

    # 250ml (aprox 1ml ~ 1g para líquidos)
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*ml\b", text)
    if m:
        value = m.group(1).replace(",", ".")
        ml = _safe_float(value)
        if ml and ml > 0:
            return float(ml), "explicit_ml"

    # 2 unidades
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(unidad|unidades)\b", text)
    if m:
        n = _safe_float(m.group(1).replace(",", "."))
        if not n or n <= 0:
            return None
        ci = calorie_db.get(item_id)
        if ci and ci.unit_grams:
            return float(n) * float(ci.unit_grams), "explicit_units"
        # fallback por si no hay unit_grams
        if item_id == "huevo":
            return float(n) * 50.0, "explicit_units"

    # 2 rebanadas (pan)
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(rebanada|rebanadas)\b", text)
    if m:
        n = _safe_float(m.group(1).replace(",", "."))
        if n and n > 0:
            ci = calorie_db.get(item_id)
            if ci and ci.unit_grams:
                return float(n) * float(ci.unit_grams), "slice"

    # 1 taza (solo MVP)
    if "taza" in text:
        if item_id == "arroz_blanco_cocido":
            return 160.0, "cup"
        # default genérico
        return 150.0, "cup"

    return None, "unknown"


def parse_portion_grams_for_item(
    item_id: str,
    portion_text: str | None,
    calorie_db: dict[str, CalorieItem],
    *,
    item_aliases: list[str] | None = None,
) -> tuple[float | None, str, str | None]:
    """Intenta extraer porción específicamente para un item cuando el texto lo menciona.

    Retorna (grams, method, matched_phrase)
    """

    text = (portion_text or "").strip().lower()
    if not text:
        return None, "unknown", None

    aliases = [a for a in (item_aliases or []) if a]
    # Buscar patrones tipo "150g de pollo" / "pollo 150g"
    for a in sorted(set(aliases), key=len, reverse=True):
        a_key = _normalize_key(a)
        if not a_key:
            continue
        # 150g de <alias>
        m = re.search(rf"(\d+(?:[\.,]\d+)?)\s*g\s*(?:de\s+)?{re.escape(a_key)}\b", _normalize_key(text))
        if m:
            grams = _safe_float(m.group(1).replace(",", "."))
            if grams and grams > 0:
                return float(grams), "explicit_grams", f"{m.group(1)}g de {a_key}"

        # <alias> 150g
        m = re.search(rf"{re.escape(a_key)}\s*(?:de\s+)?(\d+(?:[\.,]\d+)?)\s*g\b", _normalize_key(text))
        if m:
            grams = _safe_float(m.group(1).replace(",", "."))
            if grams and grams > 0:
                return float(grams), "explicit_grams", f"{a_key} {m.group(1)}g"

        # 2 unidades de <alias>
        m = re.search(rf"(\d+(?:[\.,]\d+)?)\s*(unidad|unidades)\s*(?:de\s+)?{re.escape(a_key)}\b", _normalize_key(text))
        if m:
            n = _safe_float(m.group(1).replace(",", "."))
            if n and n > 0:
                ci = calorie_db.get(item_id)
                if ci and ci.unit_grams:
                    return float(n) * float(ci.unit_grams), "explicit_units", f"{m.group(1)} unidades de {a_key}"
                if item_id == "huevo":
                    return float(n) * 50.0, "explicit_units", f"{m.group(1)} unidades de {a_key}"

        # 1 taza de <alias>
        if re.search(rf"\b(taza|tazas)\s*(?:de\s+)?{re.escape(a_key)}\b", _normalize_key(text)):
            if item_id == "arroz_blanco_cocido":
                return 160.0, "cup", f"taza de {a_key}"
            return 150.0, "cup", f"taza de {a_key}"

        # 250ml de <alias>
        m = re.search(rf"(\d+(?:[\.,]\d+)?)\s*ml\s*(?:de\s+)?{re.escape(a_key)}\b", _normalize_key(text))
        if m:
            ml = _safe_float(m.group(1).replace(",", "."))
            if ml and ml > 0:
                return float(ml), "explicit_ml", f"{m.group(1)}ml de {a_key}"

        # 2 rebanadas de <alias>
        m = re.search(rf"(\d+(?:[\.,]\d+)?)\s*(rebanada|rebanadas)\s*(?:de\s+)?{re.escape(a_key)}\b", _normalize_key(text))
        if m:
            n = _safe_float(m.group(1).replace(",", "."))
            if n and n > 0:
                ci = calorie_db.get(item_id)
                if ci and ci.unit_grams:
                    return float(n) * float(ci.unit_grams), "slice", f"{m.group(1)} rebanadas de {a_key}"

        # Cualitativo simple: "ensalada grande"
        if item_id == "ensalada_mixta" and re.search(r"\bgrande\b", _normalize_key(text)) and a_key in _normalize_key(text):
            return 250.0, "qualitative", "ensalada grande"

    # fallback global (no específico)
    grams, method = parse_portion_grams(item_id, portion_text, calorie_db)
    return grams, method, None


def shannon_entropy(probs: list[float]) -> float:
    p = [x for x in probs if x and x > 0]
    if not p:
        return 0.0
    s = sum(p)
    p = [x / s for x in p]
    return -sum(pi * math.log(pi + 1e-12) for pi in p)


def qaf_infer_from_vision(
    vision: dict[str, Any],
    calorie_db: dict[str, CalorieItem],
    *,
    aliases: dict[str, str] | None = None,
    user_id: str | None = None,
    locale: str = DEFAULT_LOCALE,
) -> dict[str, Any]:
    raw_items = vision.get("items")
    if not isinstance(raw_items, list):
        raw_items = []

    aliases = aliases or load_aliases(default_alias_path())

    normalized_items: list[NormalizedItem] = []

    def _add_norm(n: NormalizedItem | None):
        if not n:
            return
        if all(x.item_id != n.item_id for x in normalized_items):
            normalized_items.append(n)

    for it in raw_items:
        raw = str(it)
        # 1) Intento directo sobre la frase completa
        _add_norm(normalize_item(raw, aliases=aliases))

        # 2) Intento multi-item para frases compuestas: n-grams sobre tokens.
        # Ej: "arroz con pollo" debe poder generar [arroz_blanco_cocido, pollo_pechuga_asada]
        tokens = _tokenize(raw)
        if len(tokens) >= 2:
            # n-grams de 1..3 (cap) para no explotar combinatoria.
            for n in (3, 2, 1):
                if len(tokens) < n:
                    continue
                for i in range(0, len(tokens) - n + 1):
                    phrase = " ".join(tokens[i : i + n])
                    _add_norm(normalize_item(phrase, aliases=aliases))

    normalized_ids = [x.item_id for x in normalized_items]

    portion_text = str(vision.get("portion_estimate") or "").strip() or None

    # Entropía: la dejamos como señal futura; hoy la probabilidad es uniforme.
    probs = [1.0 for _ in normalized_ids]
    ent = shannon_entropy(probs)

    items_out = []
    total_kcal = 0.0
    missing = []
    reasons: list[str] = []
    follow_up_questions: list[dict[str, Any]] = []

    # Reverse index item_id -> aliases (para parsing por item)
    item_aliases: dict[str, list[str]] = {}
    for alias_key, item_id in aliases.items():
        item_aliases.setdefault(item_id, []).append(alias_key)

    per_item_conf: dict[str, dict[str, Any]] = {}

    # memoria suave: opcional (sin dependencias). Si no hay archivo, se ignora.
    memory: dict[str, Any] = {}
    try:
        mp = default_soft_memory_path()
        if mp.exists():
            memory = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        memory = {}

    def _get_memory_grams(uid: str | None, iid: str) -> float | None:
        if not uid:
            return None
        entry = ((memory or {}).get(str(uid)) or {}).get(str(iid)) or {}
        grams = entry.get("grams")
        ts = entry.get("ts")
        try:
            grams_f = float(grams)
        except Exception:
            return None
        # Ventana de recencia simple: 14 días si hay timestamp epoch.
        try:
            import time

            if ts is None:
                return grams_f
            ts_f = float(ts)
            if (time.time() - ts_f) <= (14 * 24 * 3600):
                return grams_f
        except Exception:
            return grams_f
        return None

    for nitem in normalized_items:
        item_id = nitem.item_id
        ci = calorie_db.get(item_id)
        grams, portion_method, portion_match = parse_portion_grams_for_item(
            item_id,
            portion_text,
            calorie_db,
            item_aliases=item_aliases.get(item_id) or [],
        )

        used_default = False
        if grams is None and ci and ci.default_serving_grams:
            grams = float(ci.default_serving_grams)
            portion_method = "default"
            used_default = True

        kcal = None
        if ci:
            if grams is not None and ci.kcal_per_100g is not None:
                kcal = (float(grams) / 100.0) * float(ci.kcal_per_100g)
            elif ci.kcal_per_unit is not None:
                # Si tenemos gramos y unit_grams, escalamos por número de unidades.
                if grams is not None and ci.unit_grams:
                    kcal = (float(grams) / float(ci.unit_grams)) * float(ci.kcal_per_unit)
                else:
                    kcal = float(ci.kcal_per_unit)
                    grams = grams or ci.unit_grams

        if kcal is None:
            missing.append(item_id)
            reasons.append(f"missing_item_in_db:{item_id}")
            continue

        total_kcal += float(kcal)

        conf_portion = _portion_confidence(portion_method)
        conf_item = float(nitem.confidence)
        conf_combined = max(0.0, min(1.0, 0.65 * conf_item + 0.35 * conf_portion))

        if nitem.method != "exact":
            reasons.append(f"item_normalized_{nitem.method}:{item_id}")
        if used_default:
            reasons.append(f"used_default_serving:{item_id}")
        if grams is None:
            reasons.append(f"portion_unknown:{item_id}")

        per_item_conf[item_id] = {
            "confidence_item": round(conf_item, 4),
            "confidence_portion": round(conf_portion, 4),
            "confidence": round(conf_combined, 4),
            "item_method": nitem.method,
            "portion_method": portion_method,
            "matched_alias": nitem.matched_alias,
            "portion_match": portion_match,
        }

        mem_grams = _get_memory_grams(user_id, item_id)
        # Centro: si la porción viene explícita, estrechar candidates.
        center = None
        center_source = None
        if grams is not None and portion_method in {"explicit_grams", "explicit_ml", "explicit_units"}:
            center = float(grams)
            center_source = "explicit"
        elif grams is not None and portion_method in {"cup", "slice", "qualitative"}:
            center = float(grams)
            center_source = "inferred"

        portion_candidates = build_portion_candidates(
            item_id,
            calorie_db=calorie_db,
            locale=locale,
            memory_hint_grams=mem_grams,
            center_grams=center,
            center_source=center_source,
        )

        items_out.append(
            {
                "item_id": item_id,
                "grams": grams,
                "calories": round(float(kcal), 2),
                "portion_candidates": portion_candidates,
                "selected_portion": None,
                "item_confidence": round(conf_item, 4),
                "portion_confidence": round(conf_portion, 4),
            }
        )

    # Confianza total ponderada por contribución calórica.
    total_conf = 0.0
    if items_out and total_kcal > 0:
        for it in items_out:
            item_id = it.get("item_id")
            kcal = float(it.get("calories") or 0.0)
            w = kcal / total_kcal if total_kcal else 0.0
            c = float((per_item_conf.get(str(item_id)) or {}).get("confidence") or 0.0)
            total_conf += w * c

    # uncertainty_score: 0 (seguro) .. 1 (incierto)
    uncertainty_score = max(0.0, min(1.0, 1.0 - total_conf))

    # Rango inteligente: si hay candidates, usar min/max por ítem.
    per_item_spread: dict[str, float] = {}
    low = 0.0
    high = 0.0
    best_estimate = 0.0
    for it in items_out:
        item_id = str(it.get("item_id") or "")
        ci = calorie_db.get(item_id)
        cands = it.get("portion_candidates") or []
        if not ci or not isinstance(cands, list) or not cands:
            kcal = float(it.get("calories") or 0.0)
            low += kcal
            high += kcal
            best_estimate += kcal
            continue

        def _kcal_for_grams(g: float) -> float:
            if ci.kcal_per_100g is not None:
                return (float(g) / 100.0) * float(ci.kcal_per_100g)
            if ci.kcal_per_unit is not None and ci.unit_grams:
                return (float(g) / float(ci.unit_grams)) * float(ci.kcal_per_unit)
            return float(it.get("calories") or 0.0)

        kcal_opts = [_kcal_for_grams(float(c.get("grams") or 0.0)) for c in cands]
        kcal_opts = [k for k in kcal_opts if k >= 0]
        if not kcal_opts:
            kcal = float(it.get("calories") or 0.0)
            low += kcal
            high += kcal
            best_estimate += kcal
            continue

        item_low = min(kcal_opts)
        item_high = max(kcal_opts)
        item_mid = kcal_opts[len(kcal_opts) // 2]
        low += item_low
        high += item_high
        # best estimate = cálculo actual (si existe) o mid de candidates
        best_estimate += float(it.get("calories") or item_mid)
        per_item_spread[item_id] = float(item_high - item_low)

    # Si no pudimos calcular, usar total_kcal previo.
    if best_estimate <= 0 and total_kcal > 0:
        best_estimate = total_kcal
        low = max(0.0, total_kcal * (1.0 - (0.12 + 0.23 * uncertainty_score)))
        high = total_kcal * (1.0 + (0.12 + 0.23 * uncertainty_score))

    total_kcal = float(best_estimate)

    range_driver = None
    if per_item_spread:
        range_driver = sorted(per_item_spread.items(), key=lambda kv: kv[1], reverse=True)[0][0]

    # needs_confirmation por impacto (colapso):
    range_width = max(0.0, high - low)
    rel_width = (range_width / total_kcal) if total_kcal else 1.0
    # Umbral MVP: pedir confirmación cuando el rango es lo suficientemente grande para cambiar decisiones.
    needs_confirmation = bool(missing) or (rel_width >= 0.35)

    # follow_up_questions: lista lista para UI (máximo 1 pregunta en MVP)
    if needs_confirmation and range_driver:
        driver_item = next((x for x in items_out if x.get("item_id") == range_driver), None)
        if driver_item:
            follow_up_questions.append(
                {
                    "type": "confirm_portion",
                    "item_id": range_driver,
                    "prompt": f"Confirma la porción de {range_driver}",
                    "options": driver_item.get("portion_candidates") or [],
                }
            )

    # decision por score (MVP): incertidumbre + defaults + missing + costo preguntar
    defaults_used = 0
    for iid, info in per_item_conf.items():
        if str(info.get("portion_method")) == "default":
            defaults_used += 1
    D = (defaults_used / len(per_item_conf)) if per_item_conf else 0.0
    M = 1.0 if missing else 0.0
    U = float(uncertainty_score)
    A = 1.0 if follow_up_questions else 0.0

    w_uncertainty, w_default, w_missing, w_ask = 1.0, 0.4, 1.2, 0.3
    decision_score = (w_uncertainty * U) + (w_default * D) + (w_missing * M) + (w_ask * A)

    if missing:
        decision = "needs_confirmation"
        decision_reason = "missing_items_in_db"
    elif needs_confirmation and len(items_out) > 1:
        decision = "partial"
        decision_reason = "confirm_range_driver"
    elif needs_confirmation:
        decision = "needs_confirmation"
        decision_reason = "high_uncertainty"
    else:
        decision = "accepted"
        decision_reason = "low_uncertainty"

    # Entropía alta (futura): mantenemos la señal por compatibilidad.
    if ent >= 1.0 and len(normalized_ids) >= 3:
        reasons.append("high_entropy_uniform")

    explainability = []
    explainability.append(
        f"Estimado: {round(total_kcal, 0):.0f} kcal (rango {round(low, 0):.0f}–{round(high, 0):.0f})"
    )
    if range_driver:
        explainability.append(f"Lo que más mueve el rango: {range_driver}")
    explainability = explainability[:2]

    return {
        "is_food": bool(vision.get("is_food")) if "is_food" in vision else None,
        "items": items_out,
        "missing_items": missing,
        "total_calories": round(total_kcal, 2),
        "total_calories_range": {"low": round(low, 2), "high": round(high, 2)},
        "uncertainty": {
            "entropy": round(ent, 4),
            "uncertainty_score": round(float(uncertainty_score), 4),
            "portion_text": portion_text,
        },
        "needs_confirmation": bool(needs_confirmation),
        "confidence": {
            "total": round(float(total_conf), 4),
            "per_item": per_item_conf,
        },
        "reasons": sorted(set([r for r in reasons if r])),
        "follow_up_questions": follow_up_questions,
        # compat: alias
        "suggested_questions": follow_up_questions,
        "range_driver": range_driver,
        "explainability": explainability,
        "decision": decision,
        "decision_reason": decision_reason,
        "decision_score": round(float(decision_score), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calorie-db", required=True)
    parser.add_argument("--vision-json", required=True, help="JSON string o ruta a .json")
    args = parser.parse_args()

    calorie_db = load_calorie_db(Path(args.calorie_db))

    vision_arg = args.vision_json
    if Path(vision_arg).exists():
        vision = json.loads(Path(vision_arg).read_text(encoding="utf-8"))
    else:
        vision = json.loads(vision_arg)

    # Extensiones opcionales
    user_id = None
    locale = DEFAULT_LOCALE
    if isinstance(vision, dict):
        user_id = (vision.get("user_id") or None) if isinstance(vision.get("user_id"), str) else None
        locale = str(vision.get("locale") or DEFAULT_LOCALE)
        # compat: si el JSON se pasó como {vision: {...}, user_id:...}
        if isinstance(vision.get("vision"), dict):
            user_id = str(vision.get("user_id") or user_id or "") or None
            locale = str(vision.get("locale") or locale)
            vision = vision.get("vision")

    result = qaf_infer_from_vision(vision, calorie_db, user_id=user_id, locale=locale)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
