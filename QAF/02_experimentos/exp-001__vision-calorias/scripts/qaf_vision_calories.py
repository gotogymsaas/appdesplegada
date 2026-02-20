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
        "cup": 0.7,
        "default": 0.5,
        "unknown": 0.2,
    }.get(method, 0.2)


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
    suggested_questions: list[dict[str, Any]] = []

    # Reverse index item_id -> aliases (para parsing por item)
    item_aliases: dict[str, list[str]] = {}
    for alias_key, item_id in aliases.items():
        item_aliases.setdefault(item_id, []).append(alias_key)

    per_item_conf: dict[str, dict[str, Any]] = {}

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

        items_out.append(
            {
                "item_id": item_id,
                "grams": grams,
                "calories": round(float(kcal), 2),
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

    # Rango: depende de incertidumbre de porción.
    # Base_pct se ajusta de 12% a 35% según uncertainty_score.
    base_pct = 0.12 + 0.23 * uncertainty_score
    low = max(0.0, total_kcal * (1.0 - base_pct))
    high = total_kcal * (1.0 + base_pct)

    # needs_confirmation por impacto (colapso): pedimos confirmación si el rango es grande.
    # - Siempre si faltan items en DB.
    # - O si el ancho relativo del rango supera 25%.
    range_width = max(0.0, high - low)
    rel_width = (range_width / total_kcal) if total_kcal else 1.0
    # Umbral MVP: pedir confirmación cuando el rango es lo suficientemente grande para cambiar decisiones.
    needs_confirmation = bool(missing) or (rel_width >= 0.35)

    # suggested_questions: solo cuando realmente vamos a pedir confirmación.
    if needs_confirmation and items_out:
        # elegir candidato: mayor contribución kcal y menor confianza
        def _score(it: dict[str, Any]) -> float:
            item_id = str(it.get("item_id") or "")
            kcal = float(it.get("calories") or 0.0)
            conf = float((per_item_conf.get(item_id) or {}).get("confidence") or 0.0)
            return (kcal / (total_kcal or 1.0)) * (1.0 - conf)

        target = sorted(items_out, key=_score, reverse=True)[0]
        target_id = str(target.get("item_id") or "")
        if target_id:
            # opciones simples por item (MVP)
            if target_id == "arroz_blanco_cocido":
                grams_opts = [80, 160, 240]
            elif target_id == "pollo_pechuga_asada":
                grams_opts = [100, 150, 200]
            elif target_id == "huevo":
                grams_opts = [50, 100, 150]
            else:
                grams_opts = []

            if grams_opts:
                opts = []
                ci = calorie_db.get(target_id)
                for g in grams_opts:
                    kcal = None
                    if ci and ci.kcal_per_100g is not None:
                        kcal = (float(g) / 100.0) * float(ci.kcal_per_100g)
                    opts.append({"grams": g, "calories": round(float(kcal), 2) if kcal is not None else None})

                suggested_questions.append(
                    {
                        "type": "confirm_portion",
                        "item_id": target_id,
                        "prompt": f"Confirma la porción de {target_id} para mejorar la estimación",
                        "options": opts,
                    }
                )

    # Entropía alta (futura): mantenemos la señal por compatibilidad.
    if ent >= 1.0 and len(normalized_ids) >= 3:
        reasons.append("high_entropy_uniform")

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
        "suggested_questions": suggested_questions,
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

    result = qaf_infer_from_vision(vision, calorie_db)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
