import os
from pathlib import Path

# Lista base de las 16 variables del IF.
# Se declara aquí para no forzar dependencias pesadas (numpy/pandas) en el camino determinista.
VARIABLES = [
    "s_steps",
    "s_sleep",
    "s_stress_inv",
    "s_intensity",
    "s_emotional",
    "s_social",
    "s_hrv",
    "s_bio_age",
    "s_sleep_quality",
    "s_circadian",
    "s_focus",
    "s_mood_sust",
    "s_flow",
    "s_purpose",
    "s_hobbies",
    "s_prosocial",
]

# Carpeta donde están los modelos .pkl (solo se usa si GTG_IF_USE_MODELS=1)
MODELS_DIR = Path(__file__).resolve().parent / "modelos_v6"

# Caché global para no recargar los modelos en cada worker
_fe = None
_ridge = None


def _as_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _clamp_0_10(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 10.0:
        return 10.0
    return value


def _use_models_default() -> bool:
    # Opt-in explícito: por defecto NO cargamos modelos (evita latencias de 10–15s en cold start).
    return str(os.getenv("GTG_IF_USE_MODELS", "0")).strip().lower() in ("1", "true", "yes", "y", "on")


def _load_models_if_needed():
    """Carga perezosa del FeatureEngineer y modelo Ridge (solo modo ML)."""
    global _fe, _ridge
    if _fe is not None and _ridge is not None:
        return

    # Imports pesados solo en modo ML
    import joblib
    from .feature_engineer_v6 import FeatureEngineerV6

    if _fe is None:
        fe_path = MODELS_DIR / "if_v6_feature_engineer.pkl"
        fe_loaded = joblib.load(fe_path)
        if isinstance(fe_loaded, dict):
            _fe = FeatureEngineerV6(means=fe_loaded.get("means"), stds=fe_loaded.get("stds"))
        else:
            _fe = fe_loaded

    if _ridge is None:
        ridge_path = MODELS_DIR / "if_ridge_sint_v6.pkl"
        _ridge = joblib.load(ridge_path)


def predict_if_from_scores(scores: dict, *, use_models: bool | None = None):
    """
    Recibe un diccionario con las 16 variables base del IF (todas en escala 1–10),
    calcula las features avanzadas, las pasa a los modelos y devuelve:

        (IF_pred_ridge, IF_pred_stack_or_None)

    Ejemplo:
        scores = {
            "s_steps": 7,
            "s_sleep": 8,
            "s_stress_inv": 6,
            ...
            "s_prosocial": 8
        }
    """

    if not isinstance(scores, dict):
        scores = {}

    # Modo por defecto: determinista y ultra-rápido.
    # IMPORTANTE: no tocar disco, no cargar joblib, no crear DataFrames.
    if use_models is None:
        use_models = _use_models_default()

    # Validación mínima + orden estable
    values = [_as_float(scores.get(name), default=0.0) for name in VARIABLES]

    if not use_models:
        # Promedio simple en escala 0–10
        avg_score = sum(values) / float(len(values))
        final_score = round(_clamp_0_10(avg_score), 1)
        return final_score, None

    # Modo ML (opt-in): usa Ridge y clampa a 0–10. Si falla, cae al modo determinista.
    try:
        _load_models_if_needed()
        import numpy as np
        import pandas as pd

        df = pd.DataFrame([values], columns=VARIABLES)
        if hasattr(_fe, "transform"):
            X = _fe.transform(df)
        else:
            # Si el pickle trae el FE ya listo, igual debería tener transform.
            X = df.values

        pred = _ridge.predict(X)
        ridge_val = float(np.asarray(pred).reshape(-1)[0])
        ridge_val = round(_clamp_0_10(ridge_val), 1)
        return ridge_val, None
    except Exception:
        avg_score = sum(values) / float(len(values))
        final_score = round(_clamp_0_10(avg_score), 1)
        return final_score, None
