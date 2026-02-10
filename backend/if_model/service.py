# api/if_model/service.py

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .feature_engineer_v6 import FeatureEngineerV6, VARIABLES

# Carpeta donde están los modelos .pkl
MODELS_DIR = Path(__file__).resolve().parent / "modelos_v6"

# Caché global para no recargar los modelos en cada petición
_fe = None
_ridge = None
_mlp = None
_xgb = None
_meta = None


def _load_models():
    """
    Carga perezosa (lazy load) del FeatureEngineer y de los modelos entrenados.
    """
    global _fe, _ridge, _mlp, _xgb, _meta

    # -------------------------------
    # Cargar Feature Engineer
    # -------------------------------
    if _fe is None:
        fe_path = MODELS_DIR / "if_v6_feature_engineer.pkl"
        fe_loaded = joblib.load(fe_path)

        # Si viene como diccionario con medias/desviaciones:
        if isinstance(fe_loaded, dict):
            _fe = FeatureEngineerV6(
                means=fe_loaded["means"],
                stds=fe_loaded["stds"]
            )
        else:
            _fe = fe_loaded

    # -------------------------------
    # Modelo Ridge (obligatorio)
    # -------------------------------
    if _ridge is None:
        ridge_path = MODELS_DIR / "if_ridge_sint_v6.pkl"
        _ridge = joblib.load(ridge_path)

    # -------------------------------
    # Modelos opcionales para stacking
    # -------------------------------
    try:
        mlp_path = MODELS_DIR / "if_mlp_sint_v6.pkl"
        xgb_path = MODELS_DIR / "if_xgb_sint_v6.pkl"

        if mlp_path.exists() and xgb_path.exists():
            _mlp = joblib.load(mlp_path)
            _xgb = joblib.load(xgb_path)

            # Meta-modelo (stacking)
            meta_x = MODELS_DIR / "if_meta_XGB-meta_sint_v6.pkl"
            meta_r = MODELS_DIR / "if_meta_RidgeCV-meta_sint_v6.pkl"

            if meta_x.exists():
                _meta = joblib.load(meta_x)
            elif meta_r.exists():
                _meta = joblib.load(meta_r)
            else:
                _meta = None

    except Exception:
        _mlp = None
        _xgb = None
        _meta = None


def predict_if_from_scores(scores: dict):
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

    _load_models()

    # Asegurar el orden correcto (importante)
    row = [scores[name] for name in VARIABLES]
    df = pd.DataFrame([row], columns=VARIABLES)

    # CÁLCULO DETERMINISTA (Linealidad Exacta 0-100%)
    # El usuario requiere que si los inputs son 0-10, el output sea proporcional.
    # El modelo ML puede sesgarse, así que usaremos un promedio ponderado robusto.
    
    # Lista de valores (todas las variables pesan igual por ahora)
    values = [scores[name] for name in VARIABLES]
    
    # Promedio simple
    avg_score = sum(values) / len(values)
    
    # El resultado ya está en escala 0-10
    final_score = avg_score
    
    # Redondear a 1 decimal para consistencia
    final_score = round(final_score, 1)

    return final_score, None
