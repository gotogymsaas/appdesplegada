import numpy as np
import pandas as pd
# --- Config IF: bio-age scoring (asimétrico) ---
W_BIO_BONUS   = 3.0   # bono por ser biológicamente más joven (gap<0)
W_BIO_PENALTY = 7.0   # penalización por ser biológicamente más viejo (gap>0)

def score_bio_age_bonus(gap, w_bonus=W_BIO_BONUS, w_penalty=W_BIO_PENALTY):
    """
    gap: edad_biológica - edad_cronológica (años). Acepta escalar/Series/ndarray.
    Retorna puntaje 0–10 con bono (gap<0) y penalización (gap>0).
    """
    g = np.asarray(gap, dtype=float)
    bonus   = w_bonus  * (np.clip(-g, 0.0, 15.0) / 15.0)
    penalty = w_penalty* (np.clip( g, 0.0, 15.0) / 15.0)
    s = 10.0 + bonus - penalty
    return np.clip(s, 0.0, 10.0)


VARIABLES = [
    "s_steps","s_sleep","s_stress_inv","s_intensity","s_emotional",
    "s_social","s_hrv","s_bio_age","s_sleep_quality","s_circadian",
    "s_focus","s_mood_sust","s_flow","s_purpose","s_hobbies","s_prosocial"
]
DIMENSIONES = {
    "FisicaSalud": ["s_steps","s_sleep","s_stress_inv","s_intensity","s_emotional"],
    "Social": ["s_social"],
    "Biohacking": ["s_hrv","s_bio_age","s_sleep_quality","s_circadian"],
    "CognitivoEmocional": ["s_focus","s_mood_sust","s_flow"],
    "Existencial": ["s_purpose","s_hobbies"],
    "Prosocial": ["s_prosocial"]
}

class FeatureEngineerV6:
    """Genera todas las features de v6 y guarda medias/desvs para z-score en phi_*_tanh."""
    def __init__(self, means=None, stds=None):
        self.zcols = ["s_sleep_quality","s_circadian","s_prosocial","s_purpose"]
        self.means = {c: 5.5 for c in self.zcols} if means is None else dict(means)
        self.stds  = {c: 2.0 for c in self.zcols} if stds  is None else dict(stds)

    def fit(self, df: pd.DataFrame):
        for c in self.zcols:
            self.means[c] = float(df[c].mean())
            self.stds[c]  = float(df[c].std() + 1e-9)
        return self

    def _to_angle(self, s):
        return 2*np.pi*(s-1)/9.0

    def _bd(self, emo, mood_sust, flow):
        emo  = emo/10.0; mood = mood_sust/10.0; fl = flow/10.0
        z = 0.4*emo + 0.35*mood + 0.25*fl
        return float(np.clip(10.0/(1.0 + np.exp(-2.0*(z-0.55))) * 1.02, 1, 10))

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        # Recalcular s_bio_age con esquema asimétrico SÓLO si viene bio_age_gap
        if "bio_age_gap" in df.columns:
            _gap = pd.to_numeric(df["bio_age_gap"], errors="coerce")
            _mask = _gap.notna() & np.isfinite(_gap)
            df.loc[_mask, "s_bio_age"] = score_bio_age_bonus(_gap[_mask])
        for k in VARIABLES:
            if k not in df.columns:
                raise ValueError(f"Falta la columna '{k}'")

        # Interacciones
        df["nonlin_sleep_x_quality"] = df["s_sleep"] * df["s_sleep_quality"]
        df["nonlin_focus_x_mood"]    = df["s_focus"] * df["s_mood_sust"]
        df["nonlin_stress_x_social"] = df["s_stress_inv"] * df["s_social"]
        df["nonlin_social_purpose_circ"] = (df["s_social"] * df["s_purpose"] * df["s_circadian"]) / 50.0

        # phi_*_tanh con z-score fiteado
        for c in self.zcols:
            vz = (df[c] - self.means[c])/(self.stds[c])
            df[f"phi_{c}_tanh"] = 5.0*(np.tanh(0.7*vz) + 1.0)

        # Univariadas
        df["phi_social_log"] = 3.7*np.log1p(df["s_social"])
        df["phi_focus_sqrt"] = 3.3*np.sqrt(df["s_focus"])

        # Señales periódicas
        th_circ  = self._to_angle(df["s_circadian"])
        th_sleep = self._to_angle(df["s_sleep"])
        th_mix   = self._to_angle((df["s_circadian"] + df["s_sleep_quality"])/2.0)
        df["per_circ_sin"]   = 4.0*np.sin(th_circ) + 4.0
        df["per_sleep_cos"]  = 3.5*np.cos(th_sleep) + 3.5
        df["per_mix_sin"]    = 2.5*np.sin(th_mix) + 2.5
        df["per_sin_focus"]   = (np.sin(th_circ) + 1.0) * (df["s_focus"]/5.0) * 5.0
        df["per_cos_social"]  = (np.cos(th_sleep)+ 1.0) * (np.log1p(df["s_social"]))
        df["per_sin_purpose"] = (np.sin(th_mix)+1.0) * (df["s_purpose"]/2.5)

        # BD y promedios por dimensión
        df["BD"] = [self._bd(e,m,f) for e,m,f in zip(df["s_emotional"], df["s_mood_sust"], df["s_flow"])]
        for dim, cols in DIMENSIONES.items():
            df[f"avg_{dim}"] = df[cols].mean(axis=1)

        feature_order = VARIABLES + [
            "avg_FisicaSalud","avg_Social","avg_Biohacking","avg_CognitivoEmocional","avg_Existencial","avg_Prosocial",
            "BD","nonlin_sleep_x_quality","nonlin_focus_x_mood","nonlin_stress_x_social","nonlin_social_purpose_circ",
            "phi_s_sleep_quality_tanh","phi_social_log","phi_focus_sqrt","phi_s_purpose_tanh",
            "per_circ_sin","per_sleep_cos","per_mix_sin","per_sin_focus","per_cos_social","per_sin_purpose"
        ]
        return df[feature_order].values

    def get_params(self):
        return {"means": self.means, "stds": self.stds}
