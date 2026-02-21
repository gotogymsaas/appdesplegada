# Propuesta — Exp-010 (QAF Medición muscular solo con fotografía)

## 0) Contexto
Este experimento vive dentro de la ruta final **Salud** del Vision Router (ver `docs/VISION_ROUTER_4_RUTAS.md`).

El propósito es dar una experiencia tipo “coach” con métricas **comparables** semana a semana, sin vender precisión clínica.

---

## 1) Qué sí y qué no (según arquitectura actual)

### Sí podemos (con la arquitectura actual)
- Recibir fotos por el chat (ya existe upload de attachments + `POST /api/chat/`).
- Clasificar ruta (nutrition/training/health/quantum) con Azure OpenAI Vision.
- Persistir estado semanal y resultados en `coach_weekly_state` (patrón ya usado por Exp-003/004/005).
- Aplicar metodología QAF: `decision`, `confidence`, `uncertainty_score`, `follow_up_questions`, guardrails.
- Producir “wow” textual en el chat (mapa muscular simplificado + top 3 insights + acciones).

### No podemos (hoy) sin agregar componentes
- **Segmentación corporal real** (cuerpo vs fondo) y mediciones por región con fiabilidad alta en backend.
- **Landmarks/pose estimation** en backend: el motor de postura (Exp-006) ya asume keypoints calculados por cliente o microservicio.
- Medir **centímetros reales** sin referencia de escala (cinta/objeto conocido/calibración).
- Garantizar una “definición visual” robusta a iluminación sin un pipeline de visión dedicado (control de luz + normalización).

### Implicación
El MVP de Exp-010 debe:
- Basarse en **ratios** y **cambios relativos (%)**.
- Incluir una variable #7 `consistency/confidence` que indique si la sesión es comparable.
- Pedir repetir captura cuando falten condiciones mínimas.

---

## 2) Diseño QAF (estado, colapso y guardrails)

### Estado (operacional)
Representamos una sesión de medición como un estado `S`:
- `images`: set de 4 vistas requeridas.
- `quality`: iluminación, nitidez, encuadre (heurísticas).
- `measurements`: variables 1..7.
- `confidence`: score 0..1.

### Decisión
- `decision = accepted` si: 4 fotos presentes + calidad mínima + consistencia suficiente.
- `decision = needs_confirmation` si falta una vista o la calidad es baja.

### Colapso humano mínimo
Si `needs_confirmation`, devolver **una** instrucción clara + quick-actions para retomar captura.

---

## 3) Variables (7) — definición operacional

1) **Simetría bilateral** (0–100)
- Idea: comparar izquierda vs derecha por segmentos.
- MVP: usando ratios relativos por región (cuando exista segmentación/landmarks; si no, aproximación por heurística + baja confianza).

2) **Volumen aparente por grupo** (0–100)
- Idea: “tamaño” visual relativo por grupo (hombros, pecho/espalda, glúteo, muslo, brazo).
- MVP: área/contorno por región normalizado por estatura (si hay landmarks). Sin landmarks ⇒ solo insight cualitativo.

3) **V-Taper / silueta** (0–100)
- Ratios hombros:cintura y pecho:cintura.

4) **Balance tren superior vs inferior** (0–100)
- Ratio volumen superior / volumen inferior.

5) **Postura estática** (0–100)
- Enfoque: alineación hombros/pelvis.
- Nota: si hay keypoints, podemos reutilizar ideas de Exp-006 (sin diagnosticar).

6) **Definición visual controlada** (0–100)
- MVP: contrastes/textura con control de calidad de luz; si luz mala ⇒ score se invalida o baja confianza.

7) **Consistencia de medición (confianza)** (0–100)
- Score por: distancia/encuadre, iluminación, nitidez, postura comparable.

---

## 4) Ecuaciones / principios y autores (lo que vamos a usar)

> Nota: Exp-010 usa dos capas: (a) métricas geométricas/estadísticas simples y (b) metodología QAF para incertidumbre/colapso.

### Autores en la capa QAF (incertidumbre/decisión)
- **Claude E. Shannon** — Entropía de Shannon (incertidumbre por información incompleta).
- **John von Neumann** — Entropía de von Neumann (referencia conceptual de mezcla/incoherencia de estado).
- **Thomas Bayes** — Regla de Bayes (actualización de creencias con nueva evidencia).
- **Andrey Kolmogorov** — Complejidad de Kolmogorov (referencia conceptual para evitar narrativas sobreajustadas).
- **Karl Friston** — Principio de energía libre (referencia conceptual para tradeoff entre sorpresa y modelo).
- **Erwin Schrödinger** — ecuación de Schrödinger (metáfora operacional para dinámica del estado).
- **William Rowan Hamilton** — Hamiltoniano (metáfora operacional para acoplamientos del sistema).

### Autores típicos en la capa geométrica (medidas/ratios)
- **Euclides** — geometría euclidiana (distancias, proporciones).
- **Pitágoras** — teorema de Pitágoras (distancias en plano).
- **René Descartes** — coordenadas cartesianas (x,y para landmarks).

> En implementación práctica, estas ecuaciones son “matemática estándar”; los autores se citan por rigor histórico.

---

## 5) Contrato JSON (API interna del motor)

Entrada propuesta:
```json
{
  "images": {
    "front_relaxed": {"url": "..."},
    "side_right_relaxed": {"url": "..."},
    "back_relaxed": {"url": "..."},
    "front_flex": {"url": "..."}
  },
  "user_context": {
    "height_cm": 175
  },
  "week_id": "2026-W08",
  "baseline": {"week_id": "2026-W07"}
}
```

Salida propuesta:
```json
{
  "decision": "accepted|needs_confirmation",
  "confidence": {"score": 0.0, "uncertainty_score": 0.0, "missing": []},
  "variables": {
    "symmetry": 0,
    "volume_by_group": {"shoulders": 0, "back": 0, "glutes": 0, "thigh": 0, "arms": 0},
    "v_taper": 0,
    "upper_lower_balance": 0,
    "static_posture": 0,
    "definition": 0,
    "measurement_consistency": 0
  },
  "insights": ["...", "...", "..."],
  "recommended_actions": ["...", "...", "..."],
  "routine": [{"id": "...", "name": "..."}],
  "follow_up_questions": [],
  "meta": {"algorithm": "exp-010_muscle_measure_v0", "as_of": "YYYY-MM-DD"}
}
```

---

## 6) Checklist de implementación (cuando pasemos a código)
- [ ] Definir cómo capturar 4 fotos desde chat sin pantallas nuevas (secuencia + estado en `coach_state`).
- [ ] Definir dónde persistir imágenes (attachments) y cómo referenciarlas por `week_id`.
- [ ] Crear motor `backend/api/qaf_muscle_measure/engine.py` con `decision/confidence`.
- [ ] Crear endpoint `POST /api/qaf/muscle_measure/`.
- [ ] Integrar en `chat_n8n` cuando Vision `route=health` y el usuario elija “Comparar músculo”.
- [ ] Guardrails: bloquear conclusiones si `measurement_consistency` baja.
- [ ] Dataset JSONL mínimo + runner local estilo Exp-001.

