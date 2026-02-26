# Reporte comparativo de eficiencia QAF

Fecha de corte: 2026-02-26  
Fuente: ejecución local de `run_local_eval.py` sobre datasets `dataset.sample.jsonl` (Exp-001 a Exp-009).

## 1) Comparativo por experimento (porcentual)

| Experimento | n | Métrica principal | Valor % | Métricas secundarias |
|---|---:|---|---:|---|
| Exp-001 Vision-Calorías | 3 | Cobertura calorías | **100.0%** | `needs_confirmation`: 0.0%, `missing_items`: 0.0%, `range_coverage`: 100.0%, MAE: 0.33 |
| Exp-002 Coherencia Meta-Alertas | 3 | Accepted | **66.7%** | `needs_confirmation`: 33.3%, `with_alerts`: 66.7%, `partial`: 0.0% |
| Exp-003 Perfil Metabólico | 3 | Accepted | **66.7%** | `needs_confirmation`: 33.3%, `with_alerts`: 33.3% |
| Exp-004 Meal Planner | 2 | Plan semanal válido (`days_ok`) | **100.0%** | 2/2 planes con 7 días |
| Exp-005 Predictor Tendencias | 2 | Accepted | **50.0%** | `needs_confirmation`: 50.0% |
| Exp-006 Postura Correctiva | 2 | Accepted | **50.0%** | `needs_confirmation`: 50.0% |
| Exp-007 Lifestyle Intelligence | 2 | Accepted | **50.0%** | `needs_confirmation`: 50.0% |
| Exp-008 Motivación Psicológica | 3 | Sin `needs_confirmation` (autonomía) | **66.7%** | `needs_confirmation`: 33.3% |
| Exp-009 Progresión Inteligente | 2 | Sin `needs_confirmation` (autonomía) | **100.0%** | `needs_confirmation`: 0.0% |

---

## 2) Evaluaciones porcentuales consolidadas

### 2.1 Eficiencia de decisión automática (sin pedir confirmación)

- Base: experimentos con métrica `needs_confirmation` explícita (Exp-001, 002, 003, 005, 006, 007, 008, 009).  
- Total evaluado: **20 casos**.  
- Casos con `needs_confirmation`: **6/20 = 30.0%**.  
- Casos sin `needs_confirmation` (autonomía operativa): **14/20 = 70.0%**.

### 2.2 Tasa de aceptación (solo experimentos que reportan `accepted`)

- Base: Exp-002, 003, 005, 006, 007.  
- Total evaluado: **12 casos**.  
- `accepted`: **7/12 = 58.3%**.  
- No `accepted`: **5/12 = 41.7%**.

### 2.3 Calidad de cobertura/calibración en visión (Exp-001)

- Cobertura de calorías: **100.0%**.  
- Cobertura de rango vs GT: **100.0%**.  
- Error absoluto medio (MAE): **0.33** (sobre muestra pequeña).

### 2.4 Cumplimiento estructural de planificación (Exp-004)

- Planes semanales completos: **100.0%** (2/2).

---

## 3) Lectura de eficiencia QAF (ejecutiva)

1. **Fortalezas actuales**
   - Los pipelines MVP evaluados están funcionales (sin fallas de ejecución en la corrida local).
   - Experimentos de estructura (Exp-001 y Exp-004) muestran desempeño alto en cobertura y completitud (100%).

2. **Zona media de eficiencia**
   - Exp-002 y Exp-003 sostienen **66.7%** de aceptación con control de riesgo vía `needs_confirmation` (**33.3%**).

3. **Zona de fricción operativa**
   - Exp-005/006/007 presentan **50% accepted / 50% needs_confirmation** en muestra actual.
   - Esto protege seguridad de decisión, pero reduce automatización y fluidez UX.

4. **Conclusión global de eficiencia (muestra actual)**
   - Eficiencia automática estimada: **70.0%** (sin requerir confirmación humana).
   - Eficiencia de aceptación explícita: **58.3%** en módulos que reportan `accepted`.

---

## 4) Estado de benchmarking de arquitectura (chat)

La arquitectura ya está instrumentada para benchmark de runtime con:
- `qaf_fastpath`
- `fallback_used`
- `latency_ms_total`
- `latency_ms_n8n`
- `latency_ms_qaf`
- `tokens_in` / `tokens_out` (si provienen de n8n)

Middleware: `api.middleware.BenchmarkChatMiddleware` (activo por flag `BENCHMARK_CHAT_LOGS`).

### Hallazgo
Aún no hay evidencia consolidada en documentación de resultados productivos (p95, costo/token, fastpath real, fallback real) con ventanas temporales. Es decir: la telemetría existe, pero falta el tablero/hábito de consolidación periódica.

---

## 5) Recomendación de lectura del comparativo

Para comprender eficiencia de forma práctica:
1. Usar **autonomía operativa (70.0%)** como KPI transversal inicial.
2. Usar **accepted_rate** por experimento para detectar módulos con mayor fricción (005/006/007).
3. Mantener **guardrails** (`needs_confirmation`) donde el riesgo sea alto, pero reducir falsos positivos vía calibración de umbrales y más dataset.

---

## 6) Nota metodológica

- Este reporte refleja **benchmark de laboratorio** con datasets sample (`n` bajo).  
- No sustituye benchmark productivo de volumen real.  
- Para decisiones de negocio, conviene repetir con un dataset ampliado por experimento y series temporales.
