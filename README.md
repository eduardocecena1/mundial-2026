# ⚽ Predicciones Mundial 2026 — Juego amistoso entre amigos

Sistema en Python que recopila datos históricos de selecciones, entrena un modelo
estadístico de predicción y genera recomendaciones de apuestas **para un juego
amistoso entre amigos** (no apuestas reales con dinero — el enfoque es diversión y
presumir aciertos, no asesoría financiera).

El proyecto se construye por fases:

| Fase | Qué hace | Estado |
|------|----------|--------|
| **1 — Datos** | Descarga histórico + calendario 2026 a SQLite; actualización diaria | ✅ Listo |
| **2 — Modelo** | Dixon-Coles (goles) + backtesting validado | ✅ Núcleo listo |
| **3 — Recomendación** | Las 3 "Leyes": Segura / Arriesgada / Soñador | ✅ Listo |
| **4 — Interfaz** | CLI `predicciones.py` + web Streamlit + histórico de aciertos | ✅ Listo |

---

## Instalación

Requiere **Python 3.9+**.

```bash
# 1. Crear entorno virtual e instalar dependencias
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (Opcional, solo para el modelo de tarjetas/tiros con XGBoost en macOS)
brew install libomp
```

> El sistema **funciona sin API keys**. Usa el dataset público y gratuito de
> resultados internacionales de [martj42](https://github.com/martj42/international_results)
> (la misma base que el dataset de Kaggle "International football results", pero
> actualizada a diario e incluyendo ya los partidos del Mundial 2026).
> Las claves opcionales de `.env` (API-Football, football-data.org) solo enriquecen
> estadísticas detalladas; ver `.env.example`.

---

## Fase 1 — Datos (lista de uso)

### Descargar el histórico (una vez, al inicio)

```bash
python -m src.fase1_datos.descargar_historico
# o limitando el histórico:
python -m src.fase1_datos.descargar_historico --desde 2014-01-01
```

Esto crea `data/worldcup.db` (SQLite) con:
- ~11.900 partidos internacionales desde 2014 (jugados y futuros),
- goles individuales (para "primer equipo en anotar"),
- tandas de penales,
- las **48 selecciones del Mundial 2026 y su calendario completo**, derivados de los
  propios datos (no escritos a mano).

### Actualizar cada día del Mundial

```bash
python -m src.fase1_datos.actualizar_diario                 # datos de hoy
python -m src.fase1_datos.actualizar_diario --fecha 2026-06-25
python -m src.fase1_datos.actualizar_diario --sin-descarga  # solo ver el calendario
```

Re-descarga las fuentes, **rellena los marcadores** de los partidos ya jugados
(UPSERT, sin duplicar) y muestra los partidos del día.

---

## Fase 2 — Modelo (lista de uso)

Modelo de **Dixon-Coles** (Poisson bivariado con corrección de marcadores bajos),
entrenado por máxima verosimilitud ponderada (decaimiento temporal + peso por
torneo + campo neutral + regularización para equipos con pocos datos).

```bash
# Ver el ranking de fuerzas ataque/defensa que aprende el modelo
python -m src.fase2_modelo.entrenar

# Validar el modelo contra torneos pasados (sin trampa: entrena solo con el pasado)
python -m src.backtesting.backtest

# + buscar los mejores hiperparámetros
python -m src.backtesting.backtest --tune
```

De una sola matriz de marcadores coherente se derivan todos los mercados:
1X2, doble oportunidad, Over/Under (1.5/2.5/3.5), BTTS, marcador exacto (top-3),
primer equipo en anotar y hándicap asiático. Cada predicción lleva su nivel de
**confianza** (alto/medio/bajo) según los datos disponibles de cada equipo.

**Resultado del backtest** (230 partidos de Mundial 2018/2022 y Eurocopa 2021/2024,
entrenando solo con datos previos a cada torneo):

| Métrica | Modelo | Línea base | Mejora |
|---|---|---|---|
| Accuracy 1X2 | 56.5% | ~45% | — |
| RPS (menor = mejor) | 0.1995 | 0.2388 | −16% |
| Log-loss | 0.987 | 1.093 | −10% |

El fútbol de torneo es de alta varianza (sorpresas como Arabia 2-1 Argentina en
2022), así que estas cifras son honestas y competitivas con modelos de mercado.

---

## Fases 3 y 4 — Recomendaciones diarias (uso principal)

Este es el comando del día a día durante el Mundial:

```bash
# Reporte completo con las 3 Leyes para una fecha
python predicciones.py --fecha 2026-06-25

# Bajar datos frescos antes de predecir (resultados de ayer + partidos de hoy)
python predicciones.py --fecha 2026-06-25 --actualizar

# Añadir la tabla detallada de todos los mercados por partido
python predicciones.py --fecha 2026-06-25 --detalle
```

Genera tres listas claramente diferenciadas:
- 🔒 **Ley Segura** — alta probabilidad (≥68%) y confianza alta/media.
- ⚖️ **Ley Arriesgada** — probabilidad media (45–65%), mejor pago.
- 🚀 **Ley Soñador** — marcadores exactos + una combinada (parlay) del día.

Cada apuesta trae probabilidad, "pago" potencial (×), nivel de confianza y una
**justificación con datos** (goles esperados, forma reciente, head-to-head).

### Interfaz web (Streamlit)

La forma visual y fácil de ver los picks del día sin perderse en texto:

```bash
streamlit run src/fase4_interfaz/app_streamlit.py
# luego abre http://localhost:8501 en el navegador
```

Tiene tres pestañas:
- **🎯 Picks del día** — las 3 Leyes con tarjetas de colores (verde/ámbar/rojo),
  probabilidad grande, pago potencial, confianza y justificación.
- **📊 Detalle por partido** — tarjeta por encuentro con barra 1X2 visual, goles
  esperados, marcadores probables y primer goleador.
- **🏆 Histórico de aciertos** — métricas y gráfico de cómo acertó cada Ley.

En la barra lateral eliges la fecha y puedes pulsar **🔄 Actualizar datos**.

### Histórico de aciertos (versión consola)

```bash
python -m src.fase4_interfaz.seguimiento     # cómo le fue a cada Ley en lo ya jugado
```

Evalúa las 3 Leyes contra las jornadas ya disputadas del Mundial 2026 (entrenando
cada día solo con datos previos). Resultado sobre los primeros 48 partidos:

| Ley | Aciertos reales | Probabilidad declarada |
|---|---|---|
| 🔒 Segura | **85%** (41/48) | 68–89% → bien calibrado |
| ⚖️ Arriesgada | **52%** (24/46) | 45–65% → en el blanco |
| 🚀 Soñador (marcador exacto) | 8% (4/48) | ~10–17% → coherente |

---

## Estructura del proyecto

```
worldcup-predictions/
├── config.yaml              # parámetros del modelo y de las apuestas (editable)
├── requirements.txt
├── .env.example             # claves opcionales
├── data/
│   ├── raw/                 # CSVs descargados sin tocar
│   └── worldcup.db          # base SQLite central
└── src/
    ├── fase1_datos/         # ✅ descarga + actualización + esquema BD
    ├── fase2_modelo/        # 🔜 Dixon-Coles + stats + confianza
    ├── fase3_recomendacion/ # 🔜 las 3 Leyes
    ├── fase4_interfaz/      # 🔜 CLI + web
    └── backtesting/         # 🔜 validación contra Mundiales pasados
```

---

## Notas de precisión y honestidad

- **Mercados de goles** (1X2, doble oportunidad, Over/Under, BTTS, marcador exacto):
  datos abundantes → predicciones sólidas vía Dixon-Coles.
- **Tarjetas, tiros, corners**: no incluidos a propósito. El dataset gratuito no
  los tiene, y el plan gratis de API-Football no da la temporada 2026 (solo
  2022-2024). Un modelo con datos viejos sería de confianza baja y sin valor real,
  así que se descartó para no diluir la credibilidad del sistema. La infraestructura
  para API-Football queda en `src/fase1_datos/api_football.py` por si se consigue un
  plan con acceso a 2026.
- Cada predicción vendrá con su **probabilidad (%)** y un **nivel de confianza**
  (alto/medio/bajo) según cuántos datos históricos tenga ese equipo.
