# ⚽ Predicciones Champions League 2026-27 — Juego amistoso entre amigos

Sistema en Python que recopila el histórico de fútbol de clubes europeo, entrena
un modelo estadístico y genera recomendaciones de apuestas **para un juego
amistoso entre amigos** (no apuestas reales con dinero — el enfoque es diversión
y presumir aciertos, no asesoría financiera).

Está construido como sistema **multi-competición**: la competición sobre la que
se generan los picks se elige en `config.yaml` (o en el selector de la web). Hoy
apunta a la Champions; cambiar a LaLiga, la Premier o la Serie A es una línea.

El proyecto se construye por fases:

| Fase | Qué hace | Estado |
|------|----------|--------|
| **1 — Datos** | Ingesta de 21 competiciones desde ESPN a SQLite; actualización diaria y marcadores en vivo | ✅ Listo |
| **2 — Modelo** | Dixon-Coles jerárquico por liga + backtesting validado | ✅ Listo |
| **3 — Recomendación** | Las 3 "Leyes": Segura / Arriesgada / Soñador | ✅ Listo |
| **4 — Interfaz** | CLI `predicciones.py` + web Streamlit + histórico de aciertos | ✅ Listo |

---

## Instalación

Requiere **Python 3.9+**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> El sistema **funciona sin API keys**. Todo sale del API público de ESPN
> (`site.api.espn.com`), que da el histórico, el calendario y los marcadores en
> vivo de las mismas fuentes. Las claves de `.env` son de un módulo opcional que
> hoy no está conectado al pipeline.

---

## Fase 1 — Datos

### Descargar el histórico (una vez, al inicio)

```bash
python -m src.fase1_datos.descargar_clubes
# o limitando el histórico:
python -m src.fase1_datos.descargar_clubes --desde 2018-07-01
```

Tarda 10-20 minutos la primera vez y crea `data/clubes.db` con **~45.600
partidos desde 2018** de 21 competiciones: las 15 ligas domésticas de los clubes
de Champions, más Champions, Europa League, Conference y sus tres rondas previas.

Las copas europeas son la pieza clave: son los únicos partidos que comparan
ligas distintas entre sí, y sin ellas las fuerzas de un equipo español y uno
noruego no serían comparables.

### Actualizar cada día de jornada

```bash
python -m src.fase1_datos.actualizar_diario                 # datos de hoy
python -m src.fase1_datos.actualizar_diario --fecha 2026-09-30
python -m src.fase1_datos.actualizar_diario --sin-descarga  # solo ver el calendario
```

Refresca la temporada en curso (UPSERT, sin duplicar), aplica los marcadores en
vivo/finales y rellena los goles de cada partido terminado.

### Hueco conocido de cobertura

ESPN no publica las ligas de **Chequia, Ucrania, Eslovaquia ni Azerbaiyán**. Eso
afecta a 4 de los 36 clubes de esta Champions —**Slavia Prague, Shakhtar
Donetsk, Slovan Bratislava y Sabah FK**—, que solo aportan sus partidos europeos
(22 a 102 partidos cada uno, frente a los 300-400 de un club de liga cubierta).
El sistema no lo esconde: esos clubes se agrupan en la pseudo-liga `otras` y sus
predicciones salen marcadas con **confianza baja o media**.

---

## Fase 2 — Modelo

Modelo de **Dixon-Coles** (Poisson bivariado con corrección de marcadores bajos)
con una **capa jerárquica por liga**, entrenado por máxima verosimilitud
ponderada con gradiente analítico.

```bash
# Ver el ranking de fuerzas y los efectos de liga que aprende el modelo
python -m src.fase2_modelo.entrenar

# Validar contra competiciones pasadas (sin trampa: entrena solo con el pasado)
python -m src.backtesting.backtest
python -m src.backtesting.backtest --rapido      # solo eventos de Champions
python -m src.backtesting.backtest --tune        # + búsqueda de hiperparámetros
```

### Por qué la capa jerárquica

Un Dixon-Coles plano estima una fuerza por equipo y la encoge hacia la media
**global**. Con 15 ligas de nivel muy distinto, eso sobrevalora a los equipos de
ligas débiles (que ganan casi todos sus partidos domésticos) y castiga a los de
ligas fuertes. Aquí:

```
ataque[i]  = A[liga(i)] + a[i]
defensa[i] = D[liga(i)] + d[i]
```

Cada equipo se encoge hacia la media de **su** liga, y la diferencia de nivel
entre ligas la estiman los partidos europeos. La ventaja de local también es por
liga (varía mucho entre países y cayó tras 2020).

El orden de ligas que aprende el modelo por su cuenta —Premier, LaLiga,
Bundesliga, Ligue 1, Serie A arriba; Eliteserien, Superliga griega y Premiership
escocesa abajo— es la señal de que la capa funciona.

### Qué tan bien predice (backtest, entrenando solo con el pasado)

845 partidos: 6 ventanas de Champions (fase de grupos/liga y eliminatorias de
2023-24 a 2025-26) más la Premier 2025-26 completa. La línea base son las
frecuencias históricas de local/empate/visitante.

| Métrica | Modelo | Línea base | Mejora |
|---|---|---|---|
| Accuracy 1X2 | 54.8% | — | — |
| RPS (menor = mejor) | 0.2008 | 0.2336 | −14.0% |
| Log-loss | 0.9546 | 1.0518 | −9.2% |

Solo Champions (503 partidos): accuracy **59.0%**, RPS **0.1951**. La fase liga
se predice bastante mejor que las eliminatorias (61-63% vs 44-53% de accuracy),
que es lo esperable: una eliminatoria es un partido de altísima varianza entre
dos equipos parecidos.

**La capa jerárquica se paga sola.** Aplastando los efectos de liga el modelo
colapsa al Dixon-Coles plano, y ahí se ve la diferencia:

| Sobre Champions | Accuracy | Log-loss | RPS |
|---|---|---|---|
| Jerárquico | 59.0% | 0.9067 | 0.1951 |
| Plano | 58.3% | 0.9237 | 0.1997 |

Los hiperparámetros (`vida_media_anios: 1.0`, `reg: 1.0`, `reg_liga: 0.01`)
salen de `backtest --tune`. El óptimo es una **meseta ancha**: entre 0.75 y 1.5
de vida media y entre 0.5 y 2.0 de regularización el log-loss se mueve en la
tercera decimal, así que el modelo no depende de haber acertado un valor mágico.

De una sola matriz de marcadores coherente se derivan todos los mercados: 1X2,
doble oportunidad, Over/Under (1.5/2.5/3.5), BTTS, marcador exacto (top-3),
primer equipo en anotar y hándicap asiático. Cada predicción lleva su nivel de
**confianza** (alto/medio/bajo) según los datos disponibles.

### Eliminatorias a doble partido

Desde febrero lo que importa no es un partido sino **quién pasa**.
`src/fase2_modelo/eliminatoria.py` convoluciona las matrices de ida y vuelta para
sacar la distribución del global, con las reglas UEFA vigentes: sin goles fuera
de casa (abolidos en 2021), prórroga modelada a 30/90 del ritmo del partido y
penales al 50% (no hay señal fiable para dar ventaja a nadie en una tanda).

### Simulación de la fase liga

`src/fase2_modelo/simulacion.py` simula 10.000 veces los 144 partidos de la fase
liga (los jugados con su resultado real) y devuelve, por club, la probabilidad de
acabar en **top-8** (octavos directos), **9-24** (playoff) o **eliminado**.

---

## Fases 3 y 4 — Recomendaciones diarias (uso principal)

```bash
# Reporte completo con las 3 Leyes para una fecha
python predicciones.py --fecha 2026-09-30

# Bajar datos frescos antes de predecir
python predicciones.py --fecha 2026-09-30 --actualizar

# Añadir la tabla detallada de todos los mercados por partido
python predicciones.py --fecha 2026-09-30 --detalle

# Probabilidades de clasificación de las eliminatorias del día
python predicciones.py --fecha 2027-02-17 --eliminatorias
```

Genera tres listas claramente diferenciadas:
- 🔒 **Ley Segura** — alta probabilidad (≥68%) y confianza alta/media.
- ⚖️ **Ley Arriesgada** — probabilidad media (45–65%), mejor pago.
- 🚀 **Ley Soñador** — marcadores exactos + una combinada (parlay) del día.

Cada apuesta trae probabilidad, "pago" potencial (×), nivel de confianza y una
**justificación con datos** (goles esperados, forma reciente, head-to-head).

### Interfaz web (Streamlit)

```bash
streamlit run src/fase4_interfaz/app_streamlit.py
# luego abre http://localhost:8501 en el navegador
```

Cuatro pestañas:
- **🎯 Picks del día** — 6 parlays desplegables (cortos de 3 patas y largos con
  todos los juegos), con marcador en vivo por pata.
- **📊 Detalle por partido** — tarjeta por encuentro con barra 1X2 visual, goles
  esperados, marcadores probables y primer goleador.
- **🏁 Clasificación proyectada** — Monte Carlo de la fase liga; en febrero pasa
  automáticamente a mostrar las probabilidades de eliminatoria.
- **🏆 Histórico de aciertos** — métricas y gráfico de cómo acertó cada Ley.

En la barra lateral eliges la **competición** y la fecha.

### Histórico de aciertos (versión consola)

```bash
python -m src.fase4_interfaz.seguimiento
```

Evalúa las 3 Leyes contra las jornadas ya disputadas, entrenando cada día solo
con datos previos. Sobre la fase liga 2025-26 (13 jornadas, entrenando cada día
solo con el pasado):

| Ley | Aciertos reales | Probabilidad declarada |
|---|---|---|
| 🔒 Segura | **83%** (85/102) | ≥68% → bien calibrado, incluso conservador |
| ⚖️ Arriesgada | **64%** (69/107) | 45–65% → en el borde alto de la banda |
| 🚀 Soñador (marcador exacto) | 7% (8/108) | ~10% → coherente |

Las combinadas de 3 patas pegaron 7 de 13 días (Segura) y 4 de 13 (Arriesgada).

---

## Cambiar de competición

En `config.yaml`:

```yaml
competicion:
  id: "esp.1"                     # cualquier slug de datos.competiciones
  nombre: "LaLiga 2026-27"
  temporada: 2026
```

O directamente en el selector de la barra lateral de la web. Las 21
competiciones ya descargadas siguen alimentando el modelo pase lo que pase; lo
único que cambia es sobre cuáles se generan picks.

---

## Estructura del proyecto

```
worldcup-predictions/
├── config.yaml              # competición activa, pesos, hiperparámetros
├── requirements.txt
├── predicciones.py          # entrypoint de la CLI
├── data/
│   ├── raw/espn/            # caché de respuestas de ESPN ya normalizadas
│   ├── processed/modelos/   # modelos entrenados cacheados
│   └── clubes.db            # base SQLite central
└── src/
    ├── config.py            # competición activa y catálogo de ligas
    ├── fase1_datos/         # cliente ESPN + ingesta + BD + marcadores en vivo
    ├── fase2_modelo/        # Dixon-Coles jerárquico, mercados, eliminatorias, simulación
    ├── fase3_recomendacion/ # las 3 Leyes
    ├── fase4_interfaz/      # CLI + web + seguimiento
    └── backtesting/         # validación contra competiciones pasadas
```

---

## Notas de precisión y honestidad

- **Mercados de goles** (1X2, doble oportunidad, Over/Under, BTTS, marcador
  exacto): datos abundantes → predicciones sólidas vía Dixon-Coles.
- **Tarjetas, tiros, corners**: no incluidos. El feed usado no los trae de forma
  consistente en las 21 competiciones, y un modelo con datos parciales sería de
  confianza baja y sin valor real.
- Las combinadas asumen **independencia entre patas** y la cuota es `1/p`, sin
  margen de casa. Es deliberado: sirve para comparar picks entre sí, no para
  compararse con una casa real, que siempre pagará menos.
- El 1X2 se resuelve **a 90 minutos**, como en cualquier casa. Una eliminatoria
  que se decide en la prórroga cuenta como empate en ese mercado; para saber
  quién pasa está la pestaña de eliminatorias.
- Cada predicción viene con su **probabilidad (%)** y un **nivel de confianza**
  según cuántos datos históricos tenga cada equipo.
