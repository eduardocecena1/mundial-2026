"""
app_streamlit.py — FASE 4: interfaz web visual.

Muestra las predicciones del día con tarjetas visuales por partido y las 3 Leyes
(Segura / Arriesgada / Soñador) bien diferenciadas por color, más el histórico de
aciertos del modelo en el Mundial 2026.

Ejecutar (desde la raíz del proyecto):
    streamlit run src/fase4_interfaz/app_streamlit.py
o con el atajo:
    python -m streamlit run src/fase4_interfaz/app_streamlit.py

Es para un juego amistoso entre amigos — no es asesoría de apuestas reales.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Permitir importar el paquete 'src' al ejecutar con `streamlit run`
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.fase1_datos import db
from src.fase2_modelo.entrenar import cargar_config, entrenar_modelo
from src.fase2_modelo.predecir_partido import predecir
from src.fase3_recomendacion.generar_leyes import generar
from src.fase4_interfaz.seguimiento import evaluar_rango


# --- Configuración de página -----------------------------------------------

st.set_page_config(page_title="Predicciones Mundial 2026", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
  .bloque { border-radius: 14px; padding: 16px 18px; margin-bottom: 14px;
            border: 1px solid rgba(255,255,255,.08); }
  .pick   { background: rgba(255,255,255,.04); border-radius: 10px;
            padding: 10px 14px; margin: 8px 0; }
  .pick .top { display:flex; justify-content:space-between; align-items:baseline; }
  .pick .ap  { font-size: 1.05rem; font-weight: 600; }
  .pick .pr  { font-size: 1.35rem; font-weight: 800; }
  .pick .mt  { font-size: .80rem; opacity: .70; margin-top: 4px; }
  .pago   { font-size:.85rem; opacity:.8; }
  .barra  { display:flex; height: 26px; border-radius: 7px; overflow:hidden;
            font-size:.72rem; font-weight:700; color:#0b0b0b; }
  .seg-l  { background:#4ade80; display:flex; align-items:center; justify-content:center;}
  .seg-e  { background:#fbbf24; display:flex; align-items:center; justify-content:center;}
  .seg-v  { background:#60a5fa; display:flex; align-items:center; justify-content:center;}
  .chip   { display:inline-block; padding:2px 9px; border-radius:999px;
            font-size:.72rem; font-weight:700; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

COLOR_CONF = {"alto": "#22c55e", "medio": "#f59e0b", "bajo": "#ef4444"}
EMOJI_CONF = {"alto": "🟢", "medio": "🟡", "bajo": "🔴"}


# --- Carga cacheada ---------------------------------------------------------

@st.cache_data(ttl=60 * 60 * 12, show_spinner="Actualizando datos del Mundial 2026...")
def asegurar_datos(dia: str) -> str:
    """Garantiza que la base existe y está FRESCA para el día dado.

    Cacheada con la fecha como clave (y TTL de 12 h): así, al abrir la web cada
    día, descarga automáticamente los resultados nuevos y los partidos del día —
    sin que tengas que tocar nada. En la nube reconstruye la base desde la fuente
    pública (que se actualiza sola a diario). Devuelve la fecha de última
    actualización, que sirve de 'versión' para refrescar las demás cachés.
    """
    con = db.conectar()
    db.inicializar(con)
    n = con.execute("SELECT COUNT(*) AS c FROM partidos").fetchone()["c"]
    ult = db.get_meta(con, "ultima_actualizacion")
    if n == 0 or ult != dia:
        from src.fase1_datos.actualizar_diario import actualizar
        actualizar(con)
        ult = db.get_meta(con, "ultima_actualizacion")
    con.close()
    return ult or dia


@st.cache_resource(show_spinner="Entrenando el modelo con el histórico...")
def cargar_modelo_y_cfg(version: str):
    # 'version' (fecha de los datos) es la clave de caché: el modelo se reentrena
    # automáticamente cuando entran datos nuevos.
    con = db.conectar()
    cfg = cargar_config()
    modelo = entrenar_modelo(con, cfg)
    con.close()
    return modelo, cfg


@st.cache_data(show_spinner="Calculando histórico de aciertos...")
def cargar_historico(version: str):
    con = db.conectar()
    cfg = cargar_config()
    datos = evaluar_rango(con, cfg)
    con.close()
    return datos


def fechas_mundial():
    con = db.conectar()
    filas = con.execute(
        "SELECT DISTINCT fecha FROM partidos WHERE es_mundial2026=1 ORDER BY fecha"
    ).fetchall()
    con.close()
    return [f["fecha"] for f in filas]


# --- Componentes visuales ---------------------------------------------------

def chip_confianza(nivel: str) -> str:
    c = COLOR_CONF.get(nivel, "#888")
    return (f"<span class='chip' style='background:{c}22;color:{c};"
            f"border:1px solid {c}55'>confianza {nivel}</span>")


def render_pick(rec: dict):
    chip = chip_confianza(rec["confianza"])
    st.markdown(
        f"""<div class='pick'>
              <div class='top'>
                <span class='ap'>{rec['partido']} — {rec['apuesta']}</span>
                <span class='pr'>{100*rec['prob']:.0f}%</span>
              </div>
              <div class='pago'>pago potencial ×{rec['pago']} &nbsp; {chip}</div>
              <div class='mt'>↳ {rec['motivo']}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def barra_1x2(loc, vis, p):
    l, e, v = p["local"], p["empate"], p["visitante"]
    st.markdown(
        f"""<div class='barra'>
              <div class='seg-l' style='width:{l*100:.1f}%'>{l*100:.0f}%</div>
              <div class='seg-e' style='width:{e*100:.1f}%'>{e*100:.0f}%</div>
              <div class='seg-v' style='width:{v*100:.1f}%'>{v*100:.0f}%</div>
            </div>
            <div style='display:flex;justify-content:space-between;font-size:.78rem;
                        opacity:.8;margin-top:3px'>
              <span>🟩 {loc}</span><span>🟨 Empate</span><span>🟦 {vis}</span>
            </div>""",
        unsafe_allow_html=True,
    )


def render_tarjeta_partido(con, modelo, cfg, row):
    loc, vis = row["local"], row["visitante"]
    pred = predecir(con, modelo, cfg, loc, vis, row["neutral"])
    conf = pred["confianza"]
    sede = "campo neutral" if pred["neutral"] else "con localía"
    with st.container(border=True):
        st.markdown(f"#### {loc} 🆚 {vis}")
        st.caption(f"{row['ciudad']} · {sede} · "
                   f"{EMOJI_CONF[conf['nivel']]} confianza {conf['nivel']} "
                   f"({loc} {conf['n_local']} part., {vis} {conf['n_visit']} part.)")
        barra_1x2(loc, vis, pred["1x2"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Goles esperados",
                  f"{pred['lambda_local']:.1f} – {pred['lambda_visit']:.1f}")
        c2.metric("Over 2.5", f"{100*pred['over_under']['over_2.5']:.0f}%")
        c3.metric("Ambos anotan", f"{100*pred['btts']['si']:.0f}%")

        marc = " · ".join(f"**{i}-{j}** ({100*pr:.0f}%)"
                          for i, j, pr in pred["marcadores_top3"])
        st.markdown(f"🎯 Marcadores más probables: {marc}")
        fa = pred["primer_en_anotar"]
        st.markdown(f"⚡ Primer gol: {loc} {100*fa['local']:.0f}% · "
                    f"{vis} {100*fa['visitante']:.0f}% · "
                    f"sin goles {100*fa['sin_goles']:.0f}%")


def render_historico(version: str):
    """Calcula y dibuja el histórico de aciertos (entrena el modelo por jornada)."""
    hist = cargar_historico(version)
    if not hist["fechas"]:
        st.info("Aún no hay jornadas jugadas para evaluar.")
        return
    tot = hist["totales"]
    etiquetas = {"segura": "🔒 Segura", "arriesgada": "⚖️ Arriesgada",
                 "sonador": "🚀 Soñador"}
    cols = st.columns(3)
    for col, tier in zip(cols, ("segura", "arriesgada", "sonador")):
        a, t = tot[tier]
        pct = (100 * a / t) if t else 0
        col.metric(etiquetas[tier], f"{pct:.0f}% aciertos", f"{a}/{t}")

    filas = []
    for f in hist["fechas"]:
        for tier in ("segura", "arriesgada", "sonador"):
            a, t = f[tier]
            if t:
                filas.append({"fecha": f["fecha"], "Ley": etiquetas[tier],
                              "acierto_%": 100 * a / t})
    df = pd.DataFrame(filas)
    if not df.empty:
        chart = (alt.Chart(df).mark_line(point=True)
                 .encode(x="fecha:N", y=alt.Y("acierto_%:Q",
                                              scale=alt.Scale(domain=[0, 100])),
                         color="Ley:N", tooltip=["fecha", "Ley", "acierto_%"])
                 .properties(height=320))
        st.altair_chart(chart, use_container_width=True)


# --- App --------------------------------------------------------------------

def main():
    st.title("⚽ Predicciones Mundial 2026")
    st.caption("Juego amistoso entre amigos · diversión y presumir aciertos · "
               "**no es asesoría de apuestas reales con dinero**")

    # Frescura automática diaria: descarga datos nuevos una vez al día al abrir.
    hoy = date.today().isoformat()
    version = asegurar_datos(hoy)

    fechas = fechas_mundial()
    if not fechas:
        st.error("No hay datos. Corre primero: "
                 "`python -m src.fase1_datos.descargar_historico`")
        return

    # Sidebar
    st.sidebar.header("Jornada")
    idx = fechas.index(hoy) if hoy in fechas else 0
    fecha = st.sidebar.selectbox("Elige la fecha", fechas, index=idx)
    st.sidebar.caption(f"📅 Datos actualizados al {version}")
    if st.sidebar.button("🔄 Forzar actualización ahora"):
        from src.fase1_datos.actualizar_diario import actualizar
        con = db.conectar()
        with st.spinner("Descargando datos frescos..."):
            actualizar(con)
        con.close()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Leyenda**\n\n🔒 Segura · alta prob.\n\n"
        "⚖️ Arriesgada · prob. media\n\n🚀 Soñador · alto riesgo")

    modelo, cfg = cargar_modelo_y_cfg(version)
    con = db.conectar()
    partidos = db.calendario_de_fecha(con, fecha)

    tab1, tab2, tab3 = st.tabs(["🎯 Picks del día", "📊 Detalle por partido",
                                "🏆 Histórico de aciertos"])

    # --- TAB 1: las 3 Leyes ---
    with tab1:
        if not partidos:
            st.info("No hay partidos del Mundial 2026 en esta fecha.")
        else:
            leyes = generar(con, modelo, cfg, fecha)
            st.subheader(f"Partidos del {fecha} · {len(partidos)} encuentros")

            st.markdown("<div class='bloque' style='background:rgba(34,197,94,.10)'>",
                        unsafe_allow_html=True)
            st.markdown("### 🔒 Ley Segura  ·  alta probabilidad, bajo riesgo")
            if leyes["segura"]:
                for r in leyes["segura"]:
                    render_pick(r)
            else:
                st.caption("Hoy ningún partido alcanza el umbral de seguridad.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bloque' style='background:rgba(245,158,11,.10)'>",
                        unsafe_allow_html=True)
            st.markdown("### ⚖️ Ley Arriesgada  ·  probabilidad media, mejor pago")
            if leyes["arriesgada"]:
                for r in leyes["arriesgada"]:
                    render_pick(r)
            else:
                st.caption("Sin candidatos en la banda media hoy.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bloque' style='background:rgba(239,68,68,.10)'>",
                        unsafe_allow_html=True)
            st.markdown("### 🚀 Ley Soñador  ·  baja probabilidad, alto valor")
            for r in leyes["sonador"]:
                render_pick(r)
            if leyes["parlay"]:
                p = leyes["parlay"]
                lista = "<br>".join(f"➕ {a}" for a in p["apuestas"])
                st.markdown(
                    f"""<div class='pick'>
                          <div class='ap'>🎟️ Combinada soñadora del día</div>
                          <div class='mt'>{lista}</div>
                          <div class='pago' style='margin-top:6px'>
                            probabilidad {100*p['prob']:.0f}% · pago ×{p['pago']}</div>
                        </div>""",
                    unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 2: detalle por partido ---
    with tab2:
        if not partidos:
            st.info("No hay partidos del Mundial 2026 en esta fecha.")
        else:
            cols = st.columns(2)
            for k, row in enumerate(partidos):
                with cols[k % 2]:
                    render_tarjeta_partido(con, modelo, cfg, row)

    # --- TAB 3: histórico de aciertos (bajo demanda, para no frenar la carga) ---
    with tab3:
        st.subheader("¿Qué tan bien ha acertado el modelo en este Mundial?")
        st.caption("Cada jornada se predice entrenando SOLO con datos previos "
                   "(sin trampa).")
        if st.session_state.get("calc_hist"):
            render_historico(version)
        elif st.button("📊 Calcular histórico de aciertos"):
            st.session_state["calc_hist"] = True
            st.rerun()
        else:
            st.info("Pulsa el botón para ver cómo ha acertado cada Ley en las "
                    "jornadas ya jugadas. (Tarda unos segundos: entrena el modelo "
                    "en cada fecha pasada.)")

    con.close()


main()
