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
from src.fase1_datos import marcadores_vivo as mv
from src.fase2_modelo.entrenar import cargar_config, entrenar_modelo
from src.fase2_modelo.predecir_partido import predecir
from src.fase3_recomendacion.generar_leyes import generar
from src.fase4_interfaz.seguimiento import (
    evaluar_rango, evaluar, _resultado_partido, _evaluar_parlay,
)


# --- Configuración de página -----------------------------------------------

st.set_page_config(page_title="Predicciones Mundial 2026", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
  .bloque { border-radius: 14px; padding: 16px 18px; margin-bottom: 14px;
            border: 1px solid rgba(255,255,255,.08); }
  .pick   { background: rgba(255,255,255,.04); border-radius: 10px;
            padding: 10px 14px; margin: 8px 0; border-left:4px solid transparent; }
  .pick.won  { border-left-color:#22c55e; background:rgba(34,197,94,.08); }
  .pick.lost { border-left-color:#ef4444; background:rgba(239,68,68,.07); }
  .pick.pend { border-left-color:#eab308; }
  .pick .top { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
  .pick .ap  { font-size: 1.05rem; font-weight: 600; }
  .pick .pr  { font-size: 1.35rem; font-weight: 800; }
  .pick .mt  { font-size: .80rem; opacity: .70; margin-top: 4px; }
  .res-badge { font-weight:800; font-size:.74rem; padding:3px 10px; border-radius:999px;
               white-space:nowrap; }
  .r-won { background:#22c55e22; color:#22c55e; border:1px solid #22c55e66; }
  .r-lost{ background:#ef444422; color:#ef4444; border:1px solid #ef444466; }
  .r-pend{ background:#eab30822; color:#eab308; border:1px solid #eab30866; }
  .score { font-size:.8rem; opacity:.85; font-weight:700; }
  /* Cabecera estilo casino */
  .casino-hd { background:linear-gradient(135deg,#0b3d20,#0e1117 70%);
               border:1px solid #22c55e44; border-radius:14px; padding:14px 18px;
               margin-bottom:10px; box-shadow:0 0 24px rgba(34,197,94,.12) inset; }
  .casino-hd .t { font-size:1.5rem; font-weight:900; letter-spacing:.5px; }
  .neon { color:#22c55e; text-shadow:0 0 8px rgba(34,197,94,.6); }
  .gold { color:#f5c542; text-shadow:0 0 8px rgba(245,197,66,.45); }
  .multi { margin:10px 2px 2px; padding:10px 14px; border-radius:10px;
           background:rgba(245,197,66,.10); border:1px solid rgba(245,197,66,.35);
           display:flex; justify-content:space-between; align-items:center;
           font-size:.92rem; }
  .multi .x { font-size:1.5rem; font-weight:900; }
  .pago   { font-size:.85rem; opacity:.8; }
  .barra  { display:flex; height: 26px; border-radius: 7px; overflow:hidden;
            font-size:.72rem; font-weight:700; color:#0b0b0b; }
  .seg-l  { background:#4ade80; display:flex; align-items:center; justify-content:center;}
  .seg-e  { background:#fbbf24; display:flex; align-items:center; justify-content:center;}
  .seg-v  { background:#60a5fa; display:flex; align-items:center; justify-content:center;}
  .chip   { display:inline-block; padding:2px 9px; border-radius:999px;
            font-size:.72rem; font-weight:700; }
  /* Boletos estilo casa de apuestas */
  .ticket { border-radius:12px; padding:12px 16px; margin:10px 0;
            border:1px solid rgba(255,255,255,.10);
            background:repeating-linear-gradient(45deg,rgba(255,255,255,.02),
                       rgba(255,255,255,.02) 12px,transparent 12px,transparent 24px); }
  .ticket.won  { border-left:6px solid #22c55e; }
  .ticket.lost { border-left:6px solid #ef4444; opacity:.85; }
  .ticket.void { border-left:6px solid #888; opacity:.7; }
  .tk-head { display:flex; justify-content:space-between; align-items:center;
             margin-bottom:6px; }
  .tk-fecha{ font-weight:700; font-size:.95rem; }
  .tk-badge{ font-weight:800; font-size:.82rem; padding:3px 12px; border-radius:999px; }
  .b-won { background:#22c55e22; color:#22c55e; border:1px solid #22c55e66; }
  .b-lost{ background:#ef444422; color:#ef4444; border:1px solid #ef444466; }
  .b-void{ background:#88888822; color:#aaa; border:1px solid #88888866; }
  .tk-leg { font-size:.86rem; padding:2px 0; }
  .tk-foot{ margin-top:7px; font-size:.85rem; opacity:.9;
            display:flex; justify-content:space-between; }
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


@st.cache_data(ttl=90, show_spinner=False)
def marcadores_live(fecha: str, version: str) -> dict:
    """Aplica marcadores en vivo/finales de ESPN para la fecha (cada ~90 s).
    Rellena los partidos terminados y devuelve los que están en curso."""
    con = db.conectar()
    res = mv.aplicar(con, fecha)
    con.close()
    return res


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


def _hora_orden(live: dict, local: str, visit: str) -> str:
    """Clave para ordenar partidos por horario (ISO UTC; sin hora van al final)."""
    h = (live or {}).get("horarios", {}).get((local, visit))
    return h["utc"] if h and h.get("utc") else "9999"


def _hora_mx(live: dict, local: str, visit: str):
    """Hora del partido 'HH:MM' en hora centro de México, o None si no se conoce."""
    h = (live or {}).get("horarios", {}).get((local, visit))
    return h["mx"] if h and h.get("mx") else None


def render_multiplicador(picks: list):
    """Pie de cada parlay: multiplicador combinado (producto de todas las cuotas)
    y probabilidad combinada, como si se jugaran todos los picks en un solo boleto."""
    if len(picks) < 2:
        return
    cuota = prob = 1.0
    for r in picks:
        cuota *= r["pago"]
        prob *= r["prob"]
    cuota_txt = f"×{cuota:,.0f}" if cuota >= 1000 else f"×{cuota:.2f}"
    prob_txt = f"{100*prob:.1f}%" if prob >= 0.001 else f"{100*prob:.4f}%"
    st.markdown(
        f"<div class='multi'><span>🧮 Si juegas las <b>{len(picks)}</b> juntas "
        f"(probabilidad combinada {prob_txt})</span>"
        f"<span class='x gold'>{cuota_txt}</span></div>",
        unsafe_allow_html=True)


def render_pick(rec: dict, con, fecha: str, live: dict = None):
    """Dibuja un pick como ficha de apuesta, mostrando AUTOMÁTICAMENTE si pegó o
    no (con el marcador real) en cuanto el partido se juega, o el marcador EN VIVO
    si está en curso."""
    chip = chip_confianza(rec["confianza"])
    hora = _hora_mx(live, rec["local"], rec["visitante"])
    hora_html = f"🕐 {hora} MX · " if hora else ""
    res = evaluar(con, rec, fecha)
    score = _resultado_partido(con, rec["local"], rec["visitante"], fecha)
    vivo = (live or {}).get("vivo", {}).get((rec["local"], rec["visitante"]))
    if res is True:
        clase = "won"; badge = "<span class='res-badge r-won'>PEGÓ ✅</span>"
        score_txt = f"<span class='score'>· marcador {score[0]}-{score[1]}</span>"
    elif res is False:
        clase = "lost"; badge = "<span class='res-badge r-lost'>NO PEGÓ ❌</span>"
        score_txt = f"<span class='score'>· marcador {score[0]}-{score[1]}</span>"
    elif vivo:
        clase = "pend"
        badge = f"<span class='res-badge r-pend'>🔴 EN VIVO {vivo['gl']}-{vivo['gv']}</span>"
        score_txt = f"<span class='score'>· {vivo['estado']}</span>"
    else:
        clase = "pend"; badge = "<span class='res-badge r-pend'>EN JUEGO ⏳</span>"
        score_txt = "<span class='score'>· aún no se juega</span>"
    st.markdown(
        f"""<div class='pick {clase}'>
              <div class='top'>
                <span class='ap'>{rec['partido']} — {rec['apuesta']}</span>
                <span class='pr'>{100*rec['prob']:.0f}%</span>
              </div>
              <div class='pago'>{hora_html}pago ×{rec['pago']} &nbsp; {chip} &nbsp; {badge} {score_txt}</div>
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


def render_tarjeta_partido(con, modelo, cfg, row, live: dict = None):
    loc, vis = row["local"], row["visitante"]
    pred = predecir(con, modelo, cfg, loc, vis, row["neutral"])
    conf = pred["confianza"]
    sede = "campo neutral" if pred["neutral"] else "con localía"
    hora = _hora_mx(live, loc, vis)
    hora_txt = f"🕐 {hora} hrs MX · " if hora else ""
    with st.container(border=True):
        st.markdown(f"#### {loc} 🆚 {vis}")
        st.caption(f"{hora_txt}{row['ciudad']} · {sede} · "
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

    render_parlays(hist)


def render_ticket(p: dict, mostrar_fecha: bool = True):
    """Dibuja UN boleto de combinada estilo casino, con su resultado en vivo:
    GANADO / PERDIDO / EN JUEGO / ANULADO, y cada pata con ✅/❌/⏳."""
    legs = p["legs"]
    if p["acerto"] is True:
        clase, bcls, badge, extra = "won", "b-won", "GANADO ✅", f"+{p['pago']-1:.2f} fichas"
    elif p["acerto"] is False:
        clase, bcls, badge, extra = "lost", "b-lost", "PERDIDO ❌", "-1.00 fichas"
    else:
        clase, bcls, badge, extra = "void", "b-void", "EN JUEGO ⏳", f"cuota ×{p['pago']}"
    legs_html = ""
    for leg in legs:
        marca = "✅" if leg["ok"] is True else ("❌" if leg["ok"] is False else "⏳")
        legs_html += f"<div class='tk-leg'>{marca} {leg['texto']}</div>"
    titulo = (f"📅 {p['fecha']}" if mostrar_fecha else "🎟️ Boleto del día")
    st.markdown(
        f"""<div class='ticket {clase}'>
              <div class='tk-head'>
                <span class='tk-fecha'>{titulo}</span>
                <span class='tk-badge {bcls}'>{badge}</span>
              </div>
              {legs_html}
              <div class='tk-foot'>
                <span>cuota combinada ×{p['pago']}</span><span>{extra}</span>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_parlays(hist: dict):
    """Historial de combinadas (parlays) estilo casa de apuestas: boletos con
    sus patas, cuota y marcador GANADO/PERDIDO + balance de fichas."""
    parlays = hist.get("parlays", [])
    if not parlays:
        return
    st.markdown("---")
    st.markdown("### 🎟️ Historial de combinadas (la del día con los picks más seguros)")

    res = hist.get("parlay_totales", {})
    jug, gan = res.get("jugados", 0), res.get("ganados", 0)
    bal = res.get("balance", 0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Combinadas pegadas", f"{gan}/{jug}",
              f"{(100*gan/jug):.0f}%" if jug else "s/d")
    c2.metric("Balance (1 ficha por día)", f"{bal:+.2f} fichas",
              "vas ganando" if bal > 0 else ("vas perdiendo" if bal < 0 else "en cero"),
              delta_color="normal" if bal >= 0 else "inverse")
    c3.metric("Días jugados", f"{jug}")

    st.caption("Cada día se arma un boleto con los picks más seguros. Gana solo si "
               "**todas** las patas pegan, como una combinada real. Pícale a un "
               "boleto para ver sus patas.")

    # Boletos compactos: cada uno se despliega al hacer clic (el más reciente abierto)
    for i, p in enumerate(reversed(parlays)):
        ok = p["acerto"]
        estado = ("✅ GANADO" if ok is True else "❌ PERDIDO" if ok is False
                  else "⏳ EN JUEGO")
        with st.expander(f"📅 {p['fecha']}   ·   {estado}   ·   cuota ×{p['pago']}",
                         expanded=(i == 0)):
            legs_html = ""
            for leg in p["legs"]:
                marca = "✅" if leg["ok"] is True else ("❌" if leg["ok"] is False else "⏳")
                legs_html += f"<div class='tk-leg'>{marca} {leg['texto']}</div>"
            extra = (f"+{p['pago']-1:.2f} fichas" if ok is True else
                     "-1.00 fichas" if ok is False else "sin cerrar todavía")
            st.markdown(
                f"<div class='ticket'>{legs_html}<div class='tk-foot'>"
                f"<span>cuota combinada ×{p['pago']}</span><span>{extra}</span></div></div>",
                unsafe_allow_html=True)


# --- App --------------------------------------------------------------------

def main():
    st.markdown(
        "<div class='casino-hd'>"
        "<span class='t'>🎰 <span class='neon'>MUNDIAL 2026</span> "
        "<span class='gold'>BETS</span> ⚽</span><br>"
        "<span style='opacity:.8;font-size:.88rem'>Picks del día · combinadas · "
        "histórico de aciertos — juego amistoso entre amigos, "
        "<b>no es apuesta real con dinero</b></span></div>",
        unsafe_allow_html=True)

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
    # Recordar la fecha elegida entre refrescos (se guarda en la URL), para que NO
    # se "salte de día" al recargar la página.
    fecha_url = st.query_params.get("fecha")
    if fecha_url in fechas:
        idx = fechas.index(fecha_url)
    elif hoy in fechas:
        idx = fechas.index(hoy)
    else:
        idx = len(fechas) - 1
    fecha = st.sidebar.selectbox("Elige la fecha", fechas, index=idx, key="sel_fecha")
    st.query_params["fecha"] = fecha
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
        "**Leyenda**\n\n🔒 Parlay Seguro · alta prob.\n\n"
        "⚖️ Parlay Arriesgado · prob. media\n\n🚀 Parlay Soñador · alto riesgo")

    modelo, cfg = cargar_modelo_y_cfg(version)
    # Marcadores en vivo/finales de ESPN para la fecha elegida (rellena terminados).
    live = marcadores_live(fecha, version)
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
            # Ordenar los picks por HORARIO del partido (los más temprano primero)
            for tier in ("segura", "arriesgada", "sonador"):
                leyes[tier].sort(
                    key=lambda r: _hora_orden(live, r["local"], r["visitante"]))
            st.subheader(f"Partidos del {fecha} · {len(partidos)} encuentros")

            # 🎟️ Boleto del día: combinada de los más seguros, con resultado en vivo
            parlay_dia = _evaluar_parlay(con, leyes["segura"], fecha)
            if parlay_dia:
                render_ticket(parlay_dia, mostrar_fecha=False)

            st.markdown("<div class='bloque' style='background:rgba(34,197,94,.10)'>",
                        unsafe_allow_html=True)
            st.markdown("### 🔒 Parlay Seguro  ·  alta probabilidad, bajo riesgo")
            if leyes["segura"]:
                for r in leyes["segura"]:
                    render_pick(r, con, fecha, live)
                render_multiplicador(leyes["segura"])
            else:
                st.caption("Hoy ningún partido alcanza el umbral de seguridad.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bloque' style='background:rgba(245,158,11,.10)'>",
                        unsafe_allow_html=True)
            st.markdown("### ⚖️ Parlay Arriesgado  ·  probabilidad media, mejor pago")
            if leyes["arriesgada"]:
                for r in leyes["arriesgada"]:
                    render_pick(r, con, fecha, live)
                render_multiplicador(leyes["arriesgada"])
            else:
                st.caption("Sin candidatos en la banda media hoy.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='bloque' style='background:rgba(239,68,68,.10)'>",
                        unsafe_allow_html=True)
            st.markdown("### 🚀 Parlay Soñador  ·  baja probabilidad, alto valor")
            for r in leyes["sonador"]:
                render_pick(r, con, fecha, live)
            render_multiplicador(leyes["sonador"])
            st.markdown("</div>", unsafe_allow_html=True)

    # --- TAB 2: detalle por partido ---
    with tab2:
        if not partidos:
            st.info("No hay partidos del Mundial 2026 en esta fecha.")
        else:
            # Ordenar los partidos por horario (los más temprano primero)
            partidos_ord = sorted(
                partidos, key=lambda r: _hora_orden(live, r["local"], r["visitante"]))
            cols = st.columns(2)
            for k, row in enumerate(partidos_ord):
                with cols[k % 2]:
                    render_tarjeta_partido(con, modelo, cfg, row, live)

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
            st.info("Pulsa el botón para ver cómo ha acertado cada parlay en las "
                    "jornadas ya jugadas. (Tarda unos segundos: entrena el modelo "
                    "en cada fecha pasada.)")

    con.close()


main()
