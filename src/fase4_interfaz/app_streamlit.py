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

from src.config import (cargar_config, competicion_id, competiciones,
                        fijar_competicion, liga_de_competicion, nombre_competicion)
from src.fase1_datos import db
from src.fase1_datos import marcadores_vivo as mv
from src.fase2_modelo import eliminatoria as elim
from src.fase2_modelo import simulacion as sim
from src.fase2_modelo.entrenar import entrenar_modelo
from src.fase2_modelo.predecir_partido import predecir
from src.fase3_recomendacion.generar_leyes import generar
from src.fase4_interfaz.seguimiento import (
    _evaluar_parlay, _resultado_partido, evaluar, evaluar_rango,
)


# --- Configuración de página -----------------------------------------------

st.set_page_config(page_title="Predicciones de fútbol", page_icon="⚽",
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

@st.cache_data(ttl=60 * 60 * 12, show_spinner="Actualizando datos...")
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
    cfg = cargar_config()
    n = con.execute("SELECT COUNT(*) AS c FROM partidos").fetchone()["c"]
    ult = db.get_meta(con, "ultima_actualizacion")
    if n == 0 or ult != dia:
        from src.fase1_datos.actualizar_diario import actualizar
        # Base vacía = despliegue nuevo sin `data/clubes.db`: hay que bajar TODO el
        # histórico o el modelo entrenaría con cuatro partidos. Tarda bastante, pero
        # pasa una sola vez; lo normal es que la base venga en el repo.
        actualizar(con, cfg, completo=(n == 0))
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


@st.cache_data(ttl=45, show_spinner=False)
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


def fechas_competicion(version: str, comp: str):
    con = db.conectar()
    filas = con.execute(
        "SELECT DISTINCT fecha FROM partidos WHERE competicion=? "
        "AND temporada=(SELECT MAX(temporada) FROM partidos WHERE competicion=?) "
        "ORDER BY fecha",
        (comp, comp),
    ).fetchall()
    con.close()
    return [f["fecha"] for f in filas]


@st.cache_data(show_spinner="Simulando la fase liga 10.000 veces...")
def cargar_simulacion(version: str, comp: str):
    """Monte Carlo de la fase liga: probabilidad de top-8 / playoff / eliminación."""
    con = db.conectar()
    cfg = cargar_config()
    fila = con.execute(
        "SELECT MAX(temporada) t FROM partidos WHERE competicion=?", (comp,)).fetchone()
    temporada = fila["t"] if fila else None
    if temporada is None:
        con.close()
        return None
    partidos = sim.partidos_fase_liga(con, comp, temporada)
    if not partidos:
        con.close()
        return None
    modelo, _ = cargar_modelo_y_cfg(version)
    datos = sim.simular_fase_liga(modelo, partidos, cfg, liga_de_competicion(comp, cfg))
    con.close()
    return datos


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
    pred = predecir(con, modelo, cfg, loc, vis, row["neutral"],
                    competicion=row["competicion"])
    conf = pred["confianza"]
    sede = "campo neutral" if pred["neutral"] else "con localía"
    hora = _hora_mx(live, loc, vis)
    hora_txt = f"🕐 {hora} hrs MX · " if hora else ""
    with st.container(border=True):
        st.markdown(f"#### {loc} 🆚 {vis}")
        st.caption(f"{hora_txt}{row['ciudad'] or 'sede por confirmar'} · {sede} · "
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


def render_parlay_expander(titulo: str, picks: list, con, fecha: str, live: dict):
    """Un parlay como DESPLEGABLE: el título resume (nº de patas, multiplicador y
    si pegó), y al hacer clic se abren todos los picks que lleva, ordenados por
    horario, más el multiplicador combinado."""
    if not picks:
        return
    # Multiplicador combinado (producto de las cuotas)
    cuota = 1.0
    for r in picks:
        cuota *= r["pago"]
    cuota_txt = f"×{cuota:,.0f}" if cuota >= 1000 else f"×{cuota:.2f}"
    # Estado de la combinada (pega solo si TODAS las patas pegan)
    oks = [evaluar(con, r, fecha) for r in picks]
    if oks and all(o is not None for o in oks):
        estado = "✅ PEGÓ" if all(oks) else "❌ NO PEGÓ"
    else:
        estado = "⏳ EN JUEGO"
    label = f"{titulo}  ·  {len(picks)} patas  ·  {cuota_txt}  ·  {estado}"
    with st.expander(label, expanded=False):
        picks_ord = sorted(
            picks, key=lambda r: _hora_orden(live, r["local"], r["visitante"]))
        for r in picks_ord:
            render_pick(r, con, fecha, live)
        render_multiplicador(picks)


def render_historico(version: str):
    """Calcula y dibuja el histórico de aciertos (entrena el modelo por jornada)."""
    hist = cargar_historico(version)
    if not hist["fechas"]:
        st.info("Aún no hay jornadas jugadas para evaluar.")
        return
    tot = hist["totales"]
    combo = hist.get("combinada_totales", {})
    combo_l = hist.get("combinada_largo_totales", {})
    etiquetas = {"segura": "🔒 Parlay Seguro", "arriesgada": "⚖️ Parlay Intermedio",
                 "sonador": "🚀 Parlay Soñador"}
    st.markdown("**Combinada completa** = el parlay pega solo si **todas** sus patas "
                "pegan juntas (como un boleto real). Se mide el **corto** (3 patas) y "
                "el **largo** (todos los juegos del día).")
    cols = st.columns(3)
    for col, tier in zip(cols, ("segura", "arriesgada", "sonador")):
        pg, pj = combo.get(tier, [0, 0])
        lg, lj = combo_l.get(tier, [0, 0])
        a, t = tot[tier]
        col.markdown(f"**{etiquetas[tier]}**")
        col.metric("🎟️ Corto (3 patas)", f"{pg}/{pj} días",
                   f"{(100*pg/pj if pj else 0):.0f}% pegó completa", delta_color="off")
        col.metric("🧱 Largo (todos los juegos)", f"{lg}/{lj} días",
                   f"{(100*lg/lj if lj else 0):.0f}% pegó completa", delta_color="off")
        col.caption(f"picks sueltos: {a}/{t} ({100*a/t:.0f}%)" if t else "—")

    # Gráfico: % de picks sueltos que pegaron por jornada
    filas = []
    for f in hist["fechas"]:
        for tier in ("segura", "arriesgada", "sonador"):
            a, t = f[tier]
            if t:
                filas.append({"fecha": f["fecha"], "Parlay": etiquetas[tier],
                              "acierto_%": 100 * a / t})
    df = pd.DataFrame(filas)
    if not df.empty:
        st.caption("Gráfico: % de picks sueltos acertados por jornada.")
        chart = (alt.Chart(df).mark_line(point=True)
                 .encode(x="fecha:N", y=alt.Y("acierto_%:Q",
                                              scale=alt.Scale(domain=[0, 100])),
                         color="Parlay:N", tooltip=["fecha", "Parlay", "acierto_%"])
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

def render_clasificacion(version: str, comp: str):
    """Pestaña de clasificación proyectada de la fase liga (Monte Carlo)."""
    datos = cargar_simulacion(version, comp)
    if not datos:
        st.info("Esta competición no tiene fase liga con datos suficientes.")
        return

    st.caption(
        f"{datos['n_sims']:,} simulaciones · {datos['jugados']} partidos ya jugados "
        f"y {datos['pendientes']} por jugar. Top-{datos['plazas_directas']} pasa "
        f"directo a octavos; del {datos['plazas_directas'] + 1} al "
        f"{datos['plazas_playoff']} juega el playoff; el resto queda eliminado."
        .replace(",", " "))

    filas = [{
        "Equipo": e,
        "Pos. media": round(v["posicion_media"], 1),
        "Pts medios": round(v["puntos_medios"], 1),
        "Top-8 %": round(100 * v["top_directo"], 1),
        "Playoff %": round(100 * v["playoff"], 1),
        "Fuera %": round(100 * v["eliminado"], 1),
    } for e, v in sim.tabla_ordenada(datos)]

    df = pd.DataFrame(filas)
    st.dataframe(
        df, use_container_width=True, hide_index=True,
        column_config={
            "Top-8 %": st.column_config.ProgressColumn(
                "Top-8 %", format="%.1f%%", min_value=0, max_value=100),
            "Playoff %": st.column_config.ProgressColumn(
                "Playoff %", format="%.1f%%", min_value=0, max_value=100),
            "Fuera %": st.column_config.ProgressColumn(
                "Fuera %", format="%.1f%%", min_value=0, max_value=100),
        })

    largo = df.melt(id_vars="Equipo", value_vars=["Top-8 %", "Playoff %", "Fuera %"],
                    var_name="Destino", value_name="Probabilidad")
    grafico = alt.Chart(largo).mark_bar().encode(
        y=alt.Y("Equipo:N", sort=list(df["Equipo"]), title=None),
        x=alt.X("Probabilidad:Q", title="Probabilidad (%)", stack="normalize",
                axis=alt.Axis(format="%")),
        color=alt.Color("Destino:N", scale=alt.Scale(
            domain=["Top-8 %", "Playoff %", "Fuera %"],
            range=["#22c55e", "#f59e0b", "#ef4444"]), title=None),
        tooltip=["Equipo", "Destino", "Probabilidad"],
    ).properties(height=max(320, 18 * len(df)))
    st.altair_chart(grafico, use_container_width=True)


def render_eliminatorias(con, modelo, cfg, comp: str, fecha: str, temporada: int):
    """Probabilidades de clasificación de las eliminatorias a doble partido."""
    rondas = con.execute(
        """SELECT DISTINCT ronda FROM partidos
            WHERE competicion=? AND temporada=? AND leg IS NOT NULL
            ORDER BY ronda""", (comp, temporada)).fetchall()
    if not rondas:
        st.info("Todavía no hay eliminatorias a doble partido en el calendario. "
                "Empiezan en febrero.")
        return

    liga = liga_de_competicion(comp, cfg)
    nombres = [r["ronda"] for r in rondas]
    ronda = st.selectbox("Ronda", nombres, index=len(nombres) - 1)
    partidos = con.execute(
        "SELECT * FROM partidos WHERE competicion=? AND temporada=? AND ronda=?",
        (comp, temporada, ronda)).fetchall()

    for tie in elim.emparejar_legs(partidos):
        try:
            r = elim.eliminatoria(modelo, tie["equipo_a"], tie["equipo_b"],
                                  tie["marcador_ida"], liga)
        except KeyError:
            continue
        with st.container(border=True):
            ida = (f"ida {r['marcador_ida'][0]}-{r['marcador_ida'][1]}"
                   if r["ida_jugada"] else "ida por jugar")
            st.markdown(f"**{tie['equipo_a']}** vs **{tie['equipo_b']}** · {ida}")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Pasa {tie['equipo_a']}", f"{100*r['pasa_a']:.0f}%")
            c2.metric(f"Pasa {tie['equipo_b']}", f"{100*r['pasa_b']:.0f}%")
            c3.metric("Se va a prórroga", f"{100*r['prorroga']:.0f}%")
            ou = elim.over_under_global(r, (2.5, 3.5, 4.5))
            st.caption("Goles en el global de la eliminatoria: " + " · ".join(
                f"Over {l} **{100*ou[f'over_{l}']:.0f}%**" for l in (2.5, 3.5, 4.5)))


# --- App --------------------------------------------------------------------

def main():
    # Selector de competición: se aplica ANTES de leer nada de la base, porque
    # decide qué partidos son "los del día".
    cfg = cargar_config()
    catalogo = competiciones(cfg)
    comp_url = st.query_params.get("comp")
    if comp_url in catalogo:
        fijar_competicion(comp_url)
    comp = competicion_id(cfg)
    titulo = nombre_competicion(cfg)

    st.markdown(
        "<div class='casino-hd'>"
        f"<span class='t'>🎰 <span class='neon'>{titulo.upper()}</span> "
        "<span class='gold'>BETS</span> ⚽</span><br>"
        "<span style='opacity:.8;font-size:.88rem'>Picks del día · combinadas · "
        "histórico de aciertos — juego amistoso entre amigos, "
        "<b>no es apuesta real con dinero</b></span></div>",
        unsafe_allow_html=True)

    # Frescura automática diaria: descarga datos nuevos una vez al día al abrir.
    hoy = date.today().isoformat()
    version = asegurar_datos(hoy)

    # Sidebar
    st.sidebar.header("Competición")
    slugs = list(catalogo.keys())
    nombres = {c: catalogo[c].get("nombre", c) for c in slugs}
    comp_sel = st.sidebar.selectbox(
        "Elige la competición", slugs, index=slugs.index(comp) if comp in slugs else 0,
        format_func=lambda c: nombres[c], key="sel_comp")
    if comp_sel != comp:
        st.query_params["comp"] = comp_sel
        st.query_params.pop("fecha", None)
        fijar_competicion(comp_sel)
        st.rerun()
    st.query_params["comp"] = comp_sel

    fechas = fechas_competicion(version, comp)
    if not fechas:
        st.error("No hay datos para esta competición. Corre primero: "
                 "`python -m src.fase1_datos.descargar_clubes`")
        return

    st.sidebar.header("Jornada")
    # Recordar la fecha elegida entre refrescos (se guarda en la URL), para que NO
    # se "salte de día" al recargar la página.
    fecha_url = st.query_params.get("fecha")
    if fecha_url in fechas:
        idx = fechas.index(fecha_url)
    elif hoy in fechas:
        idx = fechas.index(hoy)
    else:
        # Sin partidos hoy: la jornada más cercana en el futuro, o la última jugada.
        futuras = [f for f in fechas if f >= hoy]
        idx = fechas.index(futuras[0]) if futuras else len(fechas) - 1
    fecha = st.sidebar.selectbox("Elige la fecha", fechas, index=idx, key="sel_fecha")
    st.query_params["fecha"] = fecha
    st.sidebar.caption(f"📅 Datos actualizados al {version}")
    if st.sidebar.button("🔄 Forzar actualización ahora"):
        from src.fase1_datos.actualizar_diario import actualizar
        con = db.conectar()
        with st.spinner("Descargando datos frescos..."):
            actualizar(con, cfg)
        con.close()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Leyenda**\n\n🔒 Parlay Seguro · alta prob.\n\n"
        "⚖️ Parlay Intermedio · prob. media\n\n🚀 Parlay Soñador · alto riesgo")

    st.sidebar.caption("🔴 Marcadores en vivo · pulsa 🔄 Refrescar o recarga la página")

    cuerpo(version, fecha, comp)


def cuerpo(version: str, fecha: str, comp: str):
    """Cuerpo de la app. Los marcadores en vivo se refrescan al RECARGAR la página
    o al pulsar '🔄 Refrescar ya' (el auto-refresco automático se desactivó para
    mantener la app estable en el plan gratis)."""
    from datetime import datetime as _dt
    modelo, cfg = cargar_modelo_y_cfg(version)
    con = db.conectar()
    # Marcadores en vivo/finales de ESPN para la fecha (rellena los terminados).
    # Si ESPN falla, mv.aplicar degrada solo y la app sigue con lo ya descargado.
    try:
        live = mv.aplicar(con, fecha, cfg)
    except Exception:
        live = {"finales": 0, "vivo": {}, "horarios": {}, "goles": 0}
    cc1, cc2 = st.columns([4, 1])
    cc1.caption(f"🔴 Marcadores al **{_dt.now(mv.TZ_MX).strftime('%H:%M:%S')}** hrs MX "
                f"· pulsa 🔄 o recarga para traer los más nuevos")
    cc2.button("🔄 Refrescar ya", key="refresh_live")
    partidos = db.calendario_de_fecha(con, fecha, comp)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Picks del día", "📊 Detalle por partido",
         "🏁 Clasificación proyectada", "🏆 Histórico de aciertos"])

    # --- TAB 1: las 3 Leyes ---
    with tab1:
        if not partidos:
            st.info("No hay partidos en esta fecha.")
        else:
            leyes = generar(con, modelo, cfg, fecha)
            N = cfg["leyes"].get("max_picks", 3)
            st.subheader(f"Partidos del {fecha} · {len(partidos)} encuentros")
            st.caption("Pícale a un parlay para desplegar los picks que lleva. "
                       "**Cortos** = las 3 mejores apuestas · **Largos** = todos los "
                       "juegos del día.")

            st.markdown("##### 🎟️ Parlays cortos (las 3 mejores)")
            render_parlay_expander("🔒 Seguro", leyes["segura"][:N], con, fecha, live)
            render_parlay_expander("⚖️ Intermedio", leyes["arriesgada"][:N], con, fecha, live)
            render_parlay_expander("🚀 Soñador", leyes["sonador"][:N], con, fecha, live)

            st.markdown("##### 🧱 Parlays largos (todos los juegos del día)")
            render_parlay_expander("🔒 Seguro largo", leyes["segura"], con, fecha, live)
            render_parlay_expander("⚖️ Intermedio largo", leyes["arriesgada"], con, fecha, live)
            render_parlay_expander("🚀 Soñador largo", leyes["sonador"], con, fecha, live)

    # --- TAB 2: detalle por partido ---
    with tab2:
        if not partidos:
            st.info("No hay partidos en esta fecha.")
        else:
            # Ordenar los partidos por horario (los más temprano primero)
            partidos_ord = sorted(
                partidos, key=lambda r: _hora_orden(live, r["local"], r["visitante"]))
            cols = st.columns(2)
            for k, row in enumerate(partidos_ord):
                with cols[k % 2]:
                    render_tarjeta_partido(con, modelo, cfg, row, live)

    # --- TAB 3: fase liga (Monte Carlo) o eliminatorias, según toque ---
    with tab3:
        con_leg = [p for p in partidos if p["leg"]]
        if con_leg:
            st.subheader("Probabilidades de clasificación")
            st.caption("Global de los dos partidos, con prórroga (30 min a ritmo "
                       "proporcional) y penales al 50%. Sin regla de goles fuera "
                       "de casa: la UEFA la abolió en 2021.")
            render_eliminatorias(con, modelo, cfg, comp, fecha,
                                 con_leg[0]["temporada"])
        else:
            st.subheader("¿Dónde acaba cada equipo en la fase liga?")
            render_clasificacion(version, comp)

    # --- TAB 4: histórico de aciertos (bajo demanda, para no frenar la carga) ---
    with tab4:
        st.subheader("¿Qué tan bien ha acertado el modelo esta temporada?")
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
