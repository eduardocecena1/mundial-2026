"""
seguimiento.py — FASE 4: histórico de aciertos del modelo.

Evalúa cómo le habrían ido las 3 Leyes en jornadas YA jugadas de la competición
activa. Para cada fecha pasada entrena el modelo SOLO con datos anteriores (sin
trampa), genera las recomendaciones y las compara con el resultado real.

Nota sobre eliminatorias: el 1X2 se resuelve a 90 minutos, como en cualquier casa
de apuestas. Un partido que acaba empatado y se decide en la prórroga o en los
penales cuenta como empate para el mercado, aunque haya un clasificado.

Uso:
    python -m src.fase4_interfaz.seguimiento                       # toda la temporada jugada
    python -m src.fase4_interfaz.seguimiento --desde 2026-09-01 --hasta 2027-01-31
"""

from __future__ import annotations

import argparse

from ..config import cargar_config, competicion_id, nombre_competicion
from ..fase1_datos import db
from ..fase2_modelo.entrenar import entrenar_modelo
from ..fase3_recomendacion.generar_leyes import generar

# Centinela: el partido acabó 0-0, así que ningún equipo anotó primero.
SIN_GOLES = object()


def _partido(con, local, visit, fecha):
    """Fila del partido jugado (o None si no está)."""
    return con.execute(
        """SELECT espn_id, goles_local, goles_visitante FROM partidos
            WHERE jugado=1 AND fecha=? AND local=? AND visitante=?""",
        (fecha, local, visit),
    ).fetchone()


def _resultado_partido(con, local, visit, fecha):
    """Marcador real de un partido jugado (o None si no está)."""
    f = _partido(con, local, visit, fecha)
    return (f["goles_local"], f["goles_visitante"]) if f else None


def _primer_equipo_gol(con, local, visit, fecha):
    """Equipo que anotó primero, o None si no se puede saber.

    Se empareja por espn_id: en clubes hay eliminatorias de ida y vuelta entre
    los mismos dos equipos, así que (fecha, local, visitante) ya no identifica
    un partido de forma tan clara como en un Mundial.

    Si el partido está jugado pero no tenemos sus goles, se bajan de ESPN al
    vuelo y se guardan. Sin esto, un solo pick de 'primer gol' sin resolver
    anula el parlay del día entero en las estadísticas (`_acumula_combo`).
    """
    p = _partido(con, local, visit, fecha)
    if p is None:
        return None

    def _consulta():
        f = con.execute(
            """SELECT equipo FROM goleadores
                WHERE espn_id=? AND minuto IS NOT NULL
                ORDER BY minuto ASC LIMIT 1""",
            (p["espn_id"],),
        ).fetchone()
        return f["equipo"] if f else None

    primero = _consulta()
    if primero is not None:
        return primero

    # Sin goles registrados: si el partido acabó 0-0, no hay primer goleador y
    # eso es un dato, no una laguna.
    if (p["goles_local"], p["goles_visitante"]) == (0, 0):
        return SIN_GOLES

    from ..fase1_datos.marcadores_vivo import _guardar_goles
    from ..config import competicion_id
    try:
        if _guardar_goles(con, competicion_id(), p["espn_id"], fecha):
            con.commit()
            return _consulta()
    except Exception:
        pass                      # sin red, el pick queda sin evaluar (None)
    return None


def evaluar(con, rec: dict, fecha: str):
    """¿Acertó la apuesta? Devuelve True/False, o None si no es evaluable."""
    res = _resultado_partido(con, rec["local"], rec["visitante"], fecha)
    if res is None:
        return None
    gl, gv = res
    m, s = rec["mercado"], rec["seleccion"]
    total = gl + gv

    if m == "1X2":
        real = "local" if gl > gv else ("empate" if gl == gv else "visitante")
        return s == real
    if m == "DC":
        if s == "1X":
            return gl >= gv
        if s == "X2":
            return gl <= gv
        if s == "12":
            return gl != gv
    if m == "OU":
        linea = float(s.split("_")[1])
        return (total > linea) if s.startswith("over") else (total < linea)
    if m == "BTTS":
        si = gl >= 1 and gv >= 1
        return si if s == "si" else (not si)
    if m == "marcador":
        return s == f"{gl}-{gv}"
    if m == "1erGol":
        primero = _primer_equipo_gol(con, rec["local"], rec["visitante"], fecha)
        if primero is None:
            return None
        if primero is SIN_GOLES:
            return False          # 0-0: nadie anotó primero, el pick falla
        objetivo = rec["local"] if s == "local" else rec["visitante"]
        return primero == objetivo
    return None  # Hcap u otros: no evaluado aquí


def _fechas_jugadas(con, desde, hasta, comp=None):
    comp = comp or competicion_id()
    filas = con.execute(
        """SELECT DISTINCT fecha FROM partidos
            WHERE competicion=? AND jugado=1 AND fecha>=? AND fecha<=?
            ORDER BY fecha""",
        (comp, desde, hasta),
    ).fetchall()
    return [f["fecha"] for f in filas]


def rango_temporada(cfg=None) -> tuple[str, str]:
    """Ventana de la temporada en curso de la competición activa.

    La temporada europea va de julio a junio; sin esto habría que tocar código
    cada año."""
    from datetime import date
    cfg = cfg or cargar_config()
    temporada = cfg.get("competicion", {}).get("temporada")
    if temporada is None:
        hoy = date.today()
        temporada = hoy.year if hoy.month >= 7 else hoy.year - 1
    return f"{temporada}-07-01", f"{int(temporada) + 1}-06-30"


def evaluar_rango(con, cfg, desde=None, hasta=None) -> dict:
    """Calcula el histórico de aciertos por fecha y los totales (sin imprimir).
    Devuelve un dict reutilizable por la CLI y por la web.

    {
      "fechas": [{"fecha", "segura": [a, t], "arriesgada": [a, t], "sonador": [a, t]}, ...],
      "totales": {"segura": [a, t], "arriesgada": [a, t], "sonador": [a, t]}
    }
    """
    if desde is None or hasta is None:
        d_def, h_def = rango_temporada(cfg)
        desde, hasta = desde or d_def, hasta or h_def
    n_corto = cfg.get("leyes", {}).get("max_picks", 3)  # nº de patas del parlay "corto"
    fechas = _fechas_jugadas(con, desde, hasta)
    tot = {"segura": [0, 0], "arriesgada": [0, 0], "sonador": [0, 0]}        # picks sueltos
    combo = {"segura": [0, 0], "arriesgada": [0, 0], "sonador": [0, 0]}      # combinada corta
    combo_largo = {"segura": [0, 0], "arriesgada": [0, 0], "sonador": [0, 0]}  # combinada larga
    por_fecha = []
    parlays = []

    def _acumula_combo(acc, oks_sub, fila, clave):
        """Suma 1 día ganado/jugado si TODAS las patas son evaluables."""
        if oks_sub and all(o is not None for o in oks_sub):
            gano = all(oks_sub)
            acc[0] += 1 if gano else 0
            acc[1] += 1
            fila[clave] = 1 if gano else 0
        else:
            fila[clave] = None

    for fecha in fechas:
        # Entrenar con corte en la fecha: no usar el resultado del propio día
        modelo = entrenar_modelo(con, cfg, hasta=fecha)
        leyes = generar(con, modelo, cfg, fecha)
        fila = {"fecha": fecha}
        for tier in ("segura", "arriesgada", "sonador"):
            oks = [evaluar(con, rec, fecha) for rec in leyes[tier]]
            # Aciertos de picks SUELTOS (para el gráfico de % por jornada)
            a = sum(1 for o in oks if o is True)
            t = sum(1 for o in oks if o is not None)
            tot[tier][0] += a
            tot[tier][1] += t
            fila[tier] = [a, t]
            # COMBINADA CORTA (top 3) y LARGA (todas): pegan solo si TODAS sus
            # patas pegan, y solo cuentan cuando todas son evaluables.
            _acumula_combo(combo[tier], oks[:n_corto], fila, tier + "_combo")
            _acumula_combo(combo_largo[tier], oks, fila, tier + "_combo_largo")

        # Boleto visual de la combinada SEGURA corta del día (top 3 patas).
        p = _evaluar_parlay(con, leyes["segura"][:n_corto], fecha)
        fila["parlay"] = p
        if p:
            parlays.append(p)
        por_fecha.append(fila)

    return {"fechas": por_fecha, "totales": tot, "combinada_totales": combo,
            "combinada_largo_totales": combo_largo,
            "parlays": parlays, "parlay_totales": _resumen_parlays(parlays)}


def _evaluar_parlay(con, segura, fecha):
    """Construye y evalúa la combinada del día con TODOS los picks del parlay.
    Gana solo si TODAS las patas pegan. Devuelve None si no hay al menos 2 picks.
    'acerto' es True/False, o None si alguna pata no se pudo evaluar (pendiente)."""
    if len(segura) < 2:
        return None
    legs_src = segura
    legs = []
    oks = []
    prob = 1.0
    for r in legs_src:
        ok = evaluar(con, r, fecha)
        oks.append(ok)
        prob *= r["prob"]
        legs.append({"texto": f"{r['partido']}: {r['apuesta']}", "ok": ok})
    if any(o is None for o in oks):
        acerto = None
    elif all(o is True for o in oks):
        acerto = True
    else:
        acerto = False
    return {"fecha": fecha, "legs": legs, "prob": prob,
            "pago": round(1.0 / max(prob, 0.01), 2), "acerto": acerto}


def _resumen_parlays(parlays):
    """Totales tipo casa de apuestas: jugados, ganados y balance de fichas
    (apostando 1 ficha por combinada cada día)."""
    jugados = ganados = 0
    balance = 0.0
    for p in parlays:
        if p["acerto"] is None:
            continue  # boleto anulado: no cuenta
        jugados += 1
        if p["acerto"]:
            ganados += 1
            balance += p["pago"] - 1.0
        else:
            balance -= 1.0
    return {"jugados": jugados, "ganados": ganados, "balance": round(balance, 2)}


def correr(con, cfg, desde=None, hasta=None):
    datos = evaluar_rango(con, cfg, desde, hasta)
    if not datos["fechas"]:
        print(f"No hay fechas jugadas de {nombre_competicion(cfg)} en ese rango.")
        return

    f0, f1 = datos["fechas"][0]["fecha"], datos["fechas"][-1]["fecha"]
    print(f"HISTÓRICO DE ACIERTOS — {nombre_competicion(cfg)} ({f0} a {f1})")
    print("(cada jornada se predice entrenando SOLO con datos previos)\n")
    for fila in datos["fechas"]:
        linea = [f"  {fila['fecha']}:"]
        for tier in ("segura", "arriesgada", "sonador"):
            a, t = fila[tier]
            if t:
                linea.append(f"{tier} {a}/{t}")
        print("  " + "  |  ".join(linea))

    print("\n=== TOTALES ===")
    etiquetas = {"segura": "🔒 Segura", "arriesgada": "⚖️  Arriesgada", "sonador": "🚀 Soñador"}
    for tier in ("segura", "arriesgada", "sonador"):
        a, t = datos["totales"][tier]
        pg, pj = datos["combinada_totales"][tier]
        lg, lj = datos["combinada_largo_totales"][tier]
        pct = f"{100*a/t:.0f}%" if t else "s/d"
        pctc = f"{100*pg/pj:.0f}%" if pj else "s/d"
        pctl = f"{100*lg/lj:.0f}%" if lj else "s/d"
        print(f"  {etiquetas[tier]:16s}: corta {pg}/{pj} ({pctc})  |  "
              f"larga {lg}/{lj} ({pctl})  |  picks sueltos {a}/{t} ({pct})")


def main():
    parser = argparse.ArgumentParser(description="Histórico de aciertos del modelo.")
    parser.add_argument("--desde", default=None,
                        help="Por defecto, el inicio de la temporada en curso.")
    parser.add_argument("--hasta", default=None,
                        help="Por defecto, el fin de la temporada en curso.")
    args = parser.parse_args()
    con = db.conectar()
    cfg = cargar_config()
    correr(con, cfg, args.desde, args.hasta)
    con.close()


if __name__ == "__main__":
    main()
