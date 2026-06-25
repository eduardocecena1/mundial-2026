"""
seguimiento.py — FASE 4: histórico de aciertos del modelo.

Evalúa cómo le habrían ido las 3 Leyes en jornadas YA jugadas del Mundial 2026.
Para cada fecha pasada entrena el modelo SOLO con datos anteriores (sin trampa),
genera las recomendaciones y las compara con el resultado real.

Uso:
    python -m src.fase4_interfaz.seguimiento                       # todo el Mundial 2026 jugado
    python -m src.fase4_interfaz.seguimiento --desde 2026-06-11 --hasta 2026-06-23
"""

from __future__ import annotations

import argparse

from ..fase1_datos import db
from ..fase2_modelo.entrenar import cargar_config, entrenar_modelo
from ..fase3_recomendacion.generar_leyes import generar


def _resultado_partido(con, local, visit, fecha):
    """Marcador real de un partido jugado (o None si no está)."""
    f = con.execute(
        """SELECT goles_local, goles_visitante FROM partidos
            WHERE jugado=1 AND fecha=? AND local=? AND visitante=?""",
        (fecha, local, visit),
    ).fetchone()
    return (f["goles_local"], f["goles_visitante"]) if f else None


def _primer_equipo_gol(con, local, visit, fecha):
    """Equipo que anotó primero (de la tabla de goleadores), o None."""
    f = con.execute(
        """SELECT equipo FROM goleadores
            WHERE fecha=? AND local=? AND visitante=? AND minuto IS NOT NULL
            ORDER BY minuto ASC LIMIT 1""",
        (fecha, local, visit),
    ).fetchone()
    return f["equipo"] if f else None


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
        objetivo = rec["local"] if s == "local" else rec["visitante"]
        return primero == objetivo
    return None  # Hcap u otros: no evaluado aquí


def _fechas_jugadas(con, desde, hasta):
    filas = con.execute(
        """SELECT DISTINCT fecha FROM partidos
            WHERE es_mundial2026=1 AND jugado=1 AND fecha>=? AND fecha<=?
            ORDER BY fecha""",
        (desde, hasta),
    ).fetchall()
    return [f["fecha"] for f in filas]


def evaluar_rango(con, cfg, desde="2026-06-01", hasta="2026-12-31") -> dict:
    """Calcula el histórico de aciertos por fecha y los totales (sin imprimir).
    Devuelve un dict reutilizable por la CLI y por la web.

    {
      "fechas": [{"fecha", "segura": [a, t], "arriesgada": [a, t], "sonador": [a, t]}, ...],
      "totales": {"segura": [a, t], "arriesgada": [a, t], "sonador": [a, t]}
    }
    """
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


def correr(con, cfg, desde="2026-06-01", hasta="2026-12-31"):
    datos = evaluar_rango(con, cfg, desde, hasta)
    if not datos["fechas"]:
        print("No hay fechas jugadas del Mundial 2026 en ese rango.")
        return

    f0, f1 = datos["fechas"][0]["fecha"], datos["fechas"][-1]["fecha"]
    print(f"HISTÓRICO DE ACIERTOS — Mundial 2026 ({f0} a {f1})")
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
    parser.add_argument("--desde", default="2026-06-01")
    parser.add_argument("--hasta", default="2026-12-31")
    args = parser.parse_args()
    con = db.conectar()
    cfg = cargar_config()
    correr(con, cfg, args.desde, args.hasta)
    con.close()


if __name__ == "__main__":
    main()
