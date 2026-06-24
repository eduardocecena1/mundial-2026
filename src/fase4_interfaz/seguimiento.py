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
    fechas = _fechas_jugadas(con, desde, hasta)
    tot = {"segura": [0, 0], "arriesgada": [0, 0], "sonador": [0, 0]}
    por_fecha = []

    for fecha in fechas:
        # Entrenar con corte en la fecha: no usar el resultado del propio día
        modelo = entrenar_modelo(con, cfg, hasta=fecha)
        leyes = generar(con, modelo, cfg, fecha)
        fila = {"fecha": fecha}
        for tier in ("segura", "arriesgada", "sonador"):
            a = t = 0
            for rec in leyes[tier]:
                ok = evaluar(con, rec, fecha)
                if ok is None:
                    continue
                t += 1
                a += 1 if ok else 0
            tot[tier][0] += a
            tot[tier][1] += t
            fila[tier] = [a, t]
        por_fecha.append(fila)

    return {"fechas": por_fecha, "totales": tot}


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
        pct = f"{100*a/t:.0f}%" if t else "s/d"
        print(f"  {etiquetas[tier]:16s}: {a}/{t} aciertos  ({pct})")


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
