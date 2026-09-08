"""
marcadores_vivo.py — Marcadores EN VIVO desde la API pública de ESPN (sin key).

La descarga histórica corre una vez al día; durante una jornada de Champions eso
no basta. Este módulo trae los marcadores del día directamente de ESPN y los
aplica sobre la base:

  - Partidos TERMINADOS  -> se guardan como jugados (rellena el marcador) y se
                            pueblan sus goles en `goleadores`.
  - Partidos EN CURSO    -> se devuelven aparte para mostrar "EN VIVO x-y"
                            (no se marcan como jugados ni se usan para el modelo).

El emparejamiento con la base es por **espn_id**, no por nombre. En selecciones
un diccionario de alias bastaba; con clubes sería una fuente constante de fallos
("Internazionale" / "Inter Milan", "Bodo/Glimt", "FC Bayern München").
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ..config import cargar_config, competicion_id
from . import db
from . import espn_api as espn

# México centro = UTC-6 todo el año (el país abolió el horario de verano en 2022).
TZ_MX = timezone(timedelta(hours=-6))

# Estados de ESPN que significan "aún no ha empezado".
ESTADOS_NO_INICIADO = {"Scheduled", "Postponed", "Canceled", "Delayed", "TBD"}


def hora_mexico(iso_utc: str) -> str:
    """Convierte una fecha-hora ISO en UTC (de ESPN) a 'HH:MM' hora centro de México."""
    if not iso_utc:
        return ""
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        return dt.astimezone(TZ_MX).strftime("%H:%M")
    except ValueError:
        return ""


def obtener_marcadores(fecha_iso: str, comp: str | None = None) -> list[dict]:
    """Partidos de ESPN de la competición activa para una fecha (sin caché)."""
    comp = comp or competicion_id()
    return espn.partidos_de_fecha(comp, fecha_iso)


def _guardar_goles(con, comp: str, espn_id: str, fecha: str) -> int:
    """Baja los goles de un partido terminado y los guarda en `goleadores`.

    Sin esto, un pick de 'primer equipo en anotar' queda sin resolver y
    `seguimiento._acumula_combo` descarta el día entero del histórico de parlays.
    """
    ya = con.execute("SELECT COUNT(*) n FROM goleadores WHERE espn_id=?",
                     (espn_id,)).fetchone()["n"]
    if ya:
        return 0

    goles = espn.goles_del_partido(comp, espn_id)
    if not goles:
        return 0

    con.executemany(
        """INSERT OR IGNORE INTO goleadores
             (espn_id, fecha, equipo, equipo_id, jugador, minuto, autogol, penal)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [(espn_id, fecha, g["equipo"], g["equipo_id"], g["jugador"],
          g["minuto"], g["autogol"], g["penal"]) for g in goles],
    )
    return len(goles)


def aplicar(con, fecha_iso: str, cfg: dict | None = None) -> dict:
    """Aplica los marcadores de ESPN a la base para una fecha.

    Devuelve {"finales", "vivo", "horarios", "goles"}:
      - 'finales'  = nº de partidos terminados que se guardaron como jugados.
      - 'vivo'     = {(local, visitante): {"gl","gv","estado"}} de los partidos
                     en curso (para mostrar "EN VIVO", sin grabarlos como jugados).
      - 'horarios' = {(local, visitante): {"utc","mx"}}.
      - 'goles'    = nº de goles nuevos guardados en `goleadores`.

    Ante cualquier fallo de red devuelve la estructura vacía COMPLETA (con todas
    las claves) para que la interfaz no tenga que defenderse de un dict a medias.
    """
    vacio = {"finales": 0, "vivo": {}, "horarios": {}, "goles": 0}
    cfg = cfg or cargar_config()
    comp = competicion_id(cfg)

    try:
        partidos_espn = obtener_marcadores(fecha_iso, comp)
    except Exception:
        return vacio                      # degradación elegante: sin live, seguimos

    # Lo que tenemos en BD para esa fecha, indexado por espn_id.
    en_bd = {r["espn_id"]: r for r in con.execute(
        "SELECT espn_id, local, visitante FROM partidos WHERE competicion=? AND fecha=?",
        (comp, fecha_iso)).fetchall()}

    finales = 0
    goles_nuevos = 0
    vivo: dict = {}
    horarios: dict = {}

    for p in partidos_espn:
        fila = en_bd.get(p["espn_id"])
        if not fila:
            # Partido que ESPN tiene y nosotros no (calendario movido): lo insertamos.
            from .descargar_clubes import guardar_partidos
            guardar_partidos(con, [p])
            fila = con.execute(
                "SELECT espn_id, local, visitante FROM partidos WHERE espn_id=?",
                (p["espn_id"],)).fetchone()
            if not fila:
                continue

        clave = (fila["local"], fila["visitante"])
        horarios[clave] = {"utc": p["fecha_hora"], "mx": hora_mexico(p["fecha_hora"])}

        if p["jugado"]:
            con.execute(
                "UPDATE partidos SET goles_local=?, goles_visitante=?, jugado=1 "
                "WHERE espn_id=?",
                (p["goles_local"], p["goles_visitante"], p["espn_id"]))
            finales += 1
            goles_nuevos += _guardar_goles(con, comp, p["espn_id"], fecha_iso)
        elif p["estado"] not in ESTADOS_NO_INICIADO:
            # En curso: ESPN ya publica el marcador parcial aunque 'jugado' sea 0.
            vivo[clave] = {"gl": p["marcador_local"] or 0,
                           "gv": p["marcador_visitante"] or 0,
                           "estado": p["estado"]}

    con.commit()
    return {"finales": finales, "vivo": vivo, "horarios": horarios, "goles": goles_nuevos}


if __name__ == "__main__":
    import sys

    fecha = sys.argv[1] if len(sys.argv) > 1 else datetime.now(TZ_MX).date().isoformat()
    con = db.conectar()
    db.inicializar(con)
    r = aplicar(con, fecha)
    print(f"{fecha}: {r['finales']} finales, {len(r['vivo'])} en vivo, "
          f"{r['goles']} goles guardados")
    for (l, v), d in r["vivo"].items():
        print(f"  🔴 {l} {d['gl']}-{d['gv']} {v}  ({d['estado']})")
    con.close()
