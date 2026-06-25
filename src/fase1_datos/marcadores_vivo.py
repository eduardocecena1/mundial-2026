"""
marcadores_vivo.py — Marcadores EN VIVO desde la API pública de ESPN (sin key).

El dataset histórico (martj42) sube los resultados con retraso (a veces horas o
un día después del partido). Para que los picks y boletos del día muestren el
resultado real al instante, este módulo trae los marcadores en vivo/finales de
ESPN y los aplica sobre la base:

  - Partidos TERMINADOS  -> se guardan como jugados (rellena el marcador).
  - Partidos EN CURSO    -> se devuelven aparte para mostrar "EN VIVO x-y"
                            (no se marcan como jugados ni se usan para el modelo).

Fuente: site.api.espn.com (endpoint público de marcadores del Mundial). Sin key.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta, timezone

import requests

from . import db

# México centro = UTC-6 todo el año (el país abolió el horario de verano en 2022).
TZ_MX = timezone(timedelta(hours=-6))


def hora_mexico(iso_utc: str) -> str:
    """Convierte una fecha-hora ISO en UTC (de ESPN) a 'HH:MM' hora centro de México."""
    if not iso_utc:
        return ""
    try:
        dt = datetime.fromisoformat(iso_utc.replace("Z", "+00:00"))
        return dt.astimezone(TZ_MX).strftime("%H:%M")
    except ValueError:
        return ""

ESPN_URL = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
            "fifa.world/scoreboard?dates={fecha}")

# Alias de nombres ESPN -> forma normalizada de nuestra BD (ya normalizados).
# Solo para los que NO coinciden tras normalizar acentos/puntuación.
ALIAS = {
    "czechia": "czech republic",
    "usa": "united states",
    "ir iran": "iran",
    "korea republic": "south korea",
    "cote divoire": "ivory coast",
    "cabo verde": "cape verde",
}


def _norm(s: str) -> str:
    """Normaliza un nombre de equipo para comparar entre fuentes:
    quita acentos, pasa a minúsculas, elimina 'and'/puntuación y espacios extra."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = f" {s} ".replace(" and ", " ")          # quitar la palabra 'and'
    s = re.sub(r"\s+", " ", s).strip()
    return ALIAS.get(s, s)


def obtener_marcadores(fecha_iso: str) -> list[dict]:
    """Descarga los partidos del Mundial para una fecha 'YYYY-MM-DD' desde ESPN.
    Devuelve lista de dicts: {equipos: [(nombre, goles), ...], completed, estado}."""
    fecha = fecha_iso.replace("-", "")
    r = requests.get(ESPN_URL.format(fecha=fecha), timeout=20)
    r.raise_for_status()
    data = r.json()
    partidos = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        tipo = comp.get("status", {}).get("type", {})
        equipos = []
        for c in comp.get("competitors", []):
            nombre = c.get("team", {}).get("displayName")
            try:
                goles = int(c.get("score"))
            except (TypeError, ValueError):
                goles = None
            equipos.append((nombre, goles))
        if len(equipos) == 2:
            partidos.append({
                "equipos": equipos,
                "completed": bool(tipo.get("completed")),
                "estado": tipo.get("description", ""),
                "fecha_hora": ev.get("date", ""),  # ISO en UTC
            })
    return partidos


def aplicar(con, fecha_iso: str) -> dict:
    """Aplica los marcadores de ESPN a la base para una fecha.

    Devuelve {"finales": n, "vivo": {(local, visitante): {"gl","gv","estado"}}}:
      - 'finales' = nº de partidos terminados que se guardaron como jugados.
      - 'vivo'    = partidos en curso (para mostrar "EN VIVO", sin grabarlos
                    como jugados).
    """
    try:
        espn = obtener_marcadores(fecha_iso)
    except Exception:
        return {"finales": 0, "vivo": {}}  # degradación elegante: sin live, seguimos

    # Partidos del Mundial 2026 que tenemos en BD para esa fecha
    db_rows = con.execute(
        "SELECT local, visitante FROM partidos "
        "WHERE es_mundial2026=1 AND fecha=?", (fecha_iso,)).fetchall()

    finales = 0
    vivo: dict = {}
    horarios: dict = {}
    for row in db_rows:
        local, visit = row["local"], row["visitante"]
        nl, nv = _norm(local), _norm(visit)
        for e in espn:
            (n1, g1), (n2, g2) = e["equipos"]
            en1, en2 = _norm(n1), _norm(n2)
            if {en1, en2} != {nl, nv}:
                continue
            # Hora del partido (ISO UTC para ordenar + 'HH:MM' México para mostrar)
            horarios[(local, visit)] = {
                "utc": e["fecha_hora"], "mx": hora_mexico(e["fecha_hora"])}
            # Asignar el marcador a NUESTRO orden local/visitante por nombre
            gl = g1 if en1 == nl else g2
            gv = g1 if en1 == nv else g2
            if e["completed"]:
                con.execute(
                    "UPDATE partidos SET goles_local=?, goles_visitante=?, jugado=1 "
                    "WHERE es_mundial2026=1 AND fecha=? AND local=? AND visitante=?",
                    (gl, gv, fecha_iso, local, visit))
                finales += 1
            elif gl is not None and gv is not None and e["estado"] not in ("Scheduled",):
                vivo[(local, visit)] = {"gl": gl, "gv": gv, "estado": e["estado"]}
            break
    con.commit()
    return {"finales": finales, "vivo": vivo, "horarios": horarios}
