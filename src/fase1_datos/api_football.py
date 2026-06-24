"""
api_football.py — Capa OPCIONAL de estadísticas detalladas (API-Football).

Trae tarjetas, tiros, tiros a puerta, corners, faltas y posesión por partido,
que el dataset gratuito de goles no tiene. Pensado para el plan GRATIS
(100 requests/día), por eso:

  - CACHÉ permanente en SQLite: cada estadística se pide UNA sola vez y se guarda.
  - GUARDIÁN DE PRESUPUESTO: cuenta los requests del día y se detiene antes del
    límite (deja margen de seguridad). El conteo se reinicia cada día.
  - DEGRADACIÓN ELEGANTE: si no hay datos o se agota el presupuesto, el sistema
    sigue funcionando con los mercados de goles y baja la confianza; no inventa.

La key se lee de la variable de entorno API_FOOTBALL_KEY (archivo .env).

-------------------------------------------------------------------------------
NOTA IMPORTANTE (hallazgo del 2026-06-24):
El PLAN GRATIS de API-Football NO da acceso a la temporada actual (2026): la API
responde "Free plans do not have access to this season, try from 2022 to 2024."
Por tanto NO se pueden traer estadísticas EN VIVO del Mundial 2026 con el plan
gratis. Sí están disponibles temporadas 2022-2024 (Mundial 2022, Euro 2024, etc.),
útiles solo como prior histórico de confianza baja.

Decisión del proyecto: este módulo queda como infraestructura OPCIONAL y NO está
conectado al pipeline de predicción, porque un modelo de tarjetas/tiros con datos
de 2022-2024 sería de confianza baja y no aportaría un mercado fiable. El valor del
sistema está en los mercados de goles bien calibrados. Si en el futuro se consigue
un plan con la temporada 2026, este cliente ya está listo para alimentar esos
mercados (ajustando TEMPORADA_2026 / la temporada objetivo).
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import requests

from . import db

BASE = "https://v3.football.api-sports.io"
LIGA_MUNDIAL = 1          # id de la FIFA World Cup en API-Football
TEMPORADA_2026 = 2026
LIMITE_DIARIO_SEGURO = 90  # nos detenemos aquí aunque el plan permita 100


# --- Carga de la key --------------------------------------------------------

def cargar_key() -> str | None:
    """Lee API_FOOTBALL_KEY del entorno o del archivo .env."""
    if os.environ.get("API_FOOTBALL_KEY"):
        return os.environ["API_FOOTBALL_KEY"]
    env = db.RAIZ_PROYECTO / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("API_FOOTBALL_KEY="):
                v = line.split("=", 1)[1].strip()
                return v or None
    return None


# --- Guardián de presupuesto ------------------------------------------------

def _presupuesto_hoy(con) -> int:
    """Cuántos requests llevamos hoy (se reinicia cada día)."""
    hoy = date.today().isoformat()
    d = db.get_meta(con, "api_req_date")
    if d != hoy:
        db.set_meta(con, "api_req_date", hoy)
        db.set_meta(con, "api_req_count", "0")
        return 0
    return int(db.get_meta(con, "api_req_count", "0"))


def _registrar_request(con) -> None:
    n = _presupuesto_hoy(con) + 1
    db.set_meta(con, "api_req_count", str(n))


def requests_restantes(con) -> int:
    return max(0, LIMITE_DIARIO_SEGURO - _presupuesto_hoy(con))


class PresupuestoAgotado(Exception):
    """Se alcanzó el límite diario seguro de requests."""


# --- Cliente HTTP -----------------------------------------------------------

def _get(con, key: str, endpoint: str, params: dict) -> dict:
    """GET contra la API respetando el guardián de presupuesto y registrando el uso."""
    if requests_restantes(con) <= 0:
        raise PresupuestoAgotado(
            f"Límite diario seguro ({LIMITE_DIARIO_SEGURO}) alcanzado. "
            "Reintenta mañana; lo ya descargado queda en caché.")
    r = requests.get(f"{BASE}/{endpoint}",
                     headers={"x-apisports-key": key}, params=params, timeout=30)
    _registrar_request(con)
    r.raise_for_status()
    return r.json()


# --- Esquema de caché de estadísticas ---------------------------------------

ESQUEMA_STATS = """
CREATE TABLE IF NOT EXISTS api_fixtures (
    fixture_id INTEGER PRIMARY KEY,
    fecha      TEXT,
    local_api  TEXT,
    visit_api  TEXT,
    estado     TEXT          -- 'FT' terminado, 'NS' por jugar, etc.
);

CREATE TABLE IF NOT EXISTS stats_api (
    fixture_id   INTEGER,
    equipo       TEXT,        -- nombre del equipo según API-Football
    es_local     INTEGER,
    tiros        INTEGER,
    tiros_puerta INTEGER,
    corners      INTEGER,
    faltas       INTEGER,
    amarillas    INTEGER,
    rojas        INTEGER,
    posesion     REAL,
    PRIMARY KEY (fixture_id, equipo)
);
"""


def inicializar_stats(con) -> None:
    con.executescript(ESQUEMA_STATS)
    con.commit()


# --- Descarga de fixtures y estadísticas ------------------------------------

def sincronizar_fixtures(con, key: str) -> int:
    """Trae la lista de partidos del Mundial 2026 (1 request) y la cachea.
    Necesario para conocer los fixture_id con los que pedir estadísticas."""
    data = _get(con, key, "fixtures",
                {"league": LIGA_MUNDIAL, "season": TEMPORADA_2026})
    filas = []
    for it in data.get("response", []):
        fx = it["fixture"]
        teams = it["teams"]
        filas.append((
            fx["id"], fx["date"][:10],
            teams["home"]["name"], teams["away"]["name"],
            fx["status"]["short"],
        ))
    con.executemany(
        """INSERT INTO api_fixtures (fixture_id, fecha, local_api, visit_api, estado)
           VALUES (?,?,?,?,?)
           ON CONFLICT(fixture_id) DO UPDATE SET estado=excluded.estado""",
        filas,
    )
    con.commit()
    return len(filas)


def _parse_stat(valor):
    """Normaliza valores de la API ('45%', None, '12') a número."""
    if valor is None:
        return None
    if isinstance(valor, str) and valor.endswith("%"):
        try:
            return float(valor.rstrip("%"))
        except ValueError:
            return None
    try:
        return int(valor)
    except (ValueError, TypeError):
        return None


def descargar_estadisticas_fixture(con, key: str, fixture_id: int) -> bool:
    """Descarga y cachea las estadísticas de UN fixture (1 request).
    Devuelve True si guardó datos. Si ya está en caché, no gasta request."""
    ya = con.execute("SELECT 1 FROM stats_api WHERE fixture_id=? LIMIT 1",
                     (fixture_id,)).fetchone()
    if ya:
        return True  # caché: no gastamos cuota

    info = con.execute("SELECT local_api, visit_api FROM api_fixtures WHERE fixture_id=?",
                       (fixture_id,)).fetchone()
    data = _get(con, key, "fixtures/statistics", {"fixture": fixture_id})
    resp = data.get("response", [])
    if not resp:
        return False

    locales = info["local_api"] if info else None
    filas = []
    for equipo_block in resp:
        equipo = equipo_block["team"]["name"]
        es_local = 1 if equipo == locales else 0
        st = {s["type"]: s["value"] for s in equipo_block.get("statistics", [])}
        filas.append((
            fixture_id, equipo, es_local,
            _parse_stat(st.get("Total Shots")),
            _parse_stat(st.get("Shots on Goal")),
            _parse_stat(st.get("Corner Kicks")),
            _parse_stat(st.get("Fouls")),
            _parse_stat(st.get("Yellow Cards")),
            _parse_stat(st.get("Red Cards")),
            _parse_stat(st.get("Ball Possession")),
        ))
    con.executemany(
        """INSERT OR REPLACE INTO stats_api
             (fixture_id, equipo, es_local, tiros, tiros_puerta, corners,
              faltas, amarillas, rojas, posesion)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        filas,
    )
    con.commit()
    return True


def fixtures_jugados_sin_stats(con) -> list[int]:
    """fixture_id de partidos terminados que aún no tienen estadísticas cacheadas."""
    filas = con.execute(
        """SELECT f.fixture_id FROM api_fixtures f
            WHERE f.estado IN ('FT','AET','PEN')
              AND NOT EXISTS (SELECT 1 FROM stats_api s
                              WHERE s.fixture_id = f.fixture_id)
            ORDER BY f.fecha""").fetchall()
    return [r["fixture_id"] for r in filas]
