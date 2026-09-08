"""
espn_api.py — Cliente del API público de ESPN para fútbol.

Es la fuente de datos única del proyecto en modo clubes: sirve tanto el
histórico (para entrenar) como el calendario y los marcadores en vivo. No
necesita API key.

Endpoints usados (ambos públicos):
  scoreboard : .../soccer/{slug}/scoreboard?dates=YYYYMMDD-YYYYMMDD
  summary    : .../soccer/{slug}/summary?event={id}   -> eventos de gol

Dos detalles del API que condicionan el diseño:

1. El scoreboard devuelve como mucho **100 eventos por petición**, sin avisar
   de que truncó. Por eso `partidos_rango` trocea por mes y, si un tramo vuelve
   con 100 eventos justos, lo vuelve a trocear por semanas.

2. Las respuestas de fechas pasadas son inmutables, las de hoy/futuro no. La
   caché en disco solo guarda tramos que terminan ANTES de hoy; así el
   histórico se descarga una vez y los partidos en curso siempre van frescos.
   Se cachea el resultado YA NORMALIZADO, no el JSON de ESPN: el crudo trae
   plantillas, retransmisiones y cuotas que no usamos y multiplicaba por 20 el
   tamaño en disco.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
DIR_CACHE = RAIZ_PROYECTO / "data" / "raw" / "espn"

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# ESPN corta la respuesta del scoreboard aquí, en silencio.
TOPE_EVENTOS = 100

PAUSA_S = 0.25      # cortesía entre peticiones
REINTENTOS = 3
TIMEOUT_S = 30

# Ojo: NO poner un User-Agent de navegador. ESPN devuelve 403 a los UA tipo Chrome
# y en cambio sirve sin problema con el UA por defecto de requests. Comprobado.
_SESION = requests.Session()


class ESPNNoDisponible(Exception):
    """La competición no existe en ESPN (404) o el feed no responde."""


# --- Capa HTTP --------------------------------------------------------------

def _get(url: str) -> dict:
    """GET con reintentos y backoff exponencial."""
    ultimo_error: Exception | None = None
    for intento in range(REINTENTOS):
        try:
            resp = _SESION.get(url, timeout=TIMEOUT_S)
            # 404 = esa competición no existe en ESPN; reintentar no arregla nada.
            if resp.status_code == 404:
                raise ESPNNoDisponible(url)
            resp.raise_for_status()
            datos = resp.json()
            time.sleep(PAUSA_S)
            return datos
        except ESPNNoDisponible:
            raise
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            ultimo_error = e
            time.sleep(2 ** intento)
    raise ESPNNoDisponible(f"ESPN falló tras {REINTENTOS} intentos: {url}") from ultimo_error


def _ymd(f: date) -> str:
    return f.strftime("%Y%m%d")


def _ruta_cache(slug: str, desde: date, hasta: date) -> Path:
    return DIR_CACHE / slug / f"{_ymd(desde)}-{_ymd(hasta)}.json"


def _pedir_tramo(slug: str, desde: date, hasta: date, usar_cache: bool = True) -> dict:
    """Un tramo de scoreboard ya normalizado, con caché en disco.

    Devuelve {"n_crudo": nº de eventos que trajo ESPN, "partidos": [...]}.
    `n_crudo` es lo que permite detectar el truncamiento en el tope de 100: no
    sirve contar los partidos normalizados, porque descartamos los malformados.
    """
    hoy = datetime.now(timezone.utc).date()
    cacheable = usar_cache and hasta < hoy
    ruta = _ruta_cache(slug, desde, hasta)

    if cacheable and ruta.exists():
        try:
            return json.loads(ruta.read_text())
        except (json.JSONDecodeError, KeyError):
            ruta.unlink()      # caché corrupta: se vuelve a bajar

    datos = _get(f"{BASE}/{slug}/scoreboard?dates={_ymd(desde)}-{_ymd(hasta)}")
    eventos = datos.get("events", [])
    resultado = {
        "n_crudo": len(eventos),
        "partidos": [p for p in (_normalizar(ev, slug) for ev in eventos) if p],
    }

    if cacheable:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(resultado))
    return resultado


# --- Normalización de un evento --------------------------------------------

def _a_int(valor) -> int | None:
    try:
        return int(valor)
    except (TypeError, ValueError):
        return None


def _leg(notas: list) -> int | None:
    """'1st Leg' / '2nd Leg' -> 1 / 2. None si no es eliminatoria a doble partido."""
    for n in notas or []:
        texto = str(n.get("headline") or n.get("text") or "").lower()
        if "1st leg" in texto:
            return 1
        if "2nd leg" in texto:
            return 2
    return None


def _normalizar(ev: dict, slug: str) -> dict | None:
    """Convierte un evento de ESPN al dict plano que guardamos en SQLite.

    Devuelve None si al evento le falta algo esencial (pasa con partidos
    aplazados o mal cargados en el feed)."""
    try:
        comp = ev["competitions"][0]
        competidores = comp["competitors"]
        if len(competidores) != 2:
            return None
        por_lado = {c["homeAway"]: c for c in competidores}
        local, visitante = por_lado.get("home"), por_lado.get("away")
        if not local or not visitante:
            return None
    except (KeyError, IndexError):
        return None

    estado = comp.get("status", {}).get("type", {})
    completado = bool(estado.get("completed"))
    gl = _a_int(local.get("score"))
    gv = _a_int(visitante.get("score"))

    # Un partido cuenta como jugado solo si terminó Y tiene marcador.
    jugado = 1 if (completado and gl is not None and gv is not None) else 0

    fecha_hora = ev.get("date", "")           # ISO UTC, p.ej. '2026-09-08T19:00Z'
    direccion = (comp.get("venue") or {}).get("address") or {}

    return {
        "espn_id": str(ev["id"]),
        "fecha": fecha_hora[:10],
        "fecha_hora": fecha_hora,
        "competicion": slug,
        "temporada": (ev.get("season") or {}).get("year"),
        "ronda": (ev.get("season") or {}).get("slug"),
        "leg": _leg(comp.get("notes")),
        "local": local["team"]["displayName"],
        "local_id": str(local["team"]["id"]),
        "visitante": visitante["team"]["displayName"],
        "visitante_id": str(visitante["team"]["id"]),
        # goles_* solo se rellenan cuando el partido terminó: son lo que entra
        # en la BD y alimenta el modelo. marcador_* es el tanteo actual, que en
        # un partido en curso es justo lo que la interfaz necesita mostrar.
        "goles_local": gl if jugado else None,
        "goles_visitante": gv if jugado else None,
        "marcador_local": gl,
        "marcador_visitante": gv,
        "global_local": _a_int(local.get("aggregateScore")),
        "global_visitante": _a_int(visitante.get("aggregateScore")),
        "jugado": jugado,
        "estado": estado.get("description", ""),
        "neutral": 1 if comp.get("neutralSite") else 0,
        "ciudad": direccion.get("city"),
        "pais": direccion.get("country"),
    }


# --- API pública ------------------------------------------------------------

def _tramos(desde: date, hasta: date, dias: int):
    """Parte [desde, hasta] en tramos de como mucho `dias` días."""
    cursor = desde
    while cursor <= hasta:
        fin = min(cursor + timedelta(days=dias - 1), hasta)
        yield cursor, fin
        cursor = fin + timedelta(days=1)


def _partidos_tramo(slug: str, desde: date, hasta: date, usar_cache: bool) -> list[dict]:
    """Partidos de un tramo, bisecando cuando ESPN trunca.

    Si la respuesta trae el tope de 100 eventos no sabemos cuántos se quedaron
    fuera, así que partimos el tramo por la mitad y repetimos. La bisección
    cuesta muchas menos peticiones que trocear siempre fino "por si acaso"."""
    datos = _pedir_tramo(slug, desde, hasta, usar_cache)

    if datos["n_crudo"] < TOPE_EVENTOS or desde == hasta:
        return datos["partidos"]

    medio = desde + (hasta - desde) // 2
    return (_partidos_tramo(slug, desde, medio, usar_cache)
            + _partidos_tramo(slug, medio + timedelta(days=1), hasta, usar_cache))


def partidos_rango(slug: str, desde: date, hasta: date, usar_cache: bool = True,
                   dias_tramo: int = 60) -> list[dict]:
    """Todos los partidos de una competición entre dos fechas, ya normalizados."""
    partidos: dict[str, dict] = {}

    for ini, fin in _tramos(desde, hasta, dias_tramo):
        for p in _partidos_tramo(slug, ini, fin, usar_cache):
            partidos[p["espn_id"]] = p          # dedup por si los tramos se solapan

    return sorted(partidos.values(), key=lambda p: (p["fecha"], p["espn_id"]))


def partidos_de_fecha(slug: str, fecha_iso: str) -> list[dict]:
    """Partidos de un solo día. Nunca usa caché: es lo que alimenta el modo en vivo."""
    f = datetime.strptime(fecha_iso, "%Y-%m-%d").date()
    return _pedir_tramo(slug, f, f, usar_cache=False)["partidos"]


def goles_del_partido(slug: str, espn_id: str) -> list[dict]:
    """Goles de un partido vía el endpoint summary.

    Devuelve [{equipo_id, equipo, jugador, minuto, autogol, penal}, ...] ordenado
    por minuto. Alimenta el mercado 'primer equipo en anotar'.

    Los penales de la tanda vienen marcados con shootout=True y se excluyen: no
    son goles del partido y arruinarían el 'primer equipo en anotar'.
    """
    try:
        datos = _get(f"{BASE}/{slug}/summary?event={espn_id}")
    except ESPNNoDisponible:
        return []

    goles = []
    for ev in datos.get("keyEvents") or []:
        if not ev.get("scoringPlay") or ev.get("shootout"):
            continue
        tipo = str((ev.get("type") or {}).get("text", "")).lower()
        equipo = ev.get("team") or {}
        if not equipo.get("id"):
            continue                      # un gol sin equipo no sirve aguas abajo
        atletas = ev.get("athletesInvolved") or []
        segundos = (ev.get("clock") or {}).get("value")
        goles.append({
            "equipo_id": str(equipo["id"]),
            "equipo": equipo.get("displayName"),
            "jugador": atletas[0].get("displayName") if atletas else None,
            "minuto": int(segundos // 60) if isinstance(segundos, (int, float)) else None,
            "autogol": 1 if "own goal" in tipo else 0,
            "penal": 1 if "penalty" in tipo else 0,
        })

    return sorted(goles, key=lambda g: (g["minuto"] is None, g["minuto"] or 0))


def ganador_tanda(slug: str, espn_id: str) -> str | None:
    """Id del equipo que ganó la tanda de penales, o None si no hubo tanda.

    Se usa para las eliminatorias: el 1X2 de casa de apuestas es a 90 minutos,
    pero para saber quién pasó hace falta la tanda."""
    try:
        datos = _get(f"{BASE}/{slug}/summary?event={espn_id}")
    except ESPNNoDisponible:
        return None

    anotados: dict[str, int] = {}
    hubo_tanda = False
    for ev in datos.get("keyEvents") or []:
        if not ev.get("shootout"):
            continue
        hubo_tanda = True
        equipo = ev.get("team") or {}
        if equipo.get("id") and ev.get("scoringPlay"):
            anotados[str(equipo["id"])] = anotados.get(str(equipo["id"]), 0) + 1

    if not hubo_tanda or len(anotados) == 0:
        return None
    return max(anotados, key=anotados.get)
