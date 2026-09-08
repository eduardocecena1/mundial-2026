"""
config.py — Carga y helpers de `config.yaml`.

Vive en la raíz de `src/` (y no dentro de una fase) porque lo necesitan tanto la
capa de datos como el modelo y la interfaz, y así se evita el import circular
que aparecería si `db.py` tuviera que importar de `fase2_modelo`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_CONFIG = RAIZ_PROYECTO / "config.yaml"

_CACHE: dict[str, dict] = {}

# Competición elegida en caliente (el selector de la app). Sobrescribe a
# config.yaml sin reescribir el archivo, que es lo que queremos en la nube:
# el sistema de archivos de Streamlit Cloud es efímero.
_OVERRIDE_COMPETICION: str | None = None


def cargar_config(ruta=None) -> dict:
    """Lee config.yaml (cacheado por ruta)."""
    ruta = Path(ruta) if ruta else RUTA_CONFIG
    clave = str(ruta)
    if clave not in _CACHE:
        with open(ruta, "r", encoding="utf-8") as f:
            _CACHE[clave] = yaml.safe_load(f)
    return _CACHE[clave]


def limpiar_cache() -> None:
    """Fuerza una relectura de config.yaml (lo usa el selector de la app)."""
    _CACHE.clear()


# --- Competición objetivo ---------------------------------------------------

def competicion(cfg: dict | None = None) -> dict:
    cfg = cfg or cargar_config()
    return cfg.get("competicion", {})


def fijar_competicion(slug: str | None) -> None:
    """Cambia la competición objetivo solo para este proceso."""
    global _OVERRIDE_COMPETICION
    _OVERRIDE_COMPETICION = slug


def competicion_id(cfg: dict | None = None) -> str:
    if _OVERRIDE_COMPETICION:
        return _OVERRIDE_COMPETICION
    return competicion(cfg).get("id", "uefa.champions")


def nombre_competicion(cfg: dict | None = None) -> str:
    cfg = cfg or cargar_config()
    slug = competicion_id(cfg)
    comp = competicion(cfg)
    if slug == comp.get("id") and comp.get("nombre"):
        return comp["nombre"]
    # Competición elegida en el selector: usamos su nombre del catálogo.
    return competiciones(cfg).get(slug, {}).get("nombre", slug)


def ruta_db(cfg: dict | None = None) -> Path:
    cfg = cfg or cargar_config()
    rel = competicion(cfg).get("bd", "data/clubes.db")
    ruta = Path(rel)
    return ruta if ruta.is_absolute() else RAIZ_PROYECTO / ruta


# --- Competiciones que alimentan el modelo ----------------------------------

def competiciones(cfg: dict | None = None) -> dict[str, dict]:
    cfg = cfg or cargar_config()
    return cfg.get("datos", {}).get("competiciones", {}) or {}


def liga_de_competicion(slug: str, cfg: dict | None = None) -> str:
    """Clave de agrupación para el modelo jerárquico. 'otras' si no está en config."""
    return competiciones(cfg).get(slug, {}).get("liga", "otras")


def peso_competicion(slug: str, cfg: dict | None = None) -> float:
    """Cuánto informa un partido de esa competición. 0.70 para las no listadas."""
    return float(competiciones(cfg).get(slug, {}).get("peso", 0.70))


def es_competicion_uefa(slug: str, cfg: dict | None = None) -> bool:
    return liga_de_competicion(slug, cfg) == "uefa"


def ligas_domesticas(cfg: dict | None = None) -> dict[str, str]:
    """slug -> clave de liga, solo de las competiciones domésticas."""
    return {slug: meta.get("liga", "otras")
            for slug, meta in competiciones(cfg).items()
            if meta.get("liga") != "uefa"}
