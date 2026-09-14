"""
parlay_global.py — Los parlays del día cruzando TODAS las ligas.

La pestaña de siempre genera picks de una sola competición, así que un día sin
Champions es un día sin boleto. Aquí se cruzan las 17 competiciones vigentes y
se arma el mismo formato de siempre —corto y largo × Seguro, Intermedio y
Soñador— quedándose con los picks de las ligas donde el modelo **tiene medido
que acierta**.

Dos ideas sostienen el módulo:

1. **Qué liga es "segura" se mide, no se supone.** `calibracion_ligas.py` recorre
   dos temporadas con walk-forward y deja en `data/calibracion_ligas.json` el
   acierto real de cada liga contra lo que el modelo declaró. De ahí salen dos
   números por liga: `skill` (cuánto le gana el modelo a la frecuencia histórica
   de esa liga) y `calib` (si es optimista o conservador al declarar sus
   probabilidades). El skill decide quién entra; la calibración ajusta la
   probabilidad para que los picks de ligas distintas sean comparables.

2. **Un boleto monotemático no es un boleto.** Sin restricción, las 10 patas más
   probables del día salen casi todas "gana o empata". Por eso la selección es un
   greedy con dos topes: una sola pata por partido y un máximo de patas del mismo
   mercado.
"""

from __future__ import annotations

from ..config import competiciones
from ..fase1_datos import db
from ..fase2_modelo.predecir_partido import predecir
from .generar_leyes import _candidatos, _justificacion, _pago

# Mismos mercados vetados que en la calibración: el hándicap es ambiguo de
# evaluar y 'primer gol' solo se puede resolver en Champions (la tabla
# `goleadores` no cubre las ligas domésticas), y una pata no evaluable anula el
# boleto entero en el histórico.
MERCADOS_EXCLUIDOS = ("Hcap", "1erGol")
SELECCIONES_EXCLUIDAS = ("over_1.5",)

TIERS = ("segura", "arriesgada", "sonador")


# --- Qué ligas entran -------------------------------------------------------

def ligas_elegibles(con, cfg: dict, cal: dict | None, fecha: str) -> dict:
    """Catálogo de ligas con su veredicto para el boleto de esa fecha.

    Devuelve slug -> {nombre, skill, calib: {tier: float}, elegible, motivo}.
    Sin JSON de calibración (`cal=None`) todas las ligas vigentes entran con
    factor 1.0: la sección sigue funcionando, solo que sin el filtro de calidad.
    """
    pg = cfg.get("parlay_global", {})
    skill_min = pg.get("skill_min", 0.08)
    n_min = pg.get("min_picks_liga", 300)

    catalogo = competiciones(cfg)
    vigentes = db.competiciones_vigentes(con, fecha, list(catalogo.keys()))

    out = {}
    for slug, meta in catalogo.items():
        info = {
            "nombre": meta.get("nombre", slug),
            "skill": None, "calib": {t: 1.0 for t in TIERS},
            "n": 0, "elegible": False, "motivo": "",
        }
        if slug not in vigentes:
            # Liga sin calendario en la temporada en curso: sus fuerzas llevan
            # meses sin actualizarse. Hoy es el caso de la liga suiza.
            info["motivo"] = "sin calendario esta temporada"
            out[slug] = info
            continue
        if not cal:
            info["elegible"] = True
            info["motivo"] = "sin calibrar"
            out[slug] = info
            continue

        e = (cal.get("ligas") or {}).get(slug)
        if not e:
            info["motivo"] = "sin datos de calibración"
            out[slug] = info
            continue

        info["skill"] = e.get("skill")
        info["n"] = (e.get("segura") or {}).get("n", 0)
        for t in TIERS:
            c = (e.get(t) or {}).get("calib")
            if c:
                info["calib"][t] = c

        if info["n"] < n_min:
            info["motivo"] = f"muestra corta ({info['n']} picks)"
        elif (info["skill"] or 0) < skill_min:
            info["motivo"] = f"skill bajo ({info['skill']:+.3f})"
        else:
            info["elegible"] = True
            info["motivo"] = f"skill {info['skill']:+.3f}"
        out[slug] = info
    return out


# --- Construcción del pool de candidatos ------------------------------------

def _pool_de_partido(pred: dict) -> list:
    """Mejor candidato de CADA mercado del partido: (mercado, sel, etiqueta, prob).

    Guardar uno por mercado (y no solo el mejor absoluto) es lo que permite que
    el tope por mercado tenga a qué recurrir cuando ya lleva 4 'gana o empata'.
    """
    mejor: dict = {}
    for mercado, seleccion, etiqueta, prob in _candidatos(pred):
        if mercado in MERCADOS_EXCLUIDOS or seleccion in SELECCIONES_EXCLUIDAS:
            continue
        if mercado not in mejor or prob > mejor[mercado][3]:
            mejor[mercado] = (mercado, seleccion, etiqueta, prob)
    return list(mejor.values())


def _candidatos_del_dia(con, modelo, cfg: dict, fecha: str, ligas: dict) -> tuple:
    """Recorre los partidos del día en las ligas elegibles y arma los candidatos.

    Devuelve (candidatos_por_tier, preds) donde `preds` indexa la predicción por
    espn_id para poder calcular la justificación solo de los picks que acaben
    entrando al boleto.
    """
    L = cfg["leyes"]
    comps = [s for s, i in ligas.items() if i["elegible"]]
    if not comps:
        return {t: [] for t in TIERS}, {}

    filas = db.calendario_multi_de_fecha(con, fecha, comps)
    cands = {t: [] for t in TIERS}
    preds = {}

    for row in filas:
        comp = row["competicion"]
        try:
            pred = predecir(con, modelo, cfg, row["local"], row["visitante"],
                            row["neutral"], competicion=comp)
        except KeyError:
            # Equipo sin histórico (recién ascendido, rival de copa). Cruzando 17
            # ligas pasa a diario; se salta el partido, no el día entero.
            continue

        espn_id = row["espn_id"]
        preds[espn_id] = pred
        nivel = pred["confianza"]["nivel"]
        calib = ligas[comp]["calib"]
        base = {
            "partido": f"{row['local']} vs {row['visitante']}",
            "local": row["local"], "visitante": row["visitante"],
            "competicion": comp, "liga": ligas[comp]["nombre"],
            "fecha_hora": row["fecha_hora"], "espn_id": espn_id,
            "confianza": nivel, "skill_liga": ligas[comp]["skill"],
        }

        for mercado, seleccion, etiqueta, prob in _pool_de_partido(pred):
            # Seguro e Intermedio se clasifican por la probabilidad AJUSTADA:
            # un 0.66 en una liga conservadora puede ser más fiable que un 0.70
            # en una donde el modelo se viene arriba.
            for tier, dentro in (
                ("segura", lambda p: p >= L["segura_min"]),
                ("arriesgada",
                 lambda p: L["arriesgada_min"] <= p <= L["arriesgada_max"]),
            ):
                p_aj = min(prob * calib[tier], 0.97)
                if not dentro(p_aj):
                    continue
                # La Ley Segura exige además datos suficientes de los equipos.
                if tier == "segura" and nivel not in ("alto", "medio"):
                    continue
                cands[tier].append({
                    **base, "mercado": mercado, "seleccion": seleccion,
                    "apuesta": etiqueta, "prob": prob, "prob_aj": round(p_aj, 4),
                    "calib": calib[tier], "pago": _pago(prob),
                })

        # Ley Soñador: el marcador exacto más probable. Un solo candidato por
        # partido, así que el tope por mercado no aplica aquí.
        i, j, pr = pred["marcadores_top3"][0]
        p_aj = min(pr * calib["sonador"], 0.97)
        cands["sonador"].append({
            **base, "mercado": "marcador", "seleccion": f"{i}-{j}",
            "apuesta": f"Marcador exacto {i}-{j}", "prob": pr,
            "prob_aj": round(p_aj, 4), "calib": calib["sonador"], "pago": _pago(pr),
        })

    return cands, preds


# --- Selección --------------------------------------------------------------

def seleccionar(candidatos: list, n: int, max_por_mercado: int | None = None) -> list:
    """Greedy por probabilidad ajustada con dos topes.

    - **Una sola pata por partido**: si no, el boleto multiplicaría
      probabilidades correlacionadas y la cuota sería mentira.
    - **Tope por mercado**: evita el boleto de 10 patas todas "gana o empata".
    """
    usados: set = set()
    por_mercado: dict = {}
    elegidos: list = []
    for c in sorted(candidatos, key=lambda c: (-c["prob_aj"], -c["prob"])):
        if c["espn_id"] in usados:
            continue
        if max_por_mercado and por_mercado.get(c["mercado"], 0) >= max_por_mercado:
            continue
        elegidos.append(c)
        usados.add(c["espn_id"])
        por_mercado[c["mercado"]] = por_mercado.get(c["mercado"], 0) + 1
        if len(elegidos) >= n:
            break
    return elegidos


def _combinada(picks: list) -> dict | None:
    """Probabilidad y cuota del boleto completo.

    La cuota se calcula con la probabilidad del MODELO, no con la ajustada: así
    el multiplicador sigue siendo comparable con el de la pestaña de Champions.
    La ajustada solo sirve para elegir y ordenar.
    """
    if not picks:
        return None
    p = 1.0
    p_aj = 1.0
    for k in picks:
        p *= k["prob"]
        p_aj *= k["prob_aj"]
    # OJO: aquí NO se usa `_pago`. Esa función topa la probabilidad en 0.01 para
    # que un pick suelto no dé una cuota absurda, pero en una combinada de 10
    # marcadores exactos la probabilidad real es de 1e-9 y el tope convertiría un
    # pago de mil millones en un ridículo "100x". En el boleto Soñador la cuota
    # desorbitada es justamente la gracia.
    return {"n": len(picks), "prob": p, "prob_aj": p_aj,
            "pago": round(1.0 / p, 2) if p > 0 else None}


# --- API pública ------------------------------------------------------------

def generar_global(con, modelo, cfg: dict, fecha: str, cal: dict | None = None) -> dict:
    """Los 6 boletos del día (corto y largo × 3 Leyes) cruzando todas las ligas."""
    pg = cfg.get("parlay_global", {})
    n_corto = pg.get("n_corto", cfg["leyes"].get("max_picks", 3))
    n_largo = pg.get("n_largo", 10)
    tope = pg.get("max_por_mercado", 4)

    ligas = ligas_elegibles(con, cfg, cal, fecha)
    cands, preds = _candidatos_del_dia(con, modelo, cfg, fecha, ligas)

    boletos = {}
    usados_motivo = []
    for tier in TIERS:
        # El tope por mercado no aplica al Soñador (todo es marcador exacto) ni
        # al corto de 3 patas, donde se usa la mitad para no encorsetarlo.
        tope_tier = None if tier == "sonador" else tope
        largo = seleccionar(cands[tier], n_largo, tope_tier)
        corto = seleccionar(cands[tier], n_corto,
                            None if tier == "sonador" else max(2, tope // 2))
        boletos[tier] = {
            "corto": corto, "largo": largo,
            "combinada_corto": _combinada(corto),
            "combinada_largo": _combinada(largo),
            "n_candidatos": len(cands[tier]),
        }
        usados_motivo.extend(corto + largo)

    # La justificación solo se calcula para las patas que se van a enseñar.
    for pick in usados_motivo:
        if "motivo" in pick:
            continue
        pred = preds.get(pick["espn_id"])
        if pred is None:
            continue
        tupla = (pick["mercado"], pick["seleccion"], pick["apuesta"], pick["prob"])
        pick["motivo"] = _justificacion(con, pred, fecha, tupla)

    elegibles = [s for s, i in ligas.items() if i["elegible"]]
    return {
        "fecha": fecha,
        "ligas": ligas,
        "n_ligas_elegibles": len(elegibles),
        "n_partidos": len(preds),
        "calibrado": bool(cal),
        "generado_el": (cal or {}).get("generado_el"),
        **boletos,
    }
