"""
generar_leyes.py — FASE 3: sistema de recomendación diaria.

Dado el calendario de un día del Mundial, genera 3 listas de apuestas:

  LEY SEGURA     : probabilidad alta (>= segura_min) y confianza alta/media.
  LEY ARRIESGADA : probabilidad media (banda arriesgada_min..max), mejor "pago".
  LEY SOÑADOR    : baja probabilidad / alto valor: marcadores exactos y una
                   combinada (parlay) de varios partidos.

Cada recomendación trae una justificación basada en datos (forma reciente,
goles esperados, head-to-head) y un "pago" potencial = 1 / probabilidad,
para darle sabor de apuesta amistosa (no es dinero real).
"""

from __future__ import annotations

from ..fase1_datos import db
from ..fase2_modelo.predecir_partido import predecir
from . import contexto


# --- Generación de candidatos por partido -----------------------------------

def _candidatos(pred: dict) -> list:
    """Convierte la predicción de un partido en una lista de apuestas candidatas:
    (mercado, seleccion, etiqueta, probabilidad). Solo mercados de goles (sólidos)."""
    loc, vis = pred["local"], pred["visitante"]
    c = []
    x = pred["1x2"]
    c += [("1X2", "local", f"Gana {loc}", x["local"]),
          ("1X2", "visitante", f"Gana {vis}", x["visitante"]),
          ("1X2", "empate", "Empate", x["empate"])]
    d = pred["doble_oportunidad"]
    c += [("DC", "1X", f"{loc} gana o empata", d["1X"]),
          ("DC", "X2", f"{vis} gana o empata", d["X2"]),
          ("DC", "12", "No hay empate", d["12"])]
    ou = pred["over_under"]
    for L in (1.5, 2.5, 3.5):
        c += [("OU", f"over_{L}", f"Más de {L} goles", ou[f"over_{L}"]),
              ("OU", f"under_{L}", f"Menos de {L} goles", ou[f"under_{L}"])]
    b = pred["btts"]
    c += [("BTTS", "si", "Ambos anotan: Sí", b["si"]),
          ("BTTS", "no", "Ambos anotan: No", b["no"])]
    fa = pred["primer_en_anotar"]
    c += [("1erGol", "local", f"{loc} anota primero", fa["local"]),
          ("1erGol", "visitante", f"{vis} anota primero", fa["visitante"])]
    h = pred["handicap_sugerido"]
    cubre = h["local_cubre"] if h["favorito"] == "local" else h["visitante_cubre"]
    quien = loc if h["favorito"] == "local" else vis
    c += [("Hcap", "fav", f"{quien} hándicap {h['linea']:+.1f}", cubre)]
    return c


def _mejor_no_trivial(candidatos, pmin, pmax=1.01):
    """Devuelve el candidato de mayor probabilidad dentro de [pmin, pmax],
    evitando los 'Más de 1.5' que casi siempre son triviales en la Ley Segura."""
    elegibles = [c for c in candidatos if pmin <= c[3] <= pmax
                 and not (c[0] == "OU" and c[1] == "over_1.5")]
    if not elegibles:
        return None
    return max(elegibles, key=lambda c: c[3])


# --- Justificación basada en datos ------------------------------------------

def _justificacion(con, pred, fecha, apuesta) -> str:
    loc, vis = pred["local"], pred["visitante"]
    fl = contexto.forma_reciente(con, loc, fecha)
    fv = contexto.forma_reciente(con, vis, fecha)
    h2h = contexto.head_to_head(con, loc, vis, fecha)
    partes = [
        f"goles esperados {loc} {pred['lambda_local']:.1f}–{pred['lambda_visit']:.1f} {vis}",
        f"{loc} forma {fl['racha'] or 's/d'} ({fl['gf_prom']:.1f} GF/{fl['gc_prom']:.1f} GC)",
        f"{vis} forma {fv['racha'] or 's/d'} ({fv['gf_prom']:.1f} GF/{fv['gc_prom']:.1f} GC)",
    ]
    if h2h["n"] > 0:
        partes.append(
            f"H2H {h2h['n']}: {loc} {h2h['gana_a']}–{h2h['empates']}–{h2h['gana_b']} {vis}")
    return "; ".join(partes)


def _pago(prob: float) -> float:
    """Cuota / pago potencial = 1 / probabilidad (mínimo 1.01)."""
    return round(1.0 / max(prob, 0.01), 2)


# --- Generación de las 3 Leyes ----------------------------------------------

def generar(con, modelo, cfg: dict, fecha: str) -> dict:
    """Genera las 3 listas de apuestas para una fecha del Mundial."""
    leyes_cfg = cfg["leyes"]
    seg_min = leyes_cfg["segura_min"]
    arr_min, arr_max = leyes_cfg["arriesgada_min"], leyes_cfg["arriesgada_max"]

    partidos = db.calendario_de_fecha(con, fecha)
    segura, arriesgada, sonador = [], [], []

    for row in partidos:
        loc, vis, neutral = row["local"], row["visitante"], row["neutral"]
        pred = predecir(con, modelo, cfg, loc, vis, neutral)
        nivel = pred["confianza"]["nivel"]
        cands = _candidatos(pred)
        encab = f"{loc} vs {vis}"

        # --- Ley Segura: mejor candidato >= seg_min y confianza no baja ---
        if nivel in ("alto", "medio"):
            best = _mejor_no_trivial(cands, seg_min)
            if best:
                segura.append({
                    "partido": encab, "local": loc, "visitante": vis,
                    "mercado": best[0], "seleccion": best[1],
                    "apuesta": best[2], "prob": best[3],
                    "pago": _pago(best[3]), "confianza": nivel,
                    "motivo": _justificacion(con, pred, fecha, best),
                })

        # --- Ley Arriesgada: mejor candidato en la banda media ---
        best_a = _mejor_no_trivial(cands, arr_min, arr_max)
        if best_a:
            arriesgada.append({
                "partido": encab, "local": loc, "visitante": vis,
                "mercado": best_a[0], "seleccion": best_a[1],
                "apuesta": best_a[2], "prob": best_a[3],
                "pago": _pago(best_a[3]), "confianza": nivel,
                "motivo": _justificacion(con, pred, fecha, best_a),
            })

        # --- Ley Soñador: marcador exacto más probable ---
        i, j, pr = pred["marcadores_top3"][0]
        sonador.append({
            "partido": encab, "local": loc, "visitante": vis,
            "mercado": "marcador", "seleccion": f"{i}-{j}",
            "apuesta": f"Marcador exacto {i}-{j}", "prob": pr,
            "pago": _pago(pr), "confianza": nivel,
            "motivo": f"el marcador más probable según el modelo ({100*pr:.0f}%)",
        })

    # Ordenar por probabilidad (las más fiables primero en cada lista)
    segura.sort(key=lambda r: r["prob"], reverse=True)
    arriesgada.sort(key=lambda r: r["prob"], reverse=True)
    sonador.sort(key=lambda r: r["prob"], reverse=True)

    # --- Combinada soñadora: parlay de los 3 picks más seguros del día ---
    parlay = None
    if len(segura) >= 2:
        top = segura[:3]
        prob_comb = 1.0
        for t in top:
            prob_comb *= t["prob"]
        parlay = {
            "apuestas": [f"{t['partido']}: {t['apuesta']}" for t in top],
            "prob": prob_comb,
            "pago": _pago(prob_comb),
        }

    return {
        "fecha": fecha,
        "n_partidos": len(partidos),
        "segura": segura,
        "arriesgada": arriesgada,
        "sonador": sonador,
        "parlay": parlay,
    }
