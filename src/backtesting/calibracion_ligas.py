"""
calibracion_ligas.py — ¿En qué ligas acierta de verdad el modelo?

El parlay multi-liga necesita responder una pregunta que el backtest normal no
responde: **cuando el modelo dice 75% en la Eredivisie, ¿pega el 75%?**. No es lo
mismo que preguntar en qué liga manda siempre el mismo equipo — una liga puede
estar desbalanceada y aun así ser impredecible entre los medianos.

Aquí se mide, liga por liga, el acierto REAL de los picks contra la probabilidad
que el modelo declaró, con un walk-forward sin fuga de futuro: cada partido se
predice con un modelo entrenado únicamente con lo anterior a su ventana.

El resultado se guarda en `data/calibracion_ligas.json` (versionado en git, ~10 KB)
y lo consume la vista multi-liga. Se regenera a mano cada pocas semanas:

    python -m src.backtesting.calibracion_ligas
    python -m src.backtesting.calibracion_ligas --rapido        # cadencia 30 días
    python -m src.backtesting.calibracion_ligas --desde 2023-07-01

## Por qué el reentrenamiento es semanal y no diario

Cada corte entrena con `hasta = inicio de la ventana`, y con ese modelo se
predicen los partidos de los 7 días siguientes. El modelo nunca ve un partido
posterior a su corte, así que no hay fuga; lo que hay es *desactualización*, que
empuja en la dirección contraria: en producción se entrena con todo hasta ayer,
o sea que la calibración medida aquí es un **suelo conservador** del acierto
real. Con `vida_media_anios: 1.0` una semana es el 1.9% de la vida media y las
fuerzas apenas se mueven.

Hacerlo por fecha jugada costaría 621 entrenamientos en vez de ~112.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from ..config import cargar_config, competiciones
from ..fase1_datos import db
from ..fase2_modelo.entrenar import entrenar_modelo
from ..fase2_modelo.predecir_partido import predecir
from ..fase3_recomendacion.generar_leyes import _candidatos
from ..fase4_interfaz import seguimiento

RUTA_JSON = Path("data/calibracion_ligas.json")

# Mercados que NO entran en el boleto multi-liga:
#   Hcap   — ambiguo de evaluar, ya excluido en `generar_leyes._mejor_no_trivial`.
#   1erGol — la tabla `goleadores` solo cubre Champions (85 goles, 21 partidos);
#            fuera de ahí hay que pedírselo a ESPN partido a partido, lo que es
#            inviable en un walk-forward de 11.000 partidos y frágil en el día a
#            día. Un pick no evaluable ANULA el boleto entero en el histórico.
MERCADOS_EXCLUIDOS = ("Hcap", "1erGol")

# 'Más de 1.5 goles' es casi siempre trivial; se excluye igual que en la pestaña
# de Champions, para que el boleto no se llene de patas regaladas.
SELECCIONES_EXCLUIDAS = ("over_1.5",)

# Pseudo-observaciones del prior en el shrinkage Beta-binomial. Con n=150 la
# evidencia propia de la liga pesa la mitad; con n=600 pesa el 80%. Se elige 150
# porque a p≈0.8 el error estándar con n=150 es 3.3 pp, el mismo orden que las
# diferencias reales entre ligas que estamos midiendo: ahí el 50/50 es el punto
# de indiferencia correcto.
K_SHRINK = 150

# Ninguna liga puede mover la probabilidad de un pick más de un ±15%. Fuera de
# esa banda, con dos temporadas de muestra, lo que se está midiendo es ruido.
CALIB_MIN, CALIB_MAX = 0.85, 1.15


# --- Walk-forward -----------------------------------------------------------

def _cortes(desde: str, hasta: str, cadencia_dias: int) -> list:
    """Fechas de corte del walk-forward: [(inicio_ventana, fin_ventana), ...]."""
    d0, d1 = date.fromisoformat(desde), date.fromisoformat(hasta)
    out, c = [], d0
    while c < d1:
        sig = min(c + timedelta(days=cadencia_dias), d1)
        out.append((c.isoformat(), sig.isoformat()))
        c = sig
    return out


def _pool_de_partido(pred: dict) -> list:
    """Mejor candidato de CADA mercado para un partido.

    Se mide sobre el mismo pool que luego usa la selección del boleto (el mejor
    de cada mercado, no solo el mejor absoluto), para que el factor de liga
    valga para cualquier mercado que acabe entrando.
    """
    mejor: dict = {}
    for mercado, seleccion, etiqueta, prob in _candidatos(pred):
        if mercado in MERCADOS_EXCLUIDOS or seleccion in SELECCIONES_EXCLUIDAS:
            continue
        if mercado not in mejor or prob > mejor[mercado][2]:
            mejor[mercado] = (mercado, seleccion, prob)
    return list(mejor.values())


def _tier(prob: float, cfg: dict) -> str | None:
    """En qué Ley cae un candidato según su probabilidad declarada."""
    L = cfg["leyes"]
    if prob >= L["segura_min"]:
        return "segura"
    if L["arriesgada_min"] <= prob <= L["arriesgada_max"]:
        return "arriesgada"
    return None


def _rps_1x2(p_local: float, p_empate: float, p_visit: float, res: str) -> float:
    """Ranked Probability Score de un 1X2 (menor = mejor)."""
    obs = {"local": (1, 0, 0), "empate": (0, 1, 0), "visitante": (0, 0, 1)}[res]
    acc_p = acc_o = 0.0
    total = 0.0
    for p, o in zip((p_local, p_empate, p_visit), obs):
        acc_p += p
        acc_o += o
        total += (acc_p - acc_o) ** 2
    return total / 2.0


def _calib_hasta_ahora(acc: dict, comp: str, tier: str, sesgo: float) -> float:
    """Factor de calibración de una liga con lo observado HASTA AHORA.

    Para el histórico de boletos no vale usar la calibración final: se habría
    calculado con los propios días que se quieren evaluar. Se recalcula de forma
    expansiva, usando solo el pasado de cada día.
    """
    n, ok, sp = acc.get(comp, {}).get(tier, [0, 0, 0.0])
    if not n or not sp:
        return 1.0
    p_decl = sp / n
    prior = p_decl * sesgo
    p_shrunk = (ok + K_SHRINK * prior) / (n + K_SHRINK)
    return min(max(p_shrunk / p_decl, CALIB_MIN), CALIB_MAX)


def evaluar_ventana(con, cfg: dict, desde: str, hasta: str,
                    cadencia_dias: int = 7, verbose: bool = True) -> dict:
    """Recorre el histórico con walk-forward y acumula aciertos por liga.

    Hace dos cosas de una sola pasada:
      1. mide el acierto real de cada liga (el ranking), y
      2. arma el boleto multi-liga de CADA día y anota si pegó (el histórico).

    Lo segundo se apoya en lo primero de forma *expansiva*: el boleto del día D se
    construye con la calibración deducida solo de los días anteriores a D. Usar la
    calibración final sería mirar el futuro.
    """
    from ..fase3_recomendacion.parlay_global import seleccionar

    pg = cfg.get("parlay_global", {})
    n_corto = pg.get("n_corto", cfg["leyes"].get("max_picks", 3))
    n_largo = pg.get("n_largo", 10)
    tope = pg.get("max_por_mercado", 4)
    n_min = pg.get("min_picks_liga", 300)
    skill_min = pg.get("skill_min", 0.08)

    catalogo = set(competiciones(cfg).keys())
    # liga -> tier -> [n, aciertos, suma_prob]
    acc: dict = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))
    # liga -> [n, suma_rps, suma_rps_base]
    rps: dict = defaultdict(lambda: [0, 0.0, 0.0])
    base = _frecuencias_base(con, desde)
    historial: list = []

    cortes = _cortes(desde, hasta, cadencia_dias)
    saltados = 0
    for i, (ini, fin) in enumerate(cortes, 1):
        partidos = con.execute(
            "SELECT * FROM partidos WHERE jugado=1 AND fecha>=? AND fecha<? "
            "ORDER BY fecha", (ini, fin)).fetchall()
        partidos = [p for p in partidos if p["competicion"] in catalogo]
        if not partidos:
            continue
        # Corte estricto: el modelo solo ve lo ANTERIOR a esta ventana.
        modelo = entrenar_modelo(con, cfg, hasta=ini)

        por_dia: dict = defaultdict(list)
        for row in partidos:
            por_dia[row["fecha"]].append(row)

        for fecha in sorted(por_dia):
            # --- sesgo global y elegibilidad con lo sabido HASTA AQUÍ ---
            glob: dict = defaultdict(lambda: [0, 0, 0.0])
            for tiers in acc.values():
                for t, (n, ok, sp) in tiers.items():
                    g = glob[t]
                    g[0] += n
                    g[1] += ok
                    g[2] += sp
            sesgo = {t: ((g[1] / g[0]) / (g[2] / g[0]) if g[0] and g[2] else 1.0)
                     for t, g in glob.items()}
            elegibles = set()
            for comp, r in rps.items():
                n_seg = acc.get(comp, {}).get("segura", [0])[0]
                skill = ((r[2] - r[1]) / r[2]) if r[2] else 0.0
                if n_seg >= n_min and skill >= skill_min:
                    elegibles.add(comp)

            cands_dia = {t: [] for t in ("segura", "arriesgada", "sonador")}
            pendientes = []          # (comp, tier, prob, ok) para actualizar luego

            for row in por_dia[fecha]:
                comp = row["competicion"]
                try:
                    pred = predecir(con, modelo, cfg, row["local"], row["visitante"],
                                    row["neutral"], competicion=comp)
                except KeyError:
                    saltados += 1
                    continue

                gl, gv = row["goles_local"], row["goles_visitante"]
                res = "local" if gl > gv else ("empate" if gl == gv else "visitante")
                x = pred["1x2"]
                b = base.get(comp) or base["__global__"]
                rps[comp][0] += 1
                rps[comp][1] += _rps_1x2(x["local"], x["empate"], x["visitante"], res)
                rps[comp][2] += _rps_1x2(b[0], b[1], b[2], res)

                rec_base = {"local": row["local"], "visitante": row["visitante"],
                            "competicion": comp, "espn_id": row["espn_id"]}
                nivel = pred["confianza"]["nivel"]

                for mercado, seleccion, prob in _pool_de_partido(pred):
                    tier = _tier(prob, cfg)
                    if tier is None:
                        continue
                    ok = seguimiento.evaluar(
                        con, {**rec_base, "mercado": mercado,
                              "seleccion": seleccion}, fecha)
                    if ok is None:
                        continue
                    pendientes.append((comp, tier, prob, ok))
                    if comp in elegibles and not (tier == "segura"
                                                  and nivel not in ("alto", "medio")):
                        cal = _calib_hasta_ahora(acc, comp, tier,
                                                 sesgo.get(tier, 1.0))
                        cands_dia[tier].append({
                            **rec_base, "mercado": mercado, "seleccion": seleccion,
                            "prob": prob, "prob_aj": min(prob * cal, 0.97), "ok": ok,
                        })

                mi, mj, mp = pred["marcadores_top3"][0]
                ok = seguimiento.evaluar(
                    con, {**rec_base, "mercado": "marcador",
                          "seleccion": f"{mi}-{mj}"}, fecha)
                if ok is not None:
                    pendientes.append((comp, "sonador", mp, ok))
                    if comp in elegibles:
                        cal = _calib_hasta_ahora(acc, comp, "sonador",
                                                 sesgo.get("sonador", 1.0))
                        cands_dia["sonador"].append({
                            **rec_base, "mercado": "marcador",
                            "seleccion": f"{mi}-{mj}", "prob": mp,
                            "prob_aj": min(mp * cal, 0.97), "ok": ok,
                        })

            # --- el boleto del día, con los mismos topes que en producción ---
            dia = {"fecha": fecha}
            for tier in ("segura", "arriesgada", "sonador"):
                tope_t = None if tier == "sonador" else tope
                for etiqueta, n, tp in (("corto", n_corto,
                                         None if tier == "sonador"
                                         else max(2, tope // 2)),
                                        ("largo", n_largo, tope_t)):
                    picks = seleccionar(cands_dia[tier], n, tp)
                    if len(picks) < 2:
                        dia[f"{tier}_{etiqueta}"] = None
                        continue
                    dia[f"{tier}_{etiqueta}"] = {
                        "n": len(picks),
                        "acerto": all(k["ok"] for k in picks),
                        "prob": _producto(k["prob"] for k in picks),
                    }
            if any(dia.get(k) for k in dia if k != "fecha"):
                historial.append(dia)

            # --- recién ahora se suma el día a los acumuladores ---
            for comp, tier, prob, ok in pendientes:
                a = acc[comp][tier]
                a[0] += 1
                a[1] += int(ok)
                a[2] += prob

        if verbose:
            print(f"  [{i:3}/{len(cortes)}] {ini} → {fin}  "
                  f"{len(partidos):4} partidos", flush=True)

    if verbose and saltados:
        print(f"  ({saltados} partidos saltados: equipo sin datos en el modelo)")
    return {"acc": acc, "rps": rps, "saltados": saltados, "historial": historial}


def _producto(xs) -> float:
    p = 1.0
    for x in xs:
        p *= x
    return p


def _frecuencias_base(con, antes_de: str) -> dict:
    """Frecuencias históricas local/empate/visitante por liga, previas a la ventana.

    Es la línea base contra la que se mide el skill: si el modelo no le gana a
    "en esta liga gana el local el 46% de las veces", no aporta nada ahí.
    """
    out = {}
    filas = con.execute(
        """SELECT competicion,
                  SUM(goles_local >  goles_visitante) l,
                  SUM(goles_local =  goles_visitante) e,
                  SUM(goles_local <  goles_visitante) v,
                  COUNT(*) n
             FROM partidos WHERE jugado=1 AND fecha < ?
            GROUP BY competicion""", (antes_de,)).fetchall()
    tot = [0, 0, 0]
    for f in filas:
        tot[0] += f["l"] or 0
        tot[1] += f["e"] or 0
        tot[2] += f["v"] or 0
        if (f["n"] or 0) >= 200:            # muestra mínima para una base propia
            n = f["n"]
            out[f["competicion"]] = (f["l"] / n, f["e"] / n, f["v"] / n)
    s = sum(tot) or 1
    out["__global__"] = (tot[0] / s, tot[1] / s, tot[2] / s)
    return out


# --- Agregación y calibración ----------------------------------------------

def agregar(datos: dict, cfg: dict) -> dict:
    """Convierte los contadores crudos en el ranking de ligas."""
    acc, rps = datos["acc"], datos["rps"]
    catalogo = competiciones(cfg)

    # Sesgo GLOBAL de cada tier: sirve de prior. Encoger hacia la media global
    # cruda castigaría a una liga que de verdad es más predecible; encoger hacia
    # "lo declarado corregido por el sesgo del tier" respeta su nivel propio.
    glob: dict = defaultdict(lambda: [0, 0, 0.0])
    for comp, tiers in acc.items():
        for tier, (n, ok, sp) in tiers.items():
            g = glob[tier]
            g[0] += n
            g[1] += ok
            g[2] += sp
    sesgo_tier = {}
    for tier, (n, ok, sp) in glob.items():
        p_real = ok / n if n else 0.0
        p_decl = sp / n if n else 0.0
        sesgo_tier[tier] = (p_real / p_decl) if p_decl else 1.0

    ligas = {}
    for comp in sorted(set(acc) | set(rps)):
        n_r, s_rps, s_base = rps.get(comp, [0, 0.0, 0.0])
        skill = ((s_base - s_rps) / s_base) if s_base else 0.0
        entry = {
            "nombre": catalogo.get(comp, {}).get("nombre", comp),
            "n_partidos": n_r,
            "rps": round(s_rps / n_r, 4) if n_r else None,
            "rps_base": round(s_base / n_r, 4) if n_r else None,
            "skill": round(skill, 4),
        }
        for tier in ("segura", "arriesgada", "sonador"):
            n, ok, sp = acc.get(comp, {}).get(tier, [0, 0, 0.0])
            if not n:
                entry[tier] = {"n": 0}
                continue
            p_real = ok / n
            p_decl = sp / n
            prior = p_decl * sesgo_tier.get(tier, 1.0)
            p_shrunk = (ok + K_SHRINK * prior) / (n + K_SHRINK)
            calib = p_shrunk / p_decl if p_decl else 1.0
            entry[tier] = {
                "n": n, "aciertos": ok,
                "p_real": round(p_real, 4),
                "p_declarada": round(p_decl, 4),
                "calib_cruda": round(p_real / p_decl, 4) if p_decl else None,
                "calib": round(min(max(calib, CALIB_MIN), CALIB_MAX), 4),
            }
        ligas[comp] = entry

    return {
        "historial": resumen_historial(datos.get("historial") or []),
        "global": {t: {"n": g[0], "aciertos": g[1],
                       "p_real": round(g[1] / g[0], 4) if g[0] else None,
                       "p_declarada": round(g[2] / g[0], 4) if g[0] else None,
                       "sesgo": round(sesgo_tier.get(t, 1.0), 4)}
                   for t, g in glob.items()},
        "ligas": ligas,
    }


def resumen_historial(dias: list) -> dict:
    """Cuántos de los boletos multi-liga habrían pegado, día a día.

    Un boleto solo cuenta si tiene al menos 2 patas evaluables; pega si pegan
    TODAS. Los primeros meses de la ventana salen flacos a propósito: hasta que
    una liga no acumula muestra no es elegible, y la calibración es expansiva.
    """
    tot: dict = {}
    for tier in ("segura", "arriesgada", "sonador"):
        for etiqueta in ("corto", "largo"):
            clave = f"{tier}_{etiqueta}"
            jugados = [d[clave] for d in dias if d.get(clave)]
            ganados = [b for b in jugados if b["acerto"]]
            tot[clave] = {
                "jugados": len(jugados),
                "ganados": len(ganados),
                "tasa": round(len(ganados) / len(jugados), 4) if jugados else None,
                "prob_media": (round(sum(b["prob"] for b in jugados) / len(jugados), 4)
                               if jugados else None),
                "patas_media": (round(sum(b["n"] for b in jugados) / len(jugados), 1)
                                if jugados else None),
            }
    return {
        "desde": dias[0]["fecha"] if dias else None,
        "hasta": dias[-1]["fecha"] if dias else None,
        "n_dias": len(dias),
        "totales": tot,
        "dias": dias,
    }


# --- Persistencia -----------------------------------------------------------

def correr(con, cfg: dict, desde: str, hasta: str, cadencia_dias: int = 7,
           salida: Path = RUTA_JSON, verbose: bool = True,
           detalle: bool = False) -> dict:
    if verbose:
        print(f"Walk-forward {desde} → {hasta}, reentrenando cada "
              f"{cadencia_dias} días…")
    datos = evaluar_ventana(con, cfg, desde, hasta, cadencia_dias, verbose)
    res = agregar(datos, cfg)
    res["version"] = 1
    res["generado_el"] = date.today().isoformat()
    res["ventana"] = {"desde": desde, "hasta": hasta,
                      "cadencia_dias": cadencia_dias}
    res["config_modelo"] = dict(cfg["modelo"])
    salida.parent.mkdir(parents=True, exist_ok=True)
    # El detalle día a día son ~250 KB que la interfaz no usa (solo lee los
    # totales) y que se reescriben enteros en cada recalibración. Se guardan solo
    # bajo `--detalle`, para análisis.
    guardar = dict(res)
    if not detalle:
        guardar["historial"] = {k: v for k, v in res["historial"].items()
                                if k != "dias"}
    salida.write_text(json.dumps(guardar, indent=1, ensure_ascii=False))
    if verbose:
        print(f"\nGuardado en {salida}")
    return res


def cargar(ruta: Path = RUTA_JSON) -> dict | None:
    """Lee el JSON de calibración; None si no existe o está corrupto.

    Devolver None es aceptable: la vista multi-liga funciona sin calibrar (con
    factor 1.0 en todas las ligas), solo que sin el filtro de "ligas seguras".
    """
    try:
        return json.loads(ruta.read_text())
    except Exception:
        return None


# --- CLI --------------------------------------------------------------------

def imprimir(res: dict) -> None:
    print("\n" + "=" * 78)
    print("ACIERTO REAL DEL MODELO POR LIGA  (walk-forward, sin fuga de futuro)")
    print("=" * 78)
    for tier, g in res["global"].items():
        if g["n"]:
            print(f"  {tier:<11} {g['aciertos']:>6}/{g['n']:<6} = "
                  f"{100*g['p_real']:5.1f}% real  vs  "
                  f"{100*g['p_declarada']:5.1f}% declarado")
    print("-" * 78)
    print(f"{'liga':<26} {'n':>5} {'real':>6} {'decl':>6} {'calib':>6} {'skill':>6}")
    print("-" * 78)
    filas = [(c, e) for c, e in res["ligas"].items() if e.get("segura", {}).get("n")]
    filas.sort(key=lambda kv: kv[1]["segura"]["calib"], reverse=True)
    for comp, e in filas:
        s = e["segura"]
        print(f"{e['nombre'][:25]:<26} {s['n']:>5} {100*s['p_real']:>5.1f}% "
              f"{100*s['p_declarada']:>5.1f}% {s['calib']:>6.3f} {e['skill']:>6.3f}")
    print("-" * 78)
    print("calib > 1 = el modelo es conservador ahí (acierta más de lo que dice)")
    print("skill     = cuánto le gana al 'siempre gana el local' de esa liga")

    h = res.get("historial") or {}
    if h.get("n_dias"):
        print("\n" + "=" * 78)
        print(f"HISTÓRICO DE LOS BOLETOS MULTI-LIGA  ({h['n_dias']} días, "
              f"{h['desde']} → {h['hasta']})")
        print("=" * 78)
        print(f"{'boleto':<20} {'pegó':>12} {'tasa':>7} {'esperado':>9} {'patas':>6}")
        print("-" * 78)
        for clave, t in h["totales"].items():
            if not t["jugados"]:
                continue
            print(f"{clave:<20} {t['ganados']:>5}/{t['jugados']:<6} "
                  f"{100*t['tasa']:>6.1f}% {100*t['prob_media']:>8.1f}% "
                  f"{t['patas_media']:>6.1f}")
        print("-" * 78)
        print("'esperado' = probabilidad media que el modelo declaraba para ese boleto")


def main():
    ap = argparse.ArgumentParser(
        description="Mide el acierto real del modelo liga por liga.")
    ap.add_argument("--desde", default="2024-07-01",
                    help="inicio de la ventana (por defecto 2 temporadas)")
    ap.add_argument("--hasta", default=None,
                    help="fin de la ventana (por defecto, hoy)")
    ap.add_argument("--cadencia", type=int, default=7,
                    help="días entre reentrenamientos del walk-forward")
    ap.add_argument("--rapido", action="store_true",
                    help="cadencia de 30 días: ~4x más rápido, algo más tosco")
    ap.add_argument("--salida", default=str(RUTA_JSON))
    ap.add_argument("--detalle", action="store_true",
                    help="guarda también el boleto de cada día (~250 KB extra)")
    args = ap.parse_args()

    cfg = cargar_config()
    con = db.conectar()
    db.inicializar(con)
    hasta = args.hasta or date.today().isoformat()
    cad = 30 if args.rapido else args.cadencia
    res = correr(con, cfg, args.desde, hasta, cad, Path(args.salida),
                 detalle=args.detalle)
    imprimir(res)
    con.close()


if __name__ == "__main__":
    main()
