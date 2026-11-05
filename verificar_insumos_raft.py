"""
verificar_insumos_raft.py
=========================
Regenera y verifica TODOS los insumos del afinamiento con RAFT que se van a citar
en el documento del trabajo de grado (tablas de validacion/reporte_raft/ y figuras).

Que hace:
  1. Regenera el reporte ejecutando analisis_raft.ipynb sobre los datos reales
     (se puede omitir con --solo-verificar).
  2. Verifica, con calculos INDEPENDIENTES de los del notebook, que cada cifra de
     cada tabla coincida con los archivos fuente (checkpoints de RAGAS, respuestas
     generadas, corpus, salidas de Kaggle).
  3. Verifica la integridad de los datos: mismas 196 preguntas, mismos contextos y
     referencias en todos los modelos, que las metricas de RAGAS correspondan a las
     respuestas guardadas, que no haya datos simulados, y que tablas y figuras sean
     mas recientes que sus fuentes.
  4. Deja constancia persistente en validacion/reporte_raft/:
       - verificacion_insumos.md    resumen legible (OK / AVISO / FALLA por chequeo)
       - verificacion_insumos.csv   el mismo resultado en tabla
       - manifiesto_insumos.json    SHA-256, tamano y fecha de cada fuente y salida,
                                    para detectar despues cualquier cambio.

Uso (desde la carpeta "TG - Pruebas 2", con el venv del proyecto):
    python verificar_insumos_raft.py                 # regenera y verifica
    python verificar_insumos_raft.py --solo-verificar
    python verificar_insumos_raft.py --corpus "RUTA/TG - Generacion de corpus"

Codigo de salida: 0 si no hay FALLAS, 1 si hay alguna.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DIR_BASE = Path(__file__).resolve().parent
METRICAS = ["faithfulness", "answer_relevancy", "answer_correctness", "semantic_similarity"]
TOL = 1e-9
N_PREGUNTAS = 196
RAFT, BASE, HAIKU, SONNET = "raft-llama3.1-8b", "llama3.1-8b-base", "claude-haiku-4.5", "claude-sonnet-5"
ESTILOS_LOCALES = ["raft", "estructurado", "directo"]
ETIQ_LLM = {SONNET: "Sonnet 5", HAIKU: "Haiku 4.5", RAFT: "Llama 3.1 8B + RAFT", BASE: "Llama 3.1 8B (base)"}
ETIQ_PROMPT = {"directo": "Directo", "estructurado": "Estructurado", "raft": "Formato RAFT"}


# ─── registro de resultados ──────────────────────────────────────────────────

class Registro:
    def __init__(self):
        self.filas = []

    def _add(self, estado, grupo, chequeo, detalle=""):
        self.filas.append({"Estado": estado, "Grupo": grupo, "Chequeo": chequeo, "Detalle": str(detalle)})
        icono = {"OK": "  OK  ", "AVISO": "AVISO ", "FALLA": "FALLA "}[estado]
        print(f"[{icono}] {grupo} | {chequeo}" + (f" -> {detalle}" if detalle else ""))

    def ok(self, g, c, d=""):
        self._add("OK", g, c, d)

    def aviso(self, g, c, d=""):
        self._add("AVISO", g, c, d)

    def falla(self, g, c, d=""):
        self._add("FALLA", g, c, d)

    def check(self, condicion, g, c, d_ok="", d_falla=""):
        (self.ok if condicion else self.falla)(g, c, d_ok if condicion else (d_falla or d_ok))
        return condicion

    def guardado(self, g, c, funcion):
        """Ejecuta un chequeo y convierte cualquier excepcion en FALLA (no detiene la verificacion)."""
        try:
            funcion()
        except Exception as e:  # noqa: BLE001
            self.falla(g, c, f"error al verificar: {type(e).__name__}: {e}")


R = Registro()


# ─── utilidades ──────────────────────────────────────────────────────────────

def sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def cerca(a, b, tol=TOL) -> bool:
    if (a is None or (isinstance(a, float) and np.isnan(a))) and (b is None or (isinstance(b, float) and np.isnan(b))):
        return True
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, encoding="utf-8-sig")


def leer_json(ruta: Path):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def leer_jsonl(ruta: Path):
    with open(ruta, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm_literal(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[\W_]+", "", re.sub(r"<br\s*/?>", " ", s))


PATRON_ABSTENCION = re.compile(
    r"no se encuentra|no (contiene|incluye|proporciona|menciona|especifica)|no hay informaci|no es posible (determinar|responder)",
    re.IGNORECASE)


def es_degenerado(texto):
    w = (texto or "").split()
    if len(w) < 40:
        return False
    ng = Counter(tuple(w[i:i + 8]) for i in range(len(w) - 8))
    if ng and ng.most_common(1)[0][1] >= 8:
        return True
    return len(w) >= 150 and len(set(w)) / len(w) < 0.20


def holm(p):
    p = np.asarray(p, dtype=float)
    orden, m = np.argsort(p), len(p)
    out, maximo = np.empty(m), 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, (m - rango) * p[i])
        out[i] = min(1.0, maximo)
    return out


# ─── 1. regeneracion ─────────────────────────────────────────────────────────

def regenerar_reporte(ruta_nb: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print(f"\n=== Regenerando el reporte con {ruta_nb.name} ===")
    nb = leer_json(ruta_nb)
    espacio = {"display": lambda *a, **k: None, "__name__": "__main__"}
    cwd = os.getcwd()
    os.chdir(ruta_nb.parent)
    try:
        for i, celda in enumerate(nb["cells"]):
            if celda["cell_type"] != "code":
                continue
            src = "".join(celda["source"])
            if src.lstrip().startswith(("%", "!")):
                continue
            exec(compile(src, f"{ruta_nb.name}[celda {i}]", "exec"), espacio)
            plt.close("all")
    finally:
        os.chdir(cwd)
    print("Reporte regenerado.\n")


# ─── 2. verificacion ─────────────────────────────────────────────────────────

def verificar(dir_pruebas: Path, dir_corpus: Path):
    V = dir_pruebas / "validacion"
    REP = V / "reporte_raft"
    CK_EVAL = V / "checkpoints_evaluacion_generacion"
    SAL = dir_corpus / "raft" / "salida"
    ENT = next((d for d in [dir_corpus / "raft" / "entrenamiento" / "kaggle_output", dir_corpus / "raft" / "entrenamiento"]
                if (d / "log_entrenamiento.json").exists()), dir_corpus / "raft" / "entrenamiento")

    fuentes = {
        "gen_claude": V / "resultados_generacion_llm.json",
        "gen_raft": V / "resultados_generacion_llm_raft.json",
        "gen_base": V / "resultados_generacion_llm_base.json",
        "det_claude": V / "evaluacion_generacion_detalle.csv",
        "det_raft": V / "evaluacion_generacion_detalle_raft.csv",
        "det_base": V / "evaluacion_generacion_detalle_base.csv",
        "res_raft": V / "evaluacion_generacion_resumen_raft.csv",
        "res_base": V / "evaluacion_generacion_resumen_base.csv",
        "val_raft": V / "validacion_respuestas_raft.csv",
        "val_base": V / "validacion_respuestas_base.csv",
        "dataset_val": V / "dataset_validacion_final.json",
        "corpus_1": dir_corpus / "JsonsDatasetByChars" / "dataset_acreditacion_primera_mitad.json",
        "corpus_2": dir_corpus / "JsonsDatasetByChars" / "dataset_acreditacion_segunda_mitad.json",
        "cot_cache": SAL / "cot_cache.jsonl",
        "raft_completo": SAL / "raft_completo.jsonl",
        "raft_train": SAL / "raft_train.jsonl",
        "raft_eval": SAL / "raft_eval.jsonl",
        "reporte_corpus": SAL / "reporte_raft.json",
        "kaggle_config": ENT / "config_entrenamiento.json",
        "kaggle_log": ENT / "log_entrenamiento.json",
    }
    for combo in [f"openai_cs800__{m}__{p}" for m in (SONNET, HAIKU) for p in ("directo", "estructurado")] + \
                 [f"openai_cs800__{m}__{p}" for m in (RAFT, BASE) for p in ESTILOS_LOCALES]:
        fuentes[f"ck_{combo}"] = CK_EVAL / f"{combo}.csv"

    salidas = {n: REP / f"{n}.csv" for n in ["corpus_fuente", "corpus_embudo", "control_calidad_llm", "ejemplos_raft",
                                             "entrenamiento", "perdida", "metricas_generacion", "wilcoxon_raft",
                                             "comportamiento", "por_tipo_evidencia"]}
    salidas["xlsx"] = REP / "reporte_raft.xlsx"
    figuras = {"fig_promedio": REP / "promedio_compuesto_raft_vs_claude.png",
               "fig_perdida": REP / "perdida_entrenamiento.png"}

    # ── A. Archivos ──────────────────────────────────────────────────────────
    G = "A. Archivos"
    faltan = [f"{k}: {v}" for k, v in {**fuentes, **salidas, **figuras}.items() if not v.exists()]
    if not R.check(not faltan, G, "Existen todas las fuentes, tablas y figuras",
                   f"{len(fuentes)} fuentes, {len(salidas)} tablas, {len(figuras)} figuras", "; ".join(faltan)):
        print("\nNo se puede seguir sin los archivos faltantes.")
        return fuentes, salidas, figuras

    # ── carga ────────────────────────────────────────────────────────────────
    gen = {}  # (modelo, estilo) -> {pregunta: resultado}
    meta = {}
    for clave in ("gen_claude", "gen_raft", "gen_base"):
        d = leer_json(fuentes[clave])
        meta[clave] = d.get("generacion", {})
        for c in d["combinaciones"]:
            if c["pipeline_retrieve"] != "openai_cs800":
                continue
            gen[(c["modelo_llm"], c["estilo_prompt"])] = {r["pregunta"]: r for r in c["resultados"]}
    combos = [(m, p) for m in (SONNET, HAIKU) for p in ("directo", "estructurado")] + \
             [(m, p) for m in (RAFT, BASE) for p in ESTILOS_LOCALES]

    ck = {}
    for m, p in combos:
        ck[(m, p)] = leer_csv(fuentes[f"ck_openai_cs800__{m}__{p}"])

    # ── B. Integridad de las generaciones ───────────────────────────────────
    G = "B. Generaciones"
    R.guardado(G, "combinaciones presentes", lambda: R.check(
        all(k in gen for k in combos), G, "Estan las 10 combinaciones openai_cs800 (4 Claude + 3 RAFT + 3 base)",
        "", f"faltan: {[k for k in combos if k not in gen]}"))

    preguntas_ref = set(gen[(HAIKU, "estructurado")])
    R.check(len(preguntas_ref) == N_PREGUNTAS, G, f"Claude tiene {N_PREGUNTAS} preguntas", str(len(preguntas_ref)))
    distintas = {f"{m}/{p}": len(set(gen[(m, p)]) ^ preguntas_ref) for m, p in combos}
    R.check(all(v == 0 for v in distintas.values()), G, "Las 10 combinaciones tienen exactamente las mismas preguntas",
            "", str({k: v for k, v in distintas.items() if v}))

    def _contextos():
        malos = Counter()
        for (m, p) in combos:
            for q, r in gen[(m, p)].items():
                ref = gen[(HAIKU, "estructurado")][q]
                if r.get("contextos_usados") != ref.get("contextos_usados"):
                    malos[f"{m}/{p} contextos"] += 1
                if r.get("respuesta_esperada") != ref.get("respuesta_esperada"):
                    malos[f"{m}/{p} referencia"] += 1
        R.check(not malos, G, "Contextos y respuesta esperada identicos en todos los modelos (por pregunta)",
                "1960 respuestas comparadas", str(dict(malos)))
    R.guardado(G, "contextos identicos", _contextos)

    def _locales():
        for m in (RAFT, BASE):
            vacias = sum(not (r.get("respuesta_generada") or "").strip() for p in ESTILOS_LOCALES for r in gen[(m, p)].values())
            trunc = sum(bool(r.get("contexto_posiblemente_truncado")) for p in ESTILOS_LOCALES for r in gen[(m, p)].values())
            lat = [r.get("latencia_s") or 0 for p in ESTILOS_LOCALES for r in gen[(m, p)].values()]
            tok = [r.get("tokens_prompt") or 0 for p in ESTILOS_LOCALES for r in gen[(m, p)].values()]
            sim = sum("simulad" in (r.get("respuesta_generada_cruda") or "").lower() for p in ESTILOS_LOCALES for r in gen[(m, p)].values())
            R.check(vacias == 0 and trunc == 0, G, f"{ETIQ_LLM[m]}: sin respuestas vacias ni contexto truncado",
                    f"vacias={vacias}, truncadas={trunc}")
            R.check(np.median(lat) > 1.0 and min(tok) > 500 and sim == 0, G,
                    f"{ETIQ_LLM[m]}: datos reales de Ollama (no simulados)",
                    f"latencia mediana {np.median(lat):.1f} s, tokens de entrada min {min(tok)}, textos 'simulado' {sim}")
        mr, mb = meta["gen_raft"], meta["gen_base"]
        iguales = all(mr.get(k) == mb.get(k) for k in ("num_ctx", "max_tokens", "temperature", "seed", "contextos"))
        R.check(iguales and "RAFT" in str(mr.get("modelo")) and "sin afinar" in str(mb.get("modelo")), G,
                "RAFT y base generados con los mismos parametros y el modelo declarado correcto",
                f"num_ctx={mr.get('num_ctx')}, temperature={mr.get('temperature')}, seed={mr.get('seed')}")
    R.guardado(G, "generaciones locales", _locales)

    # ── C. Evaluacion RAGAS ──────────────────────────────────────────────────
    G = "C. RAGAS"

    def _ck_vs_gen():
        problemas = {}
        for (m, p), df in ck.items():
            if not df["user_input"].is_unique or len(df) != N_PREGUNTAS:
                problemas[f"{m}/{p}"] = f"filas={len(df)}, unicas={df['user_input'].nunique()}"
                continue
            g = gen[(m, p)]
            distinta = sum((str(r) if pd.notna(r) else "") != (g[q].get("respuesta_generada") or "")
                           for q, r in zip(df["user_input"], df["response"]))
            if distinta:
                problemas[f"{m}/{p}"] = f"{distinta} respuestas evaluadas no coinciden con las generadas"
        R.check(not problemas, G, "Cada checkpoint de RAGAS evaluo exactamente las respuestas guardadas (10 combinaciones)",
                "196 filas unicas por combinacion; texto evaluado = respuesta_generada", str(problemas))
    R.guardado(G, "checkpoints = generaciones", _ck_vs_gen)

    def _det_vs_ck():
        for clave, modelo in (("det_raft", RAFT), ("det_base", BASE), ("det_claude", None)):
            det = leer_csv(fuentes[clave])
            if modelo is None:
                det = det[det["pipeline_retrieve"] == "openai_cs800"]
            peor = 0.0
            for (m, p), df in ck.items():
                if (modelo and m != modelo) or (modelo is None and m not in (SONNET, HAIKU)):
                    continue
                d = det[(det["modelo_llm"] == m) & (det["estilo_prompt"] == p)].set_index("user_input")[METRICAS]
                c = df.set_index("user_input")[METRICAS].reindex(d.index)
                dif = (d - c).abs().max().max()
                peor = max(peor, 0.0 if np.isnan(dif) else dif)
                assert d.isna().sum().sum() == c.isna().sum().sum(), f"{m}/{p}: nulos distintos"
            R.check(peor <= 1e-12, G, f"{fuentes[clave].name} = checkpoints de RAGAS", f"diferencia maxima {peor:.1e}")
    R.guardado(G, "detalle = checkpoints", _det_vs_ck)

    def _resumen():
        for clave, modelo in (("res_raft", RAFT), ("res_base", BASE)):
            res = leer_csv(fuentes[clave]).set_index("estilo_prompt")
            peor = max(abs(res.loc[p, met] - ck[(modelo, p)][met].mean()) for p in ESTILOS_LOCALES for met in METRICAS)
            R.check(peor <= 1e-12, G, f"{fuentes[clave].name} = promedio de los checkpoints", f"diferencia maxima {peor:.1e}")
    R.guardado(G, "resumen = checkpoints", _resumen)

    nulos = {f"{ETIQ_LLM[m]}/{p}": int(df[METRICAS].isna().sum().sum()) for (m, p), df in ck.items()}
    total_nulos = sum(nulos.values())
    (R.ok if total_nulos <= 15 else R.aviso)(G, "Metricas sin puntaje del juez (se reportan, no se imputan)",
                                             f"{total_nulos} de {len(ck) * N_PREGUNTAS * 4}: " + str({k: v for k, v in nulos.items() if v}))

    # ── D. Corpus y entrenamiento ───────────────────────────────────────────
    G = "D. Corpus y entrenamiento"
    fuente = pd.DataFrame(leer_json(fuentes["corpus_1"]) + leer_json(fuentes["corpus_2"]))
    rep_corpus = leer_json(fuentes["reporte_corpus"])
    cache = {}
    for r in leer_jsonl(fuentes["cot_cache"]):
        cache[r["id"]] = r
    cot = pd.DataFrame(list(cache.values()))
    completo = pd.DataFrame(leer_jsonl(fuentes["raft_completo"]))
    n_train_lines = len(leer_jsonl(fuentes["raft_train"]))
    n_eval_lines = len(leer_jsonl(fuentes["raft_eval"]))
    cfg = leer_json(fuentes["kaggle_config"])
    log = leer_json(fuentes["kaggle_log"])

    def _tabla_concepto(nombre):
        t = leer_csv(salidas[nombre])
        return dict(zip(t.iloc[:, 0].astype(str), t.iloc[:, 1]))

    def _corpus_fuente():
        t = _tabla_concepto("corpus_fuente")
        chunks = fuente.drop_duplicates("contexto")
        ppc = fuente.groupby("contexto").size()
        esperado = {
            "Chunks con al menos un par": len(chunks),
            "Chunks de texto": int((chunks["tipo"] == "texto").sum()),
            "Chunks de tabla": int((chunks["tipo"] == "tabla").sum()),
            "Tokens por chunk (mediana, cl100k)": float(chunks["token_count"].median()),
            "Pares tras la revisión manual": len(fuente),
            "Pares de texto": int((fuente["tipo"] == "texto").sum()),
            "Pares de tabla": int((fuente["tipo"] == "tabla").sum()),
            "Pares con nota del revisor": int(fuente["nota"].notna().sum()) if "nota" in fuente else 0,
            "Chunks con 3 pares": int((ppc == 3).sum()),
        }
        malos = {k: (t.get(k), v) for k, v in esperado.items() if not cerca(t.get(k), v)}
        R.check(not malos, G, "corpus_fuente.csv coincide con los JSON del corpus",
                f"{len(fuente)} pares, {len(chunks)} chunks", str(malos))
    R.guardado(G, "corpus_fuente", _corpus_fuente)

    def _embudo():
        t = leer_csv(salidas["corpus_embudo"]).set_index("Etapa")["Pares"]
        lim = rep_corpus["limpieza_y_exclusion_validacion"]
        ok = cot[cot["estado"] == "ok"]
        solap = int((ok["solapa_evaluacion"].fillna("NO") != "NO").sum())
        esperado = {
            "Pares del corpus fuente (revisados a mano)": len(fuente),
            "Pares enviados al LLM": len(cot),
            "Descartados por el LLM (control de calidad)": -int((cot["estado"] == "descartado").sum()),
            "Fallidos (citas no literales tras 3 intentos)": -int((cot["estado"] == "fallido").sum()),
            "Marcados por el LLM como mismo hecho que validación": -solap,
            "Ejemplos RAFT finales": len(completo),
            "  de entrenamiento": n_train_lines,
            "  de evaluación (monitoreo de la pérdida)": n_eval_lines,
        }
        malos = {k: (t.get(k), v) for k, v in esperado.items() if not cerca(t.get(k), v)}
        cuadra = (len(fuente) - sum(lim.get(k, 0) for k in ("duplicados_mismo_chunk", "excluidas_pregunta_identica",
                  "excluidas_pregunta_similar", "excluidas_mismo_hecho_que_validacion")) == len(cot)
                  and len(cot) - (cot["estado"] != "ok").sum() - solap == len(completo) == n_train_lines + n_eval_lines)
        R.check(not malos and cuadra, G, "corpus_embudo.csv cuadra de punta a punta (corpus → cache → ejemplos → jsonl)",
                f"{len(fuente)} → {len(cot)} → {len(completo)} ({n_train_lines} + {n_eval_lines})", f"{malos}, cuadra={cuadra}")
    R.guardado(G, "embudo", _embudo)

    def _fuga():
        preg_val = {p["pregunta"].strip().lower() for p in leer_json(fuentes["dataset_val"])["pares"]}
        exactas = sum(q.strip().lower() in preg_val for q in completo["pregunta"])
        tr = set(completo.loc[completo["particion"] == "train", "id_chunk_dorado"])
        ev = set(completo.loc[completo["particion"] == "eval", "id_chunk_dorado"])
        R.check(exactas == 0 and not (tr & ev), G, "Sin fuga: ninguna pregunta de validacion en el entrenamiento; train/eval sin chunk dorado comun",
                f"preguntas identicas={exactas}, chunks compartidos={len(tr & ev)}")
    R.guardado(G, "fuga", _fuga)

    def _ejemplos():
        t = leer_csv(salidas["ejemplos_raft"]).set_index("Partición")
        malos = []
        for nombre, sub in [("Entrenamiento", completo[completo["particion"] == "train"]),
                            ("Evaluación", completo[completo["particion"] == "eval"]), ("Total", completo)]:
            for col, v in [("Ejemplos", len(sub)), ("Con documento dorado", int(sub["con_documento_dorado"].sum())),
                           ("Tokens por ejemplo, mediana", int(sub["tokens_aprox"].median())),
                           ("Tokens por ejemplo, máximo", int(sub["tokens_aprox"].max()))]:
                if not cerca(t.loc[nombre, col], v):
                    malos.append(f"{nombre}/{col}: {t.loc[nombre, col]} vs {v}")
        p = completo["con_documento_dorado"].mean()
        R.check(not malos and abs(p - rep_corpus["proporcion_real_con_dorado"]) < 1e-3, G,
                "ejemplos_raft.csv coincide con raft_completo.jsonl", f"P real = {p:.3f}", "; ".join(malos))
    R.guardado(G, "ejemplos", _ejemplos)

    def _entrenamiento():
        t = _tabla_concepto("entrenamiento")
        tr = [x for x in log if "loss" in x and "eval_loss" not in x]
        ev = [x for x in log if "eval_loss" in x]
        fin = next(x for x in log if "train_loss" in x)
        esperado = {
            "Ejemplos de entrenamiento / evaluación": f"{cfg['n_train']} / {cfg['n_eval']}",
            "Tiempo de entrenamiento (min)": cfg["minutos"],
            "Pasos de entrenamiento": fin["step"],
            "Pérdida de entrenamiento inicial / final": f"{tr[0]['loss']:.4f} / {tr[-1]['loss']:.4f}",
            "Pérdida de evaluación primera / última / mínima":
                f"{ev[0]['eval_loss']:.4f} / {ev[-1]['eval_loss']:.4f} / {min(x['eval_loss'] for x in ev):.4f}",
        }
        malos = {k: (t.get(k), v) for k, v in esperado.items() if not cerca(t.get(k), v)}
        R.check(not malos, G, "entrenamiento.csv coincide con config y log de Kaggle",
                f"{cfg['minutos']} min en {cfg['gpu']}, {fin['step']} pasos", str(malos))
        R.check(cfg["n_train"] == n_train_lines and cfg["n_eval"] == n_eval_lines, G,
                "Kaggle entreno con los mismos archivos del corpus (conteos)",
                f"config {cfg['n_train']}/{cfg['n_eval']} vs jsonl {n_train_lines}/{n_eval_lines}")
        per = leer_csv(salidas["perdida"])
        R.check(len(per.dropna(subset=["loss"])) == len(tr) and len(per.dropna(subset=["eval_loss"])) == len(ev), G,
                "perdida.csv contiene todos los puntos del log", f"{len(tr)} de entrenamiento, {len(ev)} de evaluacion")
    R.guardado(G, "entrenamiento", _entrenamiento)

    # ── E. Tablas de resultados (recalculo independiente) ────────────────────
    G = "E. Tablas de resultados"
    prom = {k: df.set_index("user_input")[METRICAS].mean(axis=1) for k, df in ck.items()}

    def _metricas():
        t = leer_csv(salidas["metricas_generacion"])
        malos, filas = [], 0
        for (m, p), df in ck.items():
            fila = t[(t["Modelo"] == ETIQ_LLM[m]) & (t["Prompt"] == ETIQ_PROMPT[p])]
            if len(fila) != 1:
                malos.append(f"falta {ETIQ_LLM[m]}/{p}")
                continue
            filas += 1
            medias = df[METRICAS].mean()
            for met in METRICAS:
                if not cerca(fila[met].iloc[0], medias[met]):
                    malos.append(f"{m}/{p}/{met}")
            if not cerca(fila["Promedio"].iloc[0], medias.mean()):
                malos.append(f"{m}/{p}/Promedio")
        R.check(not malos and filas == 10, G, "metricas_generacion.csv = promedios recalculados de los checkpoints (10 filas)",
                f"{filas} combinaciones", "; ".join(malos))
    R.guardado(G, "metricas", _metricas)

    def _wilcoxon():
        t = leer_csv(salidas["wilcoxon_raft"])
        pares = [((RAFT, "estructurado"), (HAIKU, "estructurado")), ((RAFT, "estructurado"), (SONNET, "estructurado")),
                 ((RAFT, "raft"), (HAIKU, "estructurado")), ((RAFT, "raft"), (SONNET, "estructurado")),
                 ((RAFT, "raft"), (RAFT, "estructurado"))] + [((RAFT, p), (BASE, p)) for p in ESTILOS_LOCALES]
        malos, n = [], 0
        for (ma, pa), (mb, pb) in pares:
            nombre = f"{ETIQ_LLM[ma]} ({ETIQ_PROMPT[pa]}) vs. {ETIQ_LLM[mb]} ({ETIQ_PROMPT[pb]})"
            sub = t[t["Comparación"] == nombre].set_index("Métrica")
            if len(sub) != 5:
                malos.append(f"falta: {nombre}")
                continue
            ps = []
            for met in ["promedio"] + METRICAS:
                a = prom[(ma, pa)] if met == "promedio" else ck[(ma, pa)].set_index("user_input")[met]
                b = prom[(mb, pb)] if met == "promedio" else ck[(mb, pb)].set_index("user_input")[met]
                ab = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
                dif = ab["a"] - ab["b"]
                pval = wilcoxon(ab["a"], ab["b"]).pvalue
                ps.append(pval)
                fila = sub.loc[met]
                if not (cerca(fila["n"], len(ab)) and cerca(fila["Diferencia (A − B)"], dif.mean())
                        and cerca(fila["p"], pval, 1e-6) and cerca(fila["Gana A"], (dif > 0).sum())):
                    malos.append(f"{nombre}/{met}")
                n += 1
            if not all(cerca(x, y, 1e-6) for x, y in zip(sub.loc[["promedio"] + METRICAS, "p Holm (5 métricas)"], holm(ps))):
                malos.append(f"{nombre}/Holm")
        R.check(not malos and n == 40, G, "wilcoxon_raft.csv = Wilcoxon + Holm recalculados (8 comparaciones × 5 pruebas)",
                f"{n} pruebas, incluidas las 3 de RAFT frente a base", "; ".join(malos))
    R.guardado(G, "wilcoxon", _wilcoxon)

    def _comportamiento():
        t = leer_csv(salidas["comportamiento"])
        val = pd.concat([leer_csv(fuentes["val_raft"]), leer_csv(fuentes["val_base"])])
        malos = []
        for (m, p) in combos:
            fila = t[(t["Modelo"] == ETIQ_LLM[m]) & (t["Prompt"] == ETIQ_PROMPT[p])]
            if len(fila) != 1:
                malos.append(f"falta {m}/{p}")
                continue
            fila = fila.iloc[0]
            rs = list(gen[(m, p)].values())
            abst = sum(bool(PATRON_ABSTENCION.search(r.get("respuesta_generada") or "")) for r in rs)
            if not cerca(fila["Abstenciones"], abst):
                malos.append(f"{m}/{p} abstenciones {fila['Abstenciones']} vs {abst}")
            if m in (RAFT, BASE):
                lat = pd.Series([r["latencia_s"] for r in rs])
                if not cerca(fila["Latencia mediana (s)"], round(lat.median(), 1)):
                    malos.append(f"{m}/{p} latencia")
                # fallas recalculadas desde el texto con las mismas reglas de la validacion
                fallas = sum(es_degenerado(r.get("respuesta_generada_cruda")) or bool(r.get("respuesta_cortada_por_longitud"))
                             or (p == "raft" and "##Answer:" not in (r.get("respuesta_generada_cruda") or ""))
                             or (p == "estructurado" and "Respuesta final:" not in (r.get("respuesta_generada_cruda") or ""))
                             for r in rs)
                fallas_csv = int(val[(val["modelo_llm"] == m) & (val["estilo_prompt"] == p)]["falla_modelo"].sum())
                if not (cerca(fila["Fallas del modelo"], fallas) and fallas == fallas_csv):
                    malos.append(f"{m}/{p} fallas tabla={fila['Fallas del modelo']} recalculo={fallas} validacion={fallas_csv}")
                if p == "raft":
                    n_c = n_l = 0
                    for r in rs:
                        ctx = norm_literal(" ".join(r.get("contextos_usados") or []))
                        for q in re.findall(r"##begin_quote##(.*?)##end_quote##", r.get("respuesta_generada_cruda") or "", re.S):
                            n_c += 1
                            n_l += bool(norm_literal(q)) and norm_literal(q) in ctx
                    if str(fila["Citas literales / citas"]).strip() != f"{n_l} / {n_c}":
                        malos.append(f"{m}/raft citas {fila['Citas literales / citas']} vs {n_l} / {n_c}")
        R.check(not malos, G, "comportamiento.csv = recalculo desde las respuestas (abstenciones, fallas, citas, latencia)",
                "10 combinaciones", "; ".join(malos))
    R.guardado(G, "comportamiento", _comportamiento)

    def _por_tipo():
        t = leer_csv(salidas["por_tipo_evidencia"])
        tipo = {q: r.get("tipo_evidencia") for q, r in gen[(RAFT, "raft")].items()}
        malos = []
        for (m, p) in combos:
            fila = t[(t["Modelo"] == ETIQ_LLM[m]) & (t["Prompt"] == ETIQ_PROMPT[p])]
            if len(fila) != 1:
                malos.append(f"falta {m}/{p}")
                continue
            s = prom[(m, p)]
            tp = s.index.map(lambda q: "tabla" if tipo[q] in ("tabla", "texto+tabla") else "texto")
            for k in ("texto", "tabla"):
                if not cerca(fila[f"promedio – {k}"].iloc[0], s[tp == k].mean()):
                    malos.append(f"{m}/{p}/{k}")
        R.check(not malos, G, "por_tipo_evidencia.csv = recalculo por tipo de evidencia", "10 combinaciones", "; ".join(malos))
    R.guardado(G, "por_tipo", _por_tipo)

    def _xlsx():
        hojas = pd.read_excel(salidas["xlsx"], sheet_name=None)
        esperado = {"Corpus_fuente": "corpus_fuente", "Corpus_embudo": "corpus_embudo", "Ejemplos_RAFT": "ejemplos_raft",
                    "Entrenamiento": "entrenamiento", "Metricas_generacion": "metricas_generacion",
                    "Wilcoxon_RAFT": "wilcoxon_raft", "Comportamiento": "comportamiento", "Por_tipo_evidencia": "por_tipo_evidencia"}
        malos = []
        for hoja, csv in esperado.items():
            if hoja not in hojas:
                malos.append(f"falta hoja {hoja}")
                continue
            a, b = hojas[hoja], leer_csv(salidas[csv])
            if a.shape != b.shape:
                malos.append(f"{hoja}: forma {a.shape} vs {b.shape}")
        R.check(not malos, G, "reporte_raft.xlsx tiene las mismas tablas que los CSV", f"{len(hojas)} hojas", "; ".join(malos))
    R.guardado(G, "xlsx", _xlsx)

    # ── F. Figuras y vigencia ────────────────────────────────────────────────
    G = "F. Figuras y vigencia"

    def _figuras():
        from PIL import Image
        for nombre, ruta in figuras.items():
            with Image.open(ruta) as im:
                im.verify()
            with Image.open(ruta) as im:
                ancho, alto = im.size
            R.check(ancho >= 800 and alto >= 300, G, f"{ruta.name} es una imagen valida", f"{ancho}×{alto} px")
    R.guardado(G, "figuras", _figuras)

    def _vigencia():
        mas_nueva_fuente = max(fuentes.values(), key=lambda p: p.stat().st_mtime)
        t_fuente = mas_nueva_fuente.stat().st_mtime
        viejas = [p.name for p in list(salidas.values()) + list(figuras.values()) if p.stat().st_mtime < t_fuente]
        R.check(not viejas, G, "Tablas y figuras son posteriores a la fuente mas reciente",
                f"fuente mas reciente: {mas_nueva_fuente.name} ({datetime.fromtimestamp(t_fuente):%Y-%m-%d %H:%M})",
                f"desactualizadas: {viejas}. Ejecuta sin --solo-verificar para regenerarlas")
        t_fig = figuras["fig_promedio"].stat().st_mtime
        R.check(t_fig >= fuentes["det_base"].stat().st_mtime, G,
                "La figura del promedio compuesto incluye al modelo base (generada despues de su evaluacion)", "")
    R.guardado(G, "vigencia", _vigencia)

    # ── G. Orden de la carpeta ───────────────────────────────────────────────
    G = "G. Orden"
    sobrantes = [p.name for p in [V / "reporte_raft_v2", V / "validacion_respuestas_con_raft.csv",
                                  V / "resultados_generacion_llm_con_raft.json"] if p.exists()]
    if sobrantes:
        R.aviso(G, "Archivos de versiones anteriores que ningun reporte usa (se pueden borrar)", ", ".join(sobrantes))
    else:
        R.ok(G, "Sin archivos sobrantes de versiones anteriores")

    return fuentes, salidas, figuras


# ─── 3. constancia persistente ───────────────────────────────────────────────

def guardar_constancia(dir_rep: Path, fuentes, salidas, figuras, regenerado: bool):
    dir_rep.mkdir(parents=True, exist_ok=True)
    ahora = datetime.now()
    tabla = pd.DataFrame(R.filas)
    tabla.to_csv(dir_rep / "verificacion_insumos.csv", index=False, encoding="utf-8-sig")

    def info(p: Path):
        return {"ruta": str(p), "existe": p.exists(),
                **({"bytes": p.stat().st_size, "modificado": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                    "sha256": sha256(p)} if p.exists() else {})}

    manifiesto = {
        "fecha_verificacion": ahora.isoformat(timespec="seconds"),
        "reporte_regenerado_en_esta_corrida": regenerado,
        "resultado": {"OK": int((tabla["Estado"] == "OK").sum()), "AVISO": int((tabla["Estado"] == "AVISO").sum()),
                      "FALLA": int((tabla["Estado"] == "FALLA").sum())},
        "fuentes": {k: info(v) for k, v in fuentes.items()},
        "tablas": {k: info(v) for k, v in salidas.items()},
        "figuras": {k: info(v) for k, v in figuras.items()},
    }
    (dir_rep / "manifiesto_insumos.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")

    r = manifiesto["resultado"]
    lineas = [f"# Verificacion de insumos RAFT ({ahora:%Y-%m-%d %H:%M})", "",
              f"Reporte regenerado en esta corrida: **{'si' if regenerado else 'no'}**. "
              f"Resultado: **{r['OK']} OK, {r['AVISO']} AVISO, {r['FALLA']} FALLA**.", "",
              "| Estado | Grupo | Chequeo | Detalle |", "|---|---|---|---|"]
    for f in R.filas:
        det = f["Detalle"].replace("|", "/")
        lineas.append(f"| {f['Estado']} | {f['Grupo']} | {f['Chequeo']} | {det} |")
    lineas += ["", "Las huellas SHA-256 de cada fuente, tabla y figura estan en `manifiesto_insumos.json`. "
               "Si una cifra del documento se cuestiona, basta comparar el hash actual con el de esta verificacion."]
    (dir_rep / "verificacion_insumos.md").write_text("\n".join(lineas), encoding="utf-8")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo-verificar", action="store_true", help="No regenera el reporte; solo verifica lo existente.")
    ap.add_argument("--corpus", default=str(DIR_BASE.parent / "TG - Generacion de corpus"),
                    help="Carpeta del proyecto de corpus.")
    args = ap.parse_args()

    dir_corpus = Path(args.corpus)
    if not dir_corpus.exists():
        sys.exit(f"No encuentro la carpeta del corpus: {dir_corpus}. Usa --corpus RUTA.")

    regenerado = False
    if not args.solo_verificar:
        regenerar_reporte(DIR_BASE / "analisis_raft.ipynb")
        regenerado = True

    print("=== Verificando insumos ===")
    fuentes, salidas, figuras = verificar(DIR_BASE, dir_corpus)
    r = guardar_constancia(DIR_BASE / "validacion" / "reporte_raft", fuentes, salidas, figuras, regenerado)
    print(f"\nResultado: {r['OK']} OK, {r['AVISO']} AVISO, {r['FALLA']} FALLA")
    print("Constancia: validacion/reporte_raft/verificacion_insumos.md, .csv y manifiesto_insumos.json")
    sys.exit(1 if r["FALLA"] else 0)


if __name__ == "__main__":
    main()
