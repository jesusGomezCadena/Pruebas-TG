#!/usr/bin/env python3
"""
RAG sobre un PDF - variante eficiente SIN OpenAI en el retrieve.

Retrieve: BAAI/bge-m3, hibrido (denso + disperso via RRF), chunk_size=800, k=5.
Generacion: Claude Haiku 4.5, prompt estructurado.
"""

import argparse
import os
import re
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

import pymupdf4llm
from langchain_core.documents import Document
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, SparseVectorParams, Distance, PointStruct, SparseVector,
    Prefetch, FusionQuery, Fusion,
)


# config del pipeline, no tocar sin volver a correr las pruebas
PDF_PATH_DEFAULT = "./documento.pdf"
RUTA_QDRANT = "./qdrant_data_bge_hibrido"
COLECCION = "rag_bge_m3_hibrido_cs800"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 0
K = 5
LIMITE_PREFETCH = 20  # candidatos por rama (denso/disperso) antes de fusionar con RRF
DIMENSION_DENSA = 1024
MODELO_LLM = "anthropic/claude-haiku-4.5"

# usecwd=True busca el .env subiendo desde el directorio actual, no desde
# donde vive el .py - con Path(__file__).parent me estaba fallando en Windows
load_dotenv(find_dotenv(usecwd=True))

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    sys.exit("Falta la variable de entorno OPENROUTER_API_KEY")

cliente = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

print("Cargando BGE-m3 (local, la primera vez baja el modelo)...")
from FlagEmbedding import BGEM3FlagModel
modelo_bge = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)


def con_reintentos(func, intentos=3, espera=5):
    """reintenta con backoff, sin fallback a otro modelo (esto ya es el pipeline
    final, no una comparacion). Solo hace falta para las llamadas a OpenRouter,
    lo de BGE corre local y no tiene sentido reintentar eso."""
    for i in range(intentos):
        try:
            return func()
        except Exception as e:
            if i == intentos - 1:
                raise
            print(f"  fallo ({e}), reintento en {espera}s...")
            time.sleep(espera)
            espera *= 2


# --- extraccion del PDF ---
# misma logica de las pruebas: texto continuo (sin cortar parrafos en el borde
# de pagina) + tablas separadas con su metadata.

PATRON_TABLA_MD = re.compile(
    r'(^\|.*\|\s*\n'
    r'(?:\|[ \t]*:?-+:?[ \t]*\|.*\n)'
    r'(?:\|.*\|\s*\n?)+)',
    re.MULTILINE,
)


def separar_tablas(texto: str, numero_pagina: Optional[int]) -> Tuple[str, List[Dict]]:
    tablas_encontradas = PATRON_TABLA_MD.findall(texto)
    tablas = [{"numero_pagina": numero_pagina, "contenido_md": t.strip()} for t in tablas_encontradas]

    contador = {"i": 0}

    def reemplazo(match):
        contador["i"] += 1
        return f"\n[tabla {contador['i']} de la pagina {numero_pagina}]\n"

    texto_sin_tablas = PATRON_TABLA_MD.sub(reemplazo, texto)
    return texto_sin_tablas, tablas


def extraer_pdf(pdf_path: str):
    paginas = pymupdf4llm.to_markdown(
        pdf_path, page_chunks=True, table_strategy="lines_strict", header=False, footer=False
    )
    if isinstance(paginas, str):
        raise RuntimeError("pymupdf4llm no devolvio paginas separadas, revisa version / pymupdf-layout")

    partes = []
    mapa_paginas = []
    tablas = []
    offset = 0

    for pagina in paginas:
        metadata = pagina.get("metadata", {})
        numero_pagina = metadata.get("page", metadata.get("page_number"))
        texto_sin_tablas, tablas_pagina = separar_tablas(pagina.get("text", ""), numero_pagina)
        tablas.extend(tablas_pagina)

        if partes:
            partes.append(" ")
            offset += 1
        inicio = offset
        partes.append(texto_sin_tablas)
        offset += len(texto_sin_tablas)
        mapa_paginas.append((inicio, offset, numero_pagina))

    for i, t in enumerate(tablas):
        t["indice"] = i

    return "".join(partes), mapa_paginas, tablas


def paginas_de_rango(mapa_paginas, inicio, fin):
    paginas = [n for (a, b, n) in mapa_paginas if a < fin and b > inicio and n is not None]
    return paginas or [None]


def contar_tokens(texto: str) -> int:
    # el tokenizer de BGE-m3 ya esta cargado adentro de modelo_bge, lo reuso
    # en vez de bajar otro modelo aparte solo para contar
    return len(modelo_bge.tokenizer.encode(texto, add_special_tokens=False))


def chunkear(texto_narrativo: str, mapa_paginas, pdf_path: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=contar_tokens,
        add_start_index=True,
    )
    chunks = splitter.create_documents([texto_narrativo])

    documentos = []
    for chunk in chunks:
        inicio = chunk.metadata.get("start_index", 0)
        fin = inicio + len(chunk.page_content)
        paginas = [p for p in paginas_de_rango(mapa_paginas, inicio, fin) if p is not None]
        documentos.append(Document(
            page_content=chunk.page_content,
            metadata={
                "tipo": "texto",
                "pagina_inicio": min(paginas) if paginas else None,
                "pagina_fin": max(paginas) if paginas else None,
                "fuente": pdf_path,
            },
        ))
    return documentos


def documentos_de_tablas(tablas, pdf_path: str) -> List[Document]:
    return [
        Document(
            page_content=t["contenido_md"],
            metadata={"tipo": "tabla", "pagina": t["numero_pagina"], "fuente": pdf_path, "indice_tabla": t["indice"]},
        )
        for t in tablas
    ]


# --- embeddings hibridos (BGE-m3, local) + qdrant ---

def embeber_hibrido(textos: List[str], tam_lote: int = 12):
    """devuelve (vectores_densos, vectores_dispersos). disperso queda como
    lista de dicts {token_id: peso}, listo para armar un SparseVector"""
    if not textos:
        return [], []
    salida = modelo_bge.encode(
        textos, batch_size=tam_lote, max_length=8192,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
    densos = [v.tolist() for v in salida["dense_vecs"]]
    dispersos = [{int(k): v for k, v in lw.items()} for lw in salida["lexical_weights"]]
    return densos, dispersos


def ingestar(pdf_path: str, qdrant: QdrantClient):
    print(f"Extrayendo {pdf_path}...")
    texto, mapa_paginas, tablas = extraer_pdf(pdf_path)

    print("Chunkeando...")
    documentos = chunkear(texto, mapa_paginas, pdf_path) + documentos_de_tablas(tablas, pdf_path)
    print(f"{len(documentos)} documentos (texto + tablas)")

    print("Generando embeddings denso+disperso (local, no factura nada)...")
    densos, dispersos = embeber_hibrido([d.page_content for d in documentos])

    if not qdrant.collection_exists(COLECCION):
        qdrant.create_collection(
            collection_name=COLECCION,
            vectors_config={"dense": VectorParams(size=DIMENSION_DENSA, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )

    puntos = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={"dense": denso, "sparse": SparseVector(indices=list(disperso.keys()), values=list(disperso.values()))},
            payload={"page_content": d.page_content, **d.metadata},
        )
        for d, denso, disperso in zip(documentos, densos, dispersos)
    ]
    qdrant.upsert(collection_name=COLECCION, points=puntos)
    print(f"Listo, {len(puntos)} puntos indexados en '{COLECCION}'.\n")


def recuperar(qdrant: QdrantClient, pregunta: str):
    densos, dispersos = embeber_hibrido([pregunta])
    vector_denso = densos[0]
    disperso = dispersos[0]

    return qdrant.query_points(
        collection_name=COLECCION,
        prefetch=[
            Prefetch(query=vector_denso, using="dense", limit=LIMITE_PREFETCH),
            Prefetch(
                query=SparseVector(indices=list(disperso.keys()), values=list(disperso.values())),
                using="sparse",
                limit=LIMITE_PREFETCH,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=K,
        with_payload=True,
    ).points


# --- generacion ---
# el prompt estructurado gano en las pruebas frente al directo, por eso quedo
# como el default de los pipelines finales

PROMPT_ESTRUCTURADO = """Eres un asistente que responde preguntas basandote ESTRICTAMENTE en el contexto proporcionado, sin usar conocimiento externo. Sigue estos pasos antes de responder:

1. Identifica que fragmentos del contexto son relevantes para la pregunta.
2. Verifica si, con esos fragmentos, la pregunta se puede responder de forma completa, parcial, o no se puede responder.
3. Redacta la respuesta final basandote unicamente en lo que identificaste en el paso 1.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde siguiendo EXACTAMENTE este formato:
Fragmentos relevantes: <cuales y por que, brevemente>
Cobertura: <completa / parcial / nula>
Respuesta final: <tu respuesta aqui; si la cobertura es nula, escribe "No se encuentra en el contexto">"""


def formatear_contexto(hits) -> str:
    return "\n\n".join(f"--- Fragmento {i + 1} ---\n{h.payload['page_content']}" for i, h in enumerate(hits))


def extraer_campo(texto: str, etiqueta: str) -> str:
    m = re.search(rf"{etiqueta}:\s*(.*?)(?:\n[A-ZÁÉÍÓÚ][a-záéíóú ]+:|\Z)", texto, re.DOTALL)
    return m.group(1).strip() if m else ""


def generar(pregunta: str, hits) -> str:
    prompt = PROMPT_ESTRUCTURADO.format(contexto=formatear_contexto(hits), pregunta=pregunta)
    resp = con_reintentos(lambda: cliente.chat.completions.create(
        model=MODELO_LLM,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    ))
    return resp.choices[0].message.content


# --- chat interactivo ---

def main():
    parser = argparse.ArgumentParser(description="RAG sobre un PDF - eficiente sin OpenAI (BGE-m3 hibrido + Claude Haiku 4.5)")
    parser.add_argument("--pdf", default=PDF_PATH_DEFAULT, help="ruta del PDF a consultar")
    parser.add_argument("--reingestar", action="store_true", help="fuerza reindexar aunque la coleccion ya exista")
    args = parser.parse_args()

    qdrant = QdrantClient(path=RUTA_QDRANT)

    if args.reingestar or not qdrant.collection_exists(COLECCION):
        ingestar(args.pdf, qdrant)
    else:
        print(f"Usando coleccion existente '{COLECCION}' (--reingestar para forzar reindexar)\n")

    print("Listo. Escribe tu pregunta (o 'salir' para terminar).\n")

    while True:
        try:
            pregunta = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nChao.")
            break

        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            break

        hits = recuperar(qdrant, pregunta)
        if not hits:
            print("No encontre nada en la coleccion para eso.\n")
            continue

        salida = generar(pregunta, hits)
        cobertura = extraer_campo(salida, "Cobertura")
        respuesta = extraer_campo(salida, "Respuesta final") or salida.strip()

        if cobertura and cobertura.lower() != "completa":
            print(f"[cobertura: {cobertura}]")
        print(respuesta)

        paginas = sorted({h.payload.get("pagina_inicio") or h.payload.get("pagina") for h in hits} - {None})
        if paginas:
            print(f"(paginas: {', '.join(str(p) for p in paginas)})")
        print()

    qdrant.close()


if __name__ == "__main__":
    main()
