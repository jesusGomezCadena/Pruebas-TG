#!/usr/bin/env python3
"""
RAG sobre un PDF - configuracion de mayor calidad (Claude Sonnet 5).

Retrieve: OpenAI text-embedding-3-large, denso, chunk_size=800 tokens, k=5.
Generacion: Claude Sonnet 5, prompt estructurado.

Por que Sonnet (generacion, 196 preguntas, Wilcoxon pareado con Holm, ver
secciones 9.5 y 9.6 del documento):
  - Gana en semantic_similarity (p Holm = 0.0088) y queda arriba, sin
    significancia, en answer_correctness (p = 0.055).
  - Pierde en answer_relevancy frente a Haiku 4.5 (p Holm < 0.0001) y empata
    en el promedio compuesto y en faithfulness.
Es la opcion cuando importa mas la cercania a la respuesta de referencia que
el costo. La version barata es rag_eficiente.py (solo cambia MODELO_LLM).

Si la coleccion de Qdrant no existe todavia, el script indexa el PDF antes de
abrir el chat. Si ya existe, se salta directo a preguntar.

pip install pymupdf4llm langchain-text-splitters langchain-core tiktoken openai qdrant-client python-dotenv

Uso:
    # OPENROUTER_API_KEY en un .env junto al script, o exportada:
    export OPENROUTER_API_KEY=...
    python rag_mejor_calidad.py --pdf ./mi_documento.pdf
    python rag_mejor_calidad.py --reingestar   # fuerza reindexar
"""

import argparse
import os
import re
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

import pymupdf4llm
import tiktoken
from langchain_core.documents import Document
from pathlib import Path
from dotenv import load_dotenv

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct


# config del pipeline, no tocar sin volver a correr las pruebas
PDF_PATH_DEFAULT = "./documento.pdf"
RUTA_QDRANT = "./qdrant_data_produccion"
COLECCION = "rag_openai_cs800"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 0
K = 5
MODELO_EMBEDDING = "openai/text-embedding-3-large"
DIMENSION_EMBEDDING = 3072
MODELO_LLM = "anthropic/claude-sonnet-5"

current_dir = Path(__file__).parent
env_path = current_dir / '.env'

load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    sys.exit("Falta la variable de entorno OPENROUTER_API_KEY")

cliente = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
enc = tiktoken.get_encoding("cl100k_base")


def con_reintentos(func, intentos=3, espera=5):
    """reintenta con backoff, sin fallback a otro modelo (aqui no aplica ese problema
    porque esto ya no es una comparacion, es el pipeline final)"""
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
        # pasa si hay pymupdf-layout instalado y pisa el comportamiento de page_chunks
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
    return len(enc.encode(texto))


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


# --- embeddings + qdrant ---

def embeber(textos: List[str], tam_lote: int = 96) -> List[List[float]]:
    if not textos:
        return []
    vectores = []
    for i in range(0, len(textos), tam_lote):
        lote = textos[i:i + tam_lote]
        resp = con_reintentos(lambda: cliente.embeddings.create(model=MODELO_EMBEDDING, input=lote))
        vectores.extend([d.embedding for d in resp.data])
    return vectores


def ingestar(pdf_path: str, qdrant: QdrantClient):
    print(f"Extrayendo {pdf_path}...")
    texto, mapa_paginas, tablas = extraer_pdf(pdf_path)

    print("Chunkeando...")
    documentos = chunkear(texto, mapa_paginas, pdf_path) + documentos_de_tablas(tablas, pdf_path)
    print(f"{len(documentos)} documentos (texto + tablas)")

    print("Generando embeddings (esto es lo que factura OpenRouter)...")
    vectores = embeber([d.page_content for d in documentos])

    if not qdrant.collection_exists(COLECCION):
        qdrant.create_collection(
            collection_name=COLECCION,
            vectors_config=VectorParams(size=DIMENSION_EMBEDDING, distance=Distance.COSINE),
        )

    puntos = [
        PointStruct(id=str(uuid.uuid4()), vector=v, payload={"page_content": d.page_content, **d.metadata})
        for d, v in zip(documentos, vectores)
    ]
    qdrant.upsert(collection_name=COLECCION, points=puntos)
    print(f"Listo, {len(puntos)} puntos indexados en '{COLECCION}'.\n")


def recuperar(qdrant: QdrantClient, pregunta: str):
    vector = embeber([pregunta])[0]
    return qdrant.query_points(collection_name=COLECCION, query=vector, limit=K, with_payload=True).points


# --- generacion ---
# el prompt estructurado gano en las pruebas frente al directo, por eso quedo
# como el default de los dos pipelines finales

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
    parser = argparse.ArgumentParser(description="RAG sobre un PDF - mejor calidad (Claude Sonnet 5)")
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
