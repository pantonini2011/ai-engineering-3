import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src import client

logger = logging.getLogger(__name__)

DOCS_DIR = client.PROJECT_ROOT / "data" / "sample_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# Splitter jerárquico: primero por encabezado Markdown (mantiene una sección
# semántica completa -- ej. un incidente entero, con síntoma/causa/resolución
# -- en un mismo fragmento), y recién para las secciones que sigan siendo
# demasiado largas, RecursiveCharacterTextSplitter como fallback dentro de
# esa sección puntual (nunca mezcla contenido de dos secciones/documentos
# distintos en el mismo fragmento, a diferencia del splitter plano anterior).
HEADERS_A_DIVIDIR = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


def _cargar_documentos(directorio: Path) -> list[Document]:
    """Carga todos los .txt/.md de `directorio` como `Document` de LangChain,
    con `metadata={"source": <nombre de archivo>}` -- es la metadata que
    `src.chain` usa después para reportar `fuentes` sin depender de que el LLM
    las recuerde o las invente, y la que permite filtrar la búsqueda por
    documento (ver `src.retriever`)."""
    documentos = []
    for path in sorted(directorio.glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        texto = path.read_text(encoding="utf-8")
        documentos.append(Document(page_content=texto, metadata={"source": path.name}))
    return documentos


def _fragmentar_documento(documento: Document) -> list[Document]:
    """Fragmenta un `Document` en dos pasos:

    1. `MarkdownHeaderTextSplitter`: divide por encabezado (`#`/`##`/`###`),
       manteniendo cada sección semántica completa en un solo fragmento --
       ej. "## Incidente 1" queda junto con su síntoma, causa raíz Y
       resolución, en vez de que un corte ciego por cantidad de caracteres
       los separe en fragmentos distintos (ver "Hallazgo" en el README: ese
       fue exactamente el problema real observado con el splitter plano).
       `strip_headers=False` deja el título de la sección adentro del
       `page_content` (útil como contexto para el LLM, no solo en metadata).
    2. `RecursiveCharacterTextSplitter` como *fallback*, solo para las
       secciones que sigan superando `CHUNK_SIZE` después del paso 1 --
       nunca mezcla contenido de dos secciones (o documentos) distintos en
       el mismo fragmento, porque corta *dentro* de una sección ya acotada,
       no sobre el texto completo del documento.

    Si el archivo no tiene encabezados Markdown (ej. un `.txt` plano),
    `MarkdownHeaderTextSplitter` devuelve una única "sección" con todo el
    texto, y el comportamiento cae directamente al fallback."""
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_A_DIVIDIR, strip_headers=False)
    secciones = header_splitter.split_text(documento.page_content)

    fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    fragmentos = []
    for seccion in secciones:
        metadata = {**documento.metadata, **seccion.metadata}
        if len(seccion.page_content) <= CHUNK_SIZE:
            fragmentos.append(Document(page_content=seccion.page_content, metadata=metadata))
        else:
            for texto in fallback_splitter.split_text(seccion.page_content):
                fragmentos.append(Document(page_content=texto, metadata=metadata))
    return fragmentos


def ingest_documentos(
    directorio: Optional[str] = None,
    persist_directory: str = client.PERSIST_DIRECTORY,
    collection_name: str = client.COLLECTION_NAME,
    force_reindex: bool = False,
) -> Chroma:
    """Módulo de ingesta: fragmenta los documentos de `directorio` (default:
    `data/sample_docs/`) con un splitter jerárquico (encabezados Markdown, con
    `RecursiveCharacterTextSplitter` como fallback dentro de secciones largas
    -- ver `_fragmentar_documento`) y los persiste en una colección de
    ChromaDB en `persist_directory`. Si la colección ya existe y tiene
    documentos, no vuelve a indexar (salvo `force_reindex=True`) -- devuelve
    directamente el `Chroma` apuntando a la colección persistida.

    Cada fragmento se guarda con este esquema de metadatos:
    - `source`: archivo de origen (fuentes de la respuesta, filtro por documento).
    - `Header 1`/`Header 2`/`Header 3`: encabezados Markdown de la sección.
    - `env`: entorno (`RAG_ENV`) en el que se indexó.
    - `created_at`: fecha y hora UTC (ISO 8601) de la indexación.
    El texto del fragmento no va en la metadata: Chroma lo guarda como el
    documento en sí (`page_content`)."""
    directorio_path = Path(directorio) if directorio else DOCS_DIR
    embeddings = client.build_embeddings()

    if not force_reindex and client.coleccion_ya_poblada(persist_directory, collection_name, embeddings):
        logger.info("Colección '%s' ya poblada en %s: se omite la re-indexación.", collection_name, persist_directory)
        return client.get_vectorstore(persist_directory, collection_name, embeddings)

    documentos = _cargar_documentos(directorio_path)
    if not documentos:
        raise ValueError(f"No se encontraron archivos .txt/.md en '{directorio_path}'.")

    if force_reindex and client.coleccion_ya_poblada(persist_directory, collection_name, embeddings):
        # Chroma.from_documents agrega sobre la colección existente: sin
        # borrarla antes, re-indexar duplicaría todos los fragmentos.
        logger.info("force_reindex=True: se borra la colección '%s' antes de re-indexar.", collection_name)
        client.get_vectorstore(persist_directory, collection_name, embeddings).delete_collection()

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fragmentos = []
    for documento in documentos:
        for fragmento in _fragmentar_documento(documento):
            fragmento.metadata.update(env=client.RAG_ENV, created_at=created_at)
            fragmentos.append(fragmento)
    logger.info(
        "Indexando %d fragmentos de %d documento(s) en la colección '%s' (%s)...",
        len(fragmentos),
        len(documentos),
        collection_name,
        persist_directory,
    )

    vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
        collection_metadata=client.COLLECTION_METADATA,
    )
    logger.info("Indexación completa: %d fragmentos persistidos.", len(fragmentos))
    return vectorstore
