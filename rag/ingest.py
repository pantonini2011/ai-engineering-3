import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "docs"
PERSIST_DIRECTORY = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "manuales_tecnicos"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@lru_cache(maxsize=1)
def _build_embeddings() -> Embeddings:
    """Único punto de construcción del modelo de embeddings. Tanto `ingest_documentos`
    (indexación) como `rag.chain` (consulta) importan esta misma función -- así se
    evita a propósito el error #1 que señala la consigna: indexar con un modelo de
    embeddings y consultar con otro distinto, lo que vuelve la distancia vectorial
    inútil sin que nada lo avise en tiempo de ejecución.

    Se usa `HuggingFaceEmbeddings` con `sentence-transformers/all-MiniLM-L6-v2`:
    corre 100% local (el modelo se descarga una vez desde el Hub y queda
    cacheado en disco, sin API key ni servidor externo corriendo) -- coherente
    con el título de la consigna ("sistema de recuperación semántica LOCAL").

    `@lru_cache`: a diferencia de un proveedor hosteado (una llamada HTTP
    liviana), `HuggingFaceEmbeddings` carga el modelo de `sentence-transformers`
    en memoria (~2s) cada vez que se instancia. Sin cachear, cada pregunta de
    `answer_question()` -- que llama a `ingest_documentos()`, y esta a
    `_build_embeddings()` -- volvía a cargar el modelo desde cero. Cacheado,
    se carga una única vez por proceso."""
    return HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))


def _cargar_documentos(directorio: Path) -> list[Document]:
    """Carga todos los .txt/.md de `directorio` como `Document` de LangChain,
    con `metadata={"source": <nombre de archivo>}` -- es la metadata que
    `rag.chain` usa después para reportar `fuentes` sin depender de que el LLM
    las recuerde o las invente."""
    documentos = []
    for path in sorted(directorio.glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        texto = path.read_text(encoding="utf-8")
        documentos.append(Document(page_content=texto, metadata={"source": path.name}))
    return documentos


def _coleccion_ya_poblada(persist_directory: str, collection_name: str, embeddings: Embeddings) -> bool:
    """Chequea si la colección ya tiene documentos indexados, para no volver a
    fragmentar y re-embeddear todo en cada corrida (la consigna lo marca
    explícitamente como un error común a evitar: costo y tiempo innecesarios)."""
    if not os.path.isdir(persist_directory):
        return False
    vectorstore = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    return vectorstore._collection.count() > 0


def ingest_documentos(
    directorio: Optional[str] = None,
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    force_reindex: bool = False,
) -> Chroma:
    """Módulo de ingesta: fragmenta los documentos de `directorio` (default:
    `rag/docs`) con `RecursiveCharacterTextSplitter` y los persiste en una
    colección de ChromaDB en `persist_directory`. Si la colección ya existe y
    tiene documentos, no vuelve a indexar (salvo `force_reindex=True`) --
    devuelve directamente el `Chroma` apuntando a la colección persistida."""
    directorio_path = Path(directorio) if directorio else DOCS_DIR
    embeddings = _build_embeddings()

    if not force_reindex and _coleccion_ya_poblada(persist_directory, collection_name, embeddings):
        logger.info("Colección '%s' ya poblada en %s: se omite la re-indexación.", collection_name, persist_directory)
        return Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings,
        )

    documentos = _cargar_documentos(directorio_path)
    if not documentos:
        raise ValueError(f"No se encontraron archivos .txt/.md en '{directorio_path}'.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    fragmentos = splitter.split_documents(documentos)
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
    )
    logger.info("Indexación completa: %d fragmentos persistidos.", len(fragmentos))
    return vectorstore
