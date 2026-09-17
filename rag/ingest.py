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
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

load_dotenv()
logger = logging.getLogger(__name__)

DOCS_DIR = Path(__file__).parent / "docs"
PERSIST_DIRECTORY = str(Path(__file__).parent.parent / "vectorstore")
COLLECTION_NAME = "manuales_tecnicos"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
# Chroma usa L2 al cuadrado por default (no coseno) si no se especifica. Como
# los embeddings ya están normalizados (`normalize_embeddings=True`), el orden
# de similitud da igual con cualquiera de las dos métricas -- pero fijar
# "cosine" explícitamente hace que el score que devuelve Chroma sea
# literalmente `1 - similitud_coseno`, más interpretable que la alternativa
# implícita (`2 - 2·similitud_coseno`).
COLLECTION_METADATA = {"hnsw:space": "cosine"}
# Splitter jerárquico: primero por encabezado Markdown (mantiene una sección
# semántica completa -- ej. un incidente entero, con síntoma/causa/resolución
# -- en un mismo fragmento), y recién para las secciones que sigan siendo
# demasiado largas, RecursiveCharacterTextSplitter como fallback dentro de
# esa sección puntual (nunca mezcla contenido de dos secciones/documentos
# distintos en el mismo fragmento, a diferencia del splitter plano anterior).
HEADERS_A_DIVIDIR = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]


@lru_cache(maxsize=1)
def _build_embeddings() -> Embeddings:
    """Único punto de construcción del modelo de embeddings. Tanto `ingest_documentos`
    (indexación) como `rag.chain` (consulta) importan esta misma función -- así se
    evita a propósito el error #1 que señala la consigna: indexar con un modelo de
    embeddings y consultar con otro distinto, lo que vuelve la distancia vectorial
    inútil sin que nada lo avise en tiempo de ejecución.

    Se usa `HuggingFaceEmbeddings` con
    `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: corre 100%
    local (el modelo se descarga una vez desde el Hub y queda cacheado en
    disco, sin API key ni servidor externo corriendo) -- coherente con el
    título de la consigna ("sistema de recuperación semántica LOCAL"). Se
    eligió la variante *multilingüe* en vez de `all-MiniLM-L6-v2` (entrenado
    casi exclusivamente en inglés) porque todo el "cerebro" documental
    (`rag/docs/`) y las preguntas de prueba están en español -- un modelo
    solo-inglés degrada la similitud coseno ante sinonimia o lenguaje
    coloquial en español. `normalize_embeddings=True` normaliza los vectores
    a norma unitaria antes de guardarlos, que es lo que este modelo espera
    para que la distancia coseno se calcule correctamente.

    `@lru_cache`: a diferencia de un proveedor hosteado (una llamada HTTP
    liviana), `HuggingFaceEmbeddings` carga el modelo de `sentence-transformers`
    en memoria (~2s) cada vez que se instancia. Sin cachear, cada pregunta de
    `answer_question()` -- que llama a `ingest_documentos()`, y esta a
    `_build_embeddings()` -- volvía a cargar el modelo desde cero. Cacheado,
    se carga una única vez por proceso."""
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


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
    texto, y el comportamiento cae directamente al fallback -- mismo
    resultado que el splitter plano anterior para ese caso."""
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
        collection_metadata=COLLECTION_METADATA,
    )
    return vectorstore._collection.count() > 0


def ingest_documentos(
    directorio: Optional[str] = None,
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    force_reindex: bool = False,
) -> Chroma:
    """Módulo de ingesta: fragmenta los documentos de `directorio` (default:
    `rag/docs`) con un splitter jerárquico (encabezados Markdown, con
    `RecursiveCharacterTextSplitter` como fallback dentro de secciones largas
    -- ver `_fragmentar_documento`) y los persiste en una colección de
    ChromaDB en `persist_directory`. Si la colección ya existe y tiene
    documentos, no vuelve a indexar (salvo `force_reindex=True`) -- devuelve
    directamente el `Chroma` apuntando a la colección persistida."""
    directorio_path = Path(directorio) if directorio else DOCS_DIR
    embeddings = _build_embeddings()

    if not force_reindex and _coleccion_ya_poblada(persist_directory, collection_name, embeddings):
        logger.info("Colección '%s' ya poblada en %s: se omite la re-indexación.", collection_name, persist_directory)
        return Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=embeddings,
            collection_metadata=COLLECTION_METADATA,
        )

    documentos = _cargar_documentos(directorio_path)
    if not documentos:
        raise ValueError(f"No se encontraron archivos .txt/.md en '{directorio_path}'.")

    fragmentos = []
    for documento in documentos:
        fragmentos.extend(_fragmentar_documento(documento))
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
        collection_metadata=COLLECTION_METADATA,
    )
    logger.info("Indexación completa: %d fragmentos persistidos.", len(fragmentos))
    return vectorstore
