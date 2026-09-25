import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, PreTrainedTokenizerBase

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


def _resolver_persist_directory() -> str:
    """`CHROMA_PERSIST_DIR` relativo se resuelve contra la raíz del proyecto
    (no contra el directorio desde donde se corre el script), así
    `./vectorstore` apunta siempre a la misma carpeta."""
    valor = Path(os.getenv("CHROMA_PERSIST_DIR", "vectorstore"))
    return str(valor if valor.is_absolute() else PROJECT_ROOT / valor)


PERSIST_DIRECTORY = _resolver_persist_directory()
# Segmentación por entorno: Chroma no tiene "namespaces" como Pinecone, el
# equivalente es una colección por entorno dentro del mismo persist_directory
# (manuales_tecnicos_dev, manuales_tecnicos_prod). Así una re-indexación de
# prueba en dev no pisa los vectores que consulta prod.
RAG_ENV = os.getenv("RAG_ENV", "dev")
COLLECTION_NAME = f"{os.getenv('CHROMA_COLLECTION', 'manuales_tecnicos')}_{RAG_ENV}"
# Chroma usa L2 al cuadrado por default (no coseno) si no se especifica. Como
# los embeddings ya están normalizados (`normalize_embeddings=True`), el orden
# de similitud da igual con cualquiera de las dos métricas -- pero fijar
# "cosine" explícitamente hace que el score que devuelve Chroma sea
# literalmente `1 - similitud_coseno`, más interpretable que la alternativa
# implícita (`2 - 2·similitud_coseno`).
COLLECTION_METADATA = {"hnsw:space": "cosine"}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@lru_cache(maxsize=1)
def build_embeddings() -> Embeddings:
    """Único punto de construcción del modelo de embeddings. Tanto la ingesta
    (`src.ingestion`) como la consulta (`src.retriever`, a través del `Chroma`
    que devuelve `get_vectorstore`) usan esta misma función -- así se evita a
    propósito el error #1 que señala la consigna: indexar con un modelo de
    embeddings y consultar con otro distinto, lo que vuelve la distancia
    vectorial inútil sin que nada lo avise en tiempo de ejecución.

    Se usa `HuggingFaceEmbeddings` con
    `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: corre 100%
    local (el modelo se descarga una vez desde el Hub y queda cacheado en
    disco, sin API key ni servidor externo corriendo) -- coherente con el
    título de la consigna ("sistema de recuperación semántica LOCAL"). Se
    eligió la variante *multilingüe* en vez de `all-MiniLM-L6-v2` (entrenado
    casi exclusivamente en inglés) porque todo el "cerebro" documental
    (`data/sample_docs/`) y las preguntas de prueba están en español -- un
    modelo solo-inglés degrada la similitud coseno ante sinonimia o lenguaje
    coloquial en español. `normalize_embeddings=True` normaliza los vectores
    a norma unitaria antes de guardarlos, que es lo que este modelo espera
    para que la distancia coseno se calcule correctamente.

    `@lru_cache`: a diferencia de un proveedor hosteado (una llamada HTTP
    liviana), `HuggingFaceEmbeddings` carga el modelo de `sentence-transformers`
    en memoria (~2s) cada vez que se instancia. Cacheado, se carga una única
    vez por proceso."""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    # La dimensión se mide sobre un vector real (no se asume): si se cambia
    # EMBEDDING_MODEL en .env, el log refleja la dimensión del modelo nuevo.
    logger.info("Modelo de embeddings: %s (dimensión %d)", EMBEDDING_MODEL, len(embeddings.embed_query("dimensión")))
    return embeddings


@lru_cache(maxsize=1)
def build_tokenizer() -> PreTrainedTokenizerBase:
    """Tokenizer del mismo modelo de embeddings: el chunking mide el tamaño de
    cada fragmento en los tokens que ve el modelo, no en caracteres. Se
    carga desde el mismo cache local del Hub que `build_embeddings()`."""
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def get_vectorstore(
    persist_directory: str = PERSIST_DIRECTORY,
    collection_name: str = COLLECTION_NAME,
    embeddings: Optional[Embeddings] = None,
) -> Chroma:
    """Conexión a la colección persistida de ChromaDB (la crea vacía si no
    existe), siempre con la métrica coseno y el mismo modelo de embeddings."""
    return Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=embeddings or build_embeddings(),
        collection_metadata=COLLECTION_METADATA,
    )


def coleccion_ya_poblada(persist_directory: str, collection_name: str, embeddings: Embeddings) -> bool:
    """Chequea si la colección ya tiene documentos indexados, para no volver a
    fragmentar y re-embeddear todo en cada corrida (la consigna lo marca
    explícitamente como un error común a evitar: costo y tiempo innecesarios)."""
    if not os.path.isdir(persist_directory):
        return False
    return get_vectorstore(persist_directory, collection_name, embeddings)._collection.count() > 0
