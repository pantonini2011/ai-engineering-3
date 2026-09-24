import logging
from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.schemas import FragmentoRecuperado

logger = logging.getLogger(__name__)

# top_k entre 3 y 5 (recomendado por la consigna): pasar decenas de fragmentos
# al LLM no mejora la respuesta -- degrada la atención del modelo ("lost in
# the middle") y gasta tokens de más para nada.
TOP_K = 4
LARGO_EXTRACTO = 120


async def buscar_fragmentos(
    vectorstore: Chroma,
    pregunta: str,
    top_k: int = TOP_K,
    filtro: Optional[dict] = None,
) -> List[Tuple[Document, float]]:
    """Búsqueda semántica: convierte la pregunta en embedding (el `Chroma`
    ya trae el mismo modelo con el que se indexó, ver `src.client`) y
    devuelve los `top_k` fragmentos más parecidos, cada uno con su
    **similitud** coseno (1.0 = idéntico).

    `filtro` restringe la búsqueda por metadata antes de rankear, con la
    sintaxis `where` de Chroma. Ej.: `{"source": "runbook_incidentes.md"}`
    busca solo en ese documento; `{"env": "prod"}` solo en fragmentos
    indexados en prod.

    Se usa `similarity_search_with_score()` en vez de `as_retriever()`: el
    retriever descarta el score. Chroma devuelve DISTANCIA (0.0 = más
    parecido; con la colección en "cosine" es `1 - similitud_coseno`); acá se
    convierte a similitud porque es más intuitivo de leer."""
    docs_con_distancia = await vectorstore.asimilarity_search_with_score(pregunta, k=top_k, filter=filtro)
    resultados = [(doc, 1 - distancia) for doc, distancia in docs_con_distancia]
    logger.info(
        "Recuperados %d fragmento(s)%s: %s",
        len(resultados),
        f" con filtro {filtro}" if filtro else "",
        [f"{d.metadata.get('source')} (similitud={similitud:.2%})" for d, similitud in resultados],
    )
    for d, similitud in resultados:
        logger.debug(
            "CHUNK COMPLETO [%s, similitud=%.2f%%]:\n%s\n%s",
            d.metadata.get("source"), similitud * 100, "-" * 60, d.page_content,
        )
    return resultados


def seccion(doc: Document) -> Optional[str]:
    """Arma la ruta de encabezados del fragmento ('Header 1 > Header 2 > ...')
    a partir de la metadata que agrega MarkdownHeaderTextSplitter en la
    ingesta. None si el documento no tenía encabezados (ej. un .txt plano)."""
    encabezados = [doc.metadata[h] for h in ("Header 1", "Header 2", "Header 3") if doc.metadata.get(h)]
    return " > ".join(encabezados) or None


def a_fragmentos_recuperados(resultados: List[Tuple[Document, float]]) -> List[FragmentoRecuperado]:
    """Convierte el resultado de `buscar_fragmentos` al esquema de salida:
    metadata relevante, score y el comienzo del texto de cada chunk."""
    fragmentos = []
    for d, similitud in resultados:
        texto = " ".join(d.page_content.split())
        fragmentos.append(
            FragmentoRecuperado(
                fuente=d.metadata.get("source", "desconocida"),
                seccion=seccion(d),
                similitud=round(similitud, 4),
                extracto=texto if len(texto) <= LARGO_EXTRACTO else texto[:LARGO_EXTRACTO] + "...",
                created_at=d.metadata.get("created_at"),
                env=d.metadata.get("env"),
            )
        )
    return fragmentos
