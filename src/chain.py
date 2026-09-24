import logging
import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.ingest import ingest_documentos
from src.schemas import NO_CONTEXTO_MENSAJE, FragmentoRecuperado, RespuestaLLM, RespuestaRAG

load_dotenv()
logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 3
# top_k entre 3 y 5 (recomendado por la consigna): pasar decenas de fragmentos
# al LLM no mejora la respuesta -- degrada la atención del modelo ("lost in
# the middle") y gasta tokens de más para nada.
TOP_K = 4

parser = PydanticOutputParser(pydantic_object=RespuestaLLM)

# Prompt "filtro de veracidad": instruye al modelo a responder solo con el
# CONTEXTO recuperado y a decir explícitamente que no sabe si el contexto no
# alcanza, en vez de completar con conocimiento propio del modelo. Enriquecido
# con el criterio concreto de qué cuenta como "contexto suficiente" y un
# ejemplo de referencia (mismo enfoque que el Módulo 2, cuya corrección pidió
# justamente esto: instrucciones de contenido específicas + un ejemplo, no
# solo la regla general) -- el ejemplo está tomado de un caso real observado
# en este proyecto (ver "Hallazgo" en el README): el retriever puede traer
# fragmentos del tema correcto sin traer el dato puntual pedido.
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente técnico que responde preguntas EXCLUSIVAMENTE en base al "
            "CONTEXTO que se te proporciona a continuación. Nunca uses conocimiento externo "
            "ni completes con información que no esté escrita en el CONTEXTO, aunque te "
            "parezca plausible, la sepas de memoria, o sea una práctica estándar de la "
            "industria para casos similares.\n\n"
            "Criterio para decidir si el CONTEXTO alcanza: no basta con que el CONTEXTO "
            "mencione el mismo tema o sistema de la pregunta -- tiene que contener el dato, "
            "paso o regla PUNTUAL que la pregunta pide. Un CONTEXTO que describe el síntoma y "
            "la causa de un problema, pero no sus pasos de resolución, NO alcanza para "
            "responder 'cómo se resuelve' ese problema, aunque hable del mismo incidente.\n\n"
            "Ejemplo de referencia: si la pregunta es 'cuánto tarda en despacharse un pedido' y "
            "el CONTEXTO solo describe qué componentes procesan un pedido (sin ningún tiempo o "
            "SLA), la respuesta correcta es reconocer que ese dato puntual no está, no "
            "completarlo con un tiempo que suene razonable.\n\n"
            "Fidelidad técnica: cuando el CONTEXTO incluya pasos, comandos, valores numéricos o "
            "nombres de parámetros puntuales, reproducilos tal cual aparecen -- no los resumas "
            "ni los reemplaces por una paráfrasis genérica, aunque conserve el sentido general. "
            "Por ejemplo, si el CONTEXTO dice 'escalar horizontalmente el número de réplicas de "
            "PgBouncer o subir el límite de conexiones a 150', la respuesta debe conservar esos "
            "mismos datos (réplicas de PgBouncer, 150) y no decir simplemente 'escalar el "
            "sistema' -- en un runbook operativo, esa es justo la información que alguien de "
            "guardia necesita para actuar.\n\n"
            "Si el CONTEXTO tiene fragmentos de varios documentos, usá solo los que sean "
            "relevantes a la pregunta -- no mezcles datos de incidentes o reglas distintas como "
            "si fueran una sola respuesta.\n\n"
            "Si el CONTEXTO no contiene información suficiente para responder la pregunta, "
            f"marcá `contexto_encontrado` en `false` y respondé ÚNICAMENTE, sin agregar "
            f"ninguna explicación ni razonamiento antes o después: '{NO_CONTEXTO_MENSAJE}'. "
            "No inventes ni completes con supuestos.\n\n"
            "{format_instructions}",
        ),
        ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
    ]
).partial(format_instructions=parser.get_format_instructions())


class RespuestaIncompletaError(Exception):
    """El LLM cortó la respuesta por límite de tokens, o la salida no pasó el
    parseo a `RespuestaLLM` (JSON mal formado o incompleto). Mismo criterio
    que el Módulo 2."""


def _validar_salida(mensaje: BaseMessage) -> RespuestaLLM:
    """Se ejecuta después del LLM, antes de considerar la respuesta válida.
    Chequea primero el `finish_reason` (OpenAI/Ollama) / `stop_reason`
    (Anthropic) del mensaje crudo -- una respuesta cortada por límite de
    tokens puede, en casos borde, seguir pareciendo JSON válido -- y recién
    después intenta el parseo a `RespuestaLLM` con `PydanticOutputParser`.
    Cada rechazo queda logueado acá (no solo en `answer_question`) porque
    cada uno dispara un reintento de `.with_retry()` -- es el único lugar con
    visibilidad de intento por intento, ya que `answer_question` solo ve el
    resultado final de toda la cadena."""
    finish_reason = mensaje.response_metadata.get("finish_reason") or mensaje.response_metadata.get("stop_reason")
    if finish_reason in ("length", "max_tokens"):
        logger.warning(
            "Validación rechazada (finish_reason=%s): respuesta cortada por límite de tokens; se reintentará.",
            finish_reason,
        )
        raise RespuestaIncompletaError(
            f"El modelo cortó la respuesta por límite de tokens (finish_reason={finish_reason})."
        )

    try:
        resultado = parser.invoke(mensaje)
    except OutputParserException as e:
        logger.warning("Validación rechazada (parsing_error=%s): salida mal formada o incompleta; se reintentará.", e)
        raise RespuestaIncompletaError(f"Salida mal formada o incompleta: {e}") from e

    logger.info("Validación aceptada en este intento.")
    return resultado


def _build_model(provider: str = "anthropic", model: Optional[str] = None) -> BaseChatModel:
    """Mismo criterio de proveedor intercambiable de los Módulos 1 y 2. Default
    `"anthropic"` (Claude): "local" en esta consigna describe a la base
    vectorial (ChromaDB persistida en disco, ver `src/ingest.py`), no al LLM
    de generación -- que perfectamente puede (y en este caso, por default,
    sí) ser un modelo hosteado."""
    provider = provider.lower()
    if provider == "openai":
        return ChatOpenAI(model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    if provider == "anthropic":
        return ChatAnthropic(model=model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"), temperature=0)
    if provider == "ollama":
        kwargs = {"temperature": 0}
        base_url = os.getenv("OLLAMA_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url.removesuffix("/v1")
        return ChatOllama(model=model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b"), **kwargs)
    raise ValueError(f"Proveedor '{provider}' no soportado.")


def _format_docs(docs: List[Document]) -> str:
    if not docs:
        return "(no se recuperó ningún fragmento relevante para esta pregunta)"
    return "\n\n".join(f"[Fuente: {d.metadata.get('source', 'desconocida')}]\n{d.page_content}" for d in docs)


def _seccion(doc: Document) -> Optional[str]:
    """Arma la ruta de encabezados del fragmento ('Header 1 > Header 2 > ...')
    a partir de la metadata que agrega MarkdownHeaderTextSplitter en la
    ingesta. None si el documento no tenía encabezados (ej. un .txt plano)."""
    encabezados = [doc.metadata[h] for h in ("Header 1", "Header 2", "Header 3") if doc.metadata.get(h)]
    return " > ".join(encabezados) or None


def build_chain(provider: str = "anthropic", model: Optional[str] = None) -> Runnable:
    """Cadena LCEL de generación grounded: recibe `{"contexto": ..., "pregunta":
    ...}` (el contexto ya viene de los documentos recuperados por el retriever,
    ver `answer_question`) y produce un `RespuestaLLM` validado.
        PROMPT | llm | RunnableLambda(_validar_salida)
    Envuelta en `.with_retry()` (mismo criterio que el Módulo 2): ante
    `RespuestaIncompletaError` (respuesta truncada, o mal formada/incompleta
    para `PydanticOutputParser`), se reintenta la cadena completa -- un nuevo
    pedido al LLM, no solo un re-parseo del mismo texto -- hasta
    `MAX_RETRY_ATTEMPTS` veces, con backoff exponencial + jitter.
    `retry_if_exception_type` acota el reintento a esa excepción puntual: un
    error de otro tipo (ej. credenciales inválidas, proveedor caído) no vale
    la pena reintentarlo 3 veces con el mismo resultado, y se propaga directo.
    """
    llm = _build_model(provider, model)
    cadena_generacion = PROMPT | llm | RunnableLambda(_validar_salida)
    return cadena_generacion.with_retry(
        retry_if_exception_type=(RespuestaIncompletaError,),
        stop_after_attempt=MAX_RETRY_ATTEMPTS,
        wait_exponential_jitter=True,
    )


async def answer_question(
    pregunta: str,
    provider: str = "anthropic",
    model: Optional[str] = None,
    top_k: int = TOP_K,
    vectorstore: Optional[Chroma] = None,
) -> RespuestaRAG:
    """Punto de entrada end-to-end del RAG: recibe una pregunta en texto plano
    y devuelve un `RespuestaRAG` validado.

    1. **Retrieval**: convierte la pregunta en embedding (mismo modelo que se
       usó para indexar, ver `src.ingest._build_embeddings`) y recupera los
       `top_k` fragmentos más relevantes de ChromaDB.
    2. **Generación grounded**: corre `build_chain()` con esos fragmentos como
       contexto.
    3. Las `fuentes` finales se calculan en código a partir de los documentos
       que el retriever efectivamente trajo -- no las decide el LLM (ver
       `RespuestaLLM` en `src/schemas.py`) -- y quedan vacías si el modelo no
       encontró la respuesta en el contexto, aunque el retriever haya traído
       fragmentos (poco relevantes) igual.
    """
    logger.info("Recuperando contexto para la pregunta: %r", pregunta)
    inicio = time.perf_counter()
    try:
        vectorstore = vectorstore or ingest_documentos()
        # similarity_search_with_score() en vez de as_retriever().ainvoke():
        # el retriever descarta el score, y acá lo necesitamos para loguearlo.
        # Chroma devuelve DISTANCIA (0.0 = más parecido; con la colección
        # configurada a "cosine", ver src/ingest.py, es 1 - similitud_coseno).
        # Se muestra como SIMILITUD (1 - distancia; 1.0/100% = más parecido),
        # más intuitivo para leer en el log.
        docs_con_distancia = await vectorstore.asimilarity_search_with_score(pregunta, k=top_k)
    except Exception:
        duracion = time.perf_counter() - inicio
        logger.exception("Fallo recuperando contexto tras %.2fs (ChromaDB o modelo de embeddings).", duracion)
        raise
    docs = [doc for doc, _ in docs_con_distancia]
    logger.info(
        "Recuperados %d fragmento(s): %s",
        len(docs),
        [
            f"{d.metadata.get('source')} (similitud={1 - distancia:.2%})"
            for d, distancia in docs_con_distancia
        ],
    )
    for d, distancia in docs_con_distancia:
        similitud_pct = (1 - distancia) * 100
        logger.debug(
            "CHUNK COMPLETO [%s, similitud=%.2f%%]:\n%s\n%s",
            d.metadata.get("source"), similitud_pct, "-" * 60, d.page_content,
        )

    contexto = _format_docs(docs)
    chain = build_chain(provider=provider, model=model)
    try:
        resultado_llm: RespuestaLLM = await chain.ainvoke({"contexto": contexto, "pregunta": pregunta})
    except Exception:
        duracion = time.perf_counter() - inicio
        logger.exception("Fallo generando la respuesta grounded tras %.2fs (proveedor=%s)", duracion, provider)
        raise

    fuentes = sorted({d.metadata["source"] for d in docs}) if resultado_llm.contexto_encontrado else []
    fragmentos = [
        FragmentoRecuperado(
            fuente=d.metadata.get("source", "desconocida"),
            seccion=_seccion(d),
            similitud=round(1 - distancia, 4),
        )
        for d, distancia in docs_con_distancia
    ]
    respuesta = RespuestaRAG(
        pregunta=pregunta,
        respuesta=resultado_llm.respuesta,
        contexto_encontrado=resultado_llm.contexto_encontrado,
        fuentes=fuentes,
        fragmentos_recuperados=fragmentos,
    )
    duracion = time.perf_counter() - inicio
    logger.info(
        "Respuesta generada en %.2fs (contexto_encontrado=%s, fuentes=%s)",
        duracion,
        respuesta.contexto_encontrado,
        fuentes,
    )
    return respuesta
