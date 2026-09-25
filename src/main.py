import asyncio
import logging
import os
from typing import Optional, Tuple

from src.chain import API_KEYS, answer_question, fallback_por_defecto, proveedor_disponible
from src.ingestion import ingest_documentos

logger = logging.getLogger(__name__)


# Preguntas dentro del "cerebro" (data/sample_docs/*.md), sobre distintos documentos.
PREGUNTAS_CON_CONTEXTO = [
    "¿Qué componente es el cuello de botella histórico del sistema bajo carga alta, y por qué?",
    "¿Qué pasos hay que seguir para resolver un agotamiento del pool de conexiones a PostgreSQL?",
    "¿Los viernes se puede desplegar a producción?",
    "¿Qué umbral de uso de conexiones de PgBouncer dispara una alerta?",
]

# Pregunta deliberadamente fuera del contexto disponible, para verificar que
# el sistema dice "No lo sé" en vez de inventar una respuesta plausible.
PREGUNTA_SIN_CONTEXTO = "¿Cuál es la política de vacaciones del equipo de guardia?"

# Misma pregunta del pool de conexiones, pero con un filtro de metadata que
# restringe la búsqueda a un solo documento: los 4 fragmentos recuperados
# tienen que venir todos de runbook_incidentes.md.
PREGUNTA_CON_FILTRO = PREGUNTAS_CON_CONTEXTO[1]
FILTRO = {"source": "runbook_incidentes.md"}


def resolver_proveedores() -> Tuple[str, str]:
    """Devuelve `(principal, fallback)` para la generación, sin atar el sistema
    a un solo proveedor (mismo criterio de resiliencia que el Módulo 2):

    - `LLM_PROVIDER` elige el principal (default "anthropic") y
      `LLM_FALLBACK_PROVIDER` el de respaldo (default: el otro de
      anthropic/openai, ver `fallback_por_defecto`).
    - Si el principal no tiene API key pero el fallback sí, se usa el
      fallback como principal (con aviso) en vez de cortar.
    - Solo corta si ninguno de los dos está disponible, con un mensaje que
      dice qué API key cargar.
    Un fallback sin API key queda desactivado (`build_chain` avisa)."""
    principal = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    fallback = os.getenv("LLM_FALLBACK_PROVIDER", "").strip().lower() or fallback_por_defecto(principal)
    for nombre, valor in (("LLM_PROVIDER", principal), ("LLM_FALLBACK_PROVIDER", fallback)):
        if valor not in API_KEYS:
            raise SystemExit(f"{nombre}='{valor}' no soportado. Opciones: {', '.join(API_KEYS)}.")

    if proveedor_disponible(principal):
        return principal, fallback
    if proveedor_disponible(fallback):
        logger.warning(
            "El proveedor principal '%s' no tiene %s en .env: se usa '%s'.", principal, API_KEYS[principal], fallback
        )
        return fallback, principal
    raise SystemExit(
        f"Ningún proveedor de generación disponible: cargá {API_KEYS[principal]} "
        f"(LLM_PROVIDER={principal}) o {API_KEYS[fallback]} (LLM_FALLBACK_PROVIDER={fallback}) en .env "
        "(ver .env.example)."
    )


async def main(provider: str = "anthropic", fallback_provider: Optional[str] = None) -> None:
    logger.info("Proveedor de generación: %s (fallback: %s)", provider, fallback_provider or fallback_por_defecto(provider))
    print("=== Módulo de ingesta ===")
    # Se construye una única vez y se reutiliza en cada pregunta (pasándola
    # explícitamente a answer_question) -- sin esto, cada pregunta reabriría
    # su propia conexión a la colección persistida en SQLite innecesariamente.
    vectorstore = ingest_documentos()

    print("\n=== Preguntas con respuesta en el contexto ===")
    for pregunta in PREGUNTAS_CON_CONTEXTO:
        resultado = await answer_question(pregunta, provider=provider, fallback_provider=fallback_provider, vectorstore=vectorstore)
        print(f"\nPregunta: {pregunta}")
        print(resultado.model_dump_json(indent=2))

    print("\n=== Pregunta fuera del contexto disponible (debe responder 'No lo sé') ===")
    resultado = await answer_question(PREGUNTA_SIN_CONTEXTO, provider=provider, fallback_provider=fallback_provider, vectorstore=vectorstore)
    print(f"\nPregunta: {PREGUNTA_SIN_CONTEXTO}")
    print(resultado.model_dump_json(indent=2))

    print(f"\n=== Pregunta con filtro de metadata {FILTRO} ===")
    resultado = await answer_question(PREGUNTA_CON_FILTRO, provider=provider, fallback_provider=fallback_provider, vectorstore=vectorstore, filtro=FILTRO)
    print(f"\nPregunta: {PREGUNTA_CON_FILTRO}")
    print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    # El logging se configura acá (no al importar el módulo) para que rag.log
    # se cree solo al correr el script, no al importar src.main desde los tests.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("rag.log", encoding="utf-8")],
    )
    asyncio.run(main(*resolver_proveedores()))
