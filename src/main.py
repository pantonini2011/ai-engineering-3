import asyncio
import logging
import os

from src.chain import answer_question
from src.ingestion import ingest_documentos

logger = logging.getLogger(__name__)

# Proveedor de generación -> variable con su API key (Ollama corre local, sin key).
PROVEEDORES = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "ollama": None}

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


def proveedor_configurado() -> str:
    """Lee `LLM_PROVIDER` del entorno (default "anthropic") y verifica, antes
    de indexar nada, que su API key esté cargada en `.env` -- así quien corre
    el script con otra clave (ej. solo OpenAI) recibe un mensaje claro en vez
    de un error de autenticación a mitad de la corrida."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").strip().lower()
    if provider not in PROVEEDORES:
        raise SystemExit(f"LLM_PROVIDER='{provider}' no soportado. Opciones: {', '.join(PROVEEDORES)}.")
    variable = PROVEEDORES[provider]
    valor = os.getenv(variable, "").strip() if variable else None
    # Placeholders de .env.example (el actual "tu_..." y el viejo "sk-...").
    if variable and (not valor or valor.startswith("tu_") or "..." in valor):
        raise SystemExit(f"LLM_PROVIDER={provider} requiere {variable} en .env (ver .env.example).")
    return provider


async def main(provider: str = "anthropic") -> None:
    logger.info("Proveedor de generación: %s", provider)
    print("=== Módulo de ingesta ===")
    # Se construye una única vez y se reutiliza en cada pregunta (pasándola
    # explícitamente a answer_question) -- sin esto, cada pregunta reabriría
    # su propia conexión a la colección persistida en SQLite innecesariamente.
    vectorstore = ingest_documentos()

    print("\n=== Preguntas con respuesta en el contexto ===")
    for pregunta in PREGUNTAS_CON_CONTEXTO:
        resultado = await answer_question(pregunta, provider=provider, vectorstore=vectorstore)
        print(f"\nPregunta: {pregunta}")
        print(resultado.model_dump_json(indent=2))

    print("\n=== Pregunta fuera del contexto disponible (debe responder 'No lo sé') ===")
    resultado = await answer_question(PREGUNTA_SIN_CONTEXTO, provider=provider, vectorstore=vectorstore)
    print(f"\nPregunta: {PREGUNTA_SIN_CONTEXTO}")
    print(resultado.model_dump_json(indent=2))

    print(f"\n=== Pregunta con filtro de metadata {FILTRO} ===")
    resultado = await answer_question(PREGUNTA_CON_FILTRO, provider=provider, vectorstore=vectorstore, filtro=FILTRO)
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
    asyncio.run(main(proveedor_configurado()))
