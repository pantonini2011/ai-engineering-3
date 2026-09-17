import asyncio
import logging

from rag.chain import answer_question
from rag.ingest import ingest_documentos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("rag/rag.log", encoding="utf-8")],
)

# Preguntas dentro del "cerebro" (rag/docs/*.md), sobre distintos documentos.
PREGUNTAS_CON_CONTEXTO = [
    "¿Qué componente es el cuello de botella histórico del sistema bajo carga alta, y por qué?",
    "¿Qué pasos hay que seguir para resolver un agotamiento del pool de conexiones a PostgreSQL?",
    "¿Los viernes se puede desplegar a producción?",
    "¿Qué umbral de uso de conexiones de PgBouncer dispara una alerta?",
]

# Pregunta deliberadamente fuera del contexto disponible, para verificar que
# el sistema dice "No lo sé" en vez de inventar una respuesta plausible.
PREGUNTA_SIN_CONTEXTO = "¿Cuál es la política de vacaciones del equipo de guardia?"


async def main(provider: str = "anthropic") -> None:
    print("=== Módulo de ingesta ===")
    ingest_documentos()

    print("\n=== Preguntas con respuesta en el contexto ===")
    for pregunta in PREGUNTAS_CON_CONTEXTO:
        resultado = await answer_question(pregunta, provider=provider)
        print(f"\nPregunta: {pregunta}")
        print(resultado.model_dump_json(indent=2))

    print("\n=== Pregunta fuera del contexto disponible (debe responder 'No lo sé') ===")
    resultado = await answer_question(PREGUNTA_SIN_CONTEXTO, provider=provider)
    print(f"\nPregunta: {PREGUNTA_SIN_CONTEXTO}")
    print(resultado.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
