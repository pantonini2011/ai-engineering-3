from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

NO_CONTEXTO_MENSAJE = "No lo sé, no tengo información sobre eso en el contexto disponible."


class RespuestaLLM(BaseModel):
    """Lo único que el LLM decide realmente: el texto de la respuesta y si
    encontró (o no) información relevante en el CONTEXTO recuperado. Las
    fuentes NO las completa el modelo (ver `RespuestaRAG` más abajo): pedirle
    al LLM que liste nombres de archivo de memoria es una invitación a que
    invente uno plausible pero incorrecto, y Pydantic no tiene forma de
    validar contra la realidad si un nombre de archivo existe o no."""

    respuesta: str = Field(
        min_length=1,
        description=(
            "Respuesta a la pregunta del usuario, basada ÚNICAMENTE en el CONTEXTO "
            "proporcionado. Si el CONTEXTO no contiene la información necesaria, debe "
            f"decir explícitamente '{NO_CONTEXTO_MENSAJE}'."
        ),
    )
    contexto_encontrado: bool = Field(
        description=(
            "True si el CONTEXTO recuperado contenía información relevante para "
            "responder la pregunta; False si la respuesta fue 'No lo sé' (o "
            "equivalente) por falta de información en el CONTEXTO."
        ),
    )

    @field_validator("respuesta")
    @classmethod
    def limpiar_respuesta(cls, v: str) -> str:
        limpio = v.strip()
        if not limpio:
            raise ValueError("La respuesta no puede quedar vacía.")
        return limpio

    @model_validator(mode="after")
    def normalizar_respuesta_sin_contexto(self) -> "RespuestaLLM":
        """El prompt le pide al modelo responder 'literalmente' la frase
        canónica cuando `contexto_encontrado=False`, pero un LLM no siempre
        respeta esa instrucción al pie de la letra -- en la práctica, a veces
        agrega una explicación extra después de un '\\n\\n' (observado con
        Claude en pregunts fuera de contexto), lo que rompe la consistencia
        que el "filtro de veracidad" busca garantizar. Se fuerza acá en
        código, no solo en el prompt: si el modelo marcó que no encontró
        contexto, la `respuesta` queda siempre en el mensaje canónico exacto,
        sin importar qué texto haya generado el modelo."""
        if not self.contexto_encontrado:
            self.respuesta = NO_CONTEXTO_MENSAJE
        return self


class FragmentoRecuperado(BaseModel):
    """Un fragmento que trajo el retriever, con su metadata y su score. Se
    completa en código a partir de lo que devuelve ChromaDB, no lo decide el
    LLM."""

    fuente: str = Field(description="Archivo de origen del fragmento (metadata `source`).")
    seccion: Optional[str] = Field(
        default=None,
        description="Encabezados Markdown de la sección del fragmento (ej. 'Runbook > Incidente 1').",
    )
    similitud: float = Field(description="Similitud coseno con la pregunta (1 - distancia de Chroma); 1.0 = idéntico.")


class RespuestaRAG(BaseModel):
    """Contrato de salida final del sistema RAG, expuesto por `src.chain`.
    Combina lo que el LLM generó (`respuesta`/`contexto_encontrado`, vía
    `RespuestaLLM`) con las fuentes reales de los fragmentos que el retriever
    efectivamente recuperó -- calculadas en código, no por el modelo."""

    pregunta: str = Field(min_length=1, description="Pregunta original del usuario.")
    respuesta: str = Field(min_length=1, description="Respuesta generada, grounded en el contexto recuperado.")
    contexto_encontrado: bool = Field(
        description="True si se usó información real de la base vectorial para responder."
    )
    fuentes: List[str] = Field(
        default_factory=list,
        description="Nombres de archivo de origen de los fragmentos recuperados y usados como contexto.",
    )
    fragmentos_recuperados: List[FragmentoRecuperado] = Field(
        default_factory=list,
        description=(
            "Todos los fragmentos que trajo el retriever (top_k), en orden de similitud, "
            "con su metadata y score -- aunque el modelo no haya encontrado la respuesta en ellos."
        ),
    )
