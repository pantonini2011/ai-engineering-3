import pytest
from pydantic import ValidationError

from rag.schemas import NO_CONTEXTO_MENSAJE, RespuestaLLM, RespuestaRAG


class TestRespuestaLLM:
    def test_acepta_datos_validos(self):
        r = RespuestaLLM(respuesta="El pool de PostgreSQL es el cuello de botella.", contexto_encontrado=True)
        assert r.contexto_encontrado is True

    def test_normaliza_respuesta_cuando_no_hay_contexto(self):
        """El LLM no siempre respeta al pie de la letra la instrucción de
        responder 'literalmente' la frase canónica (se observó a Claude
        agregando una explicación extra después de un '\\n\\n' en una
        pregunta fuera de contexto) -- se fuerza acá, en código, sin
        depender de que el modelo cumpla la instrucción del prompt."""
        r = RespuestaLLM(
            respuesta="No lo sé.\n\nAdemás, el contexto menciona algo relacionado pero no responde esto puntualmente.",
            contexto_encontrado=False,
        )
        assert r.respuesta == NO_CONTEXTO_MENSAJE
        assert "\n" not in r.respuesta

    def test_no_toca_la_respuesta_cuando_hay_contexto(self):
        r = RespuestaLLM(respuesta="El pool de PostgreSQL es el cuello de botella.", contexto_encontrado=True)
        assert r.respuesta == "El pool de PostgreSQL es el cuello de botella."

    def test_rechaza_respuesta_vacia(self):
        with pytest.raises(ValidationError):
            RespuestaLLM(respuesta="", contexto_encontrado=True)

    def test_rechaza_respuesta_solo_espacios(self):
        with pytest.raises(ValidationError):
            RespuestaLLM(respuesta="   ", contexto_encontrado=False)

    def test_limpia_espacios(self):
        r = RespuestaLLM(respuesta="  con espacios de más  ", contexto_encontrado=True)
        assert r.respuesta == "con espacios de más"

    def test_rechaza_campo_faltante(self):
        with pytest.raises(ValidationError):
            RespuestaLLM(respuesta="algo")

    def test_rechaza_tipo_incorrecto_en_contexto_encontrado(self):
        with pytest.raises(ValidationError):
            RespuestaLLM(respuesta="algo", contexto_encontrado="no soy un bool")


class TestRespuestaRAG:
    def test_acepta_datos_validos_con_fuentes(self):
        r = RespuestaRAG(
            pregunta="¿Qué es X?",
            respuesta="X es Y.",
            contexto_encontrado=True,
            fuentes=["doc1.md", "doc2.md"],
        )
        assert r.fuentes == ["doc1.md", "doc2.md"]

    def test_fuentes_default_vacia(self):
        r = RespuestaRAG(pregunta="¿Qué es X?", respuesta="No lo sé.", contexto_encontrado=False)
        assert r.fuentes == []

    def test_rechaza_pregunta_vacia(self):
        with pytest.raises(ValidationError):
            RespuestaRAG(pregunta="", respuesta="algo", contexto_encontrado=True)

    def test_rechaza_respuesta_vacia(self):
        with pytest.raises(ValidationError):
            RespuestaRAG(pregunta="algo", respuesta="", contexto_encontrado=True)
