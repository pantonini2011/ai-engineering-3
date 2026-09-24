import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src import chain as chain_module
from src.schemas import RespuestaLLM


def _fake_llm(responses):
    return FakeListChatModel(responses=responses)


class FakeVectorstore:
    def __init__(self, docs):
        # score de similitud ficticio (1 - similitud_coseno), no relevante
        # para lo que prueban estos tests -- solo importan los documentos.
        self._docs_con_score = [(doc, 0.1 * i) for i, doc in enumerate(docs)]

    async def asimilarity_search_with_score(self, query, k=None):
        return self._docs_con_score


class FakeChain:
    def __init__(self, resultado):
        self._resultado = resultado

    async def ainvoke(self, inputs):
        return self._resultado


class FakeVectorstoreRoto:
    async def asimilarity_search_with_score(self, query, k=None):
        raise ConnectionError("ChromaDB no disponible")


class TestFormatDocs:
    def test_formatea_documentos_con_fuente(self):
        docs = [Document(page_content="Hola mundo", metadata={"source": "a.md"})]
        texto = chain_module._format_docs(docs)
        assert "[Fuente: a.md]" in texto
        assert "Hola mundo" in texto

    def test_lista_vacia(self):
        texto = chain_module._format_docs([])
        assert "no se recuperó" in texto


class TestBuildModel:
    def test_provider_desconocido_lanza_error(self):
        with pytest.raises(ValueError):
            chain_module._build_model("cohere")

    def test_anthropic_por_default(self):
        assert chain_module._build_model().__class__.__name__ == "ChatAnthropic"

    def test_openai(self):
        assert chain_module._build_model("openai").__class__.__name__ == "ChatOpenAI"


class TestValidarSalida:
    def test_acepta_salida_completa(self):
        mensaje = AIMessage(
            content='{"respuesta": "El pool de PostgreSQL.", "contexto_encontrado": true}',
            response_metadata={"finish_reason": "stop"},
        )
        resultado = chain_module._validar_salida(mensaje)
        assert isinstance(resultado, RespuestaLLM)
        assert resultado.contexto_encontrado is True

    def test_rechaza_finish_reason_length(self):
        mensaje = AIMessage(content='{"respuesta": "cortada', response_metadata={"finish_reason": "length"})
        with pytest.raises(chain_module.RespuestaIncompletaError):
            chain_module._validar_salida(mensaje)

    def test_rechaza_stop_reason_max_tokens_anthropic(self):
        mensaje = AIMessage(content='{"respuesta": "cortada', response_metadata={"stop_reason": "max_tokens"})
        with pytest.raises(chain_module.RespuestaIncompletaError):
            chain_module._validar_salida(mensaje)

    def test_rechaza_json_invalido(self):
        mensaje = AIMessage(content="esto no es JSON válido", response_metadata={})
        with pytest.raises(chain_module.RespuestaIncompletaError):
            chain_module._validar_salida(mensaje)


class TestBuildChain:
    async def test_respuesta_valida_en_primer_intento(self, monkeypatch):
        respuesta_json = '{"respuesta": "El pool de conexiones a PostgreSQL.", "contexto_encontrado": true}'
        monkeypatch.setattr(chain_module, "_build_model", lambda provider="openai", model=None: _fake_llm([respuesta_json]))
        chain = chain_module.build_chain()
        resultado = await chain.ainvoke({"contexto": "contexto de prueba", "pregunta": "¿Cuál es el cuello de botella?"})
        assert isinstance(resultado, RespuestaLLM)
        assert resultado.contexto_encontrado is True

    async def test_reintenta_ante_salida_mal_formada_y_se_recupera(self, monkeypatch):
        malformada = "esto no es JSON valido en absoluto"
        valida = '{"respuesta": "No lo sé.", "contexto_encontrado": false}'
        monkeypatch.setattr(
            chain_module, "_build_model", lambda provider="openai", model=None: _fake_llm([malformada, malformada, valida])
        )
        chain = chain_module.build_chain()
        resultado = await chain.ainvoke({"contexto": "contexto", "pregunta": "algo fuera de contexto"})
        assert resultado.contexto_encontrado is False

    async def test_agota_reintentos_y_propaga_error(self, monkeypatch):
        malformada = "esto no es JSON valido en absoluto"
        llm = _fake_llm([malformada] * 5)
        monkeypatch.setattr(chain_module, "_build_model", lambda provider="openai", model=None: llm)
        chain = chain_module.build_chain()
        with pytest.raises(chain_module.RespuestaIncompletaError):
            await chain.ainvoke({"contexto": "contexto", "pregunta": "algo"})

    async def test_no_reintenta_ante_error_no_relacionado(self, monkeypatch):
        """`retry_if_exception_type` acota el reintento a `RespuestaIncompletaError`:
        un error de otro tipo (ej. credenciales inválidas) se propaga directo,
        sin gastar los 3 reintentos en algo que no se va a resolver solo."""

        llamadas = []

        async def _llm_con_error(_input):
            llamadas.append(1)
            raise ValueError("credenciales inválidas")

        monkeypatch.setattr(
            chain_module, "_build_model", lambda provider="openai", model=None: RunnableLambda(_llm_con_error)
        )
        chain = chain_module.build_chain()
        with pytest.raises(ValueError):
            await chain.ainvoke({"contexto": "contexto", "pregunta": "algo"})
        assert len(llamadas) == 1


class TestAnswerQuestion:
    async def test_con_contexto_encontrado_incluye_fuentes_deduplicadas(self, monkeypatch):
        docs = [
            Document(page_content="frag 1", metadata={"source": "runbook_incidentes.md"}),
            Document(page_content="frag 2", metadata={"source": "runbook_incidentes.md"}),
            Document(page_content="frag 3", metadata={"source": "monitoreo_alertas.md"}),
        ]
        vectorstore = FakeVectorstore(docs)
        monkeypatch.setattr(
            chain_module,
            "build_chain",
            lambda provider="openai", model=None: FakeChain(
                RespuestaLLM(respuesta="El pool de PostgreSQL.", contexto_encontrado=True)
            ),
        )
        resultado = await chain_module.answer_question("¿Cuál es el cuello de botella?", vectorstore=vectorstore)
        assert resultado.fuentes == ["monitoreo_alertas.md", "runbook_incidentes.md"]
        assert resultado.contexto_encontrado is True
        assert resultado.pregunta == "¿Cuál es el cuello de botella?"

    async def test_sin_contexto_encontrado_fuentes_vacias(self, monkeypatch):
        docs = [Document(page_content="frag", metadata={"source": "normativa_despliegue.md"})]
        vectorstore = FakeVectorstore(docs)
        monkeypatch.setattr(
            chain_module,
            "build_chain",
            lambda provider="openai", model=None: FakeChain(
                RespuestaLLM(respuesta="No lo sé.", contexto_encontrado=False)
            ),
        )
        resultado = await chain_module.answer_question("pregunta rara", vectorstore=vectorstore)
        assert resultado.fuentes == []
        assert resultado.contexto_encontrado is False

    async def test_expone_fragmentos_con_metadata_y_similitud(self, monkeypatch):
        """Cada fragmento recuperado sale en la respuesta con su fuente, su
        sección (encabezados Markdown) y su similitud (1 - distancia), aunque
        el modelo no haya encontrado la respuesta en ellos."""
        docs = [
            Document(
                page_content="frag 1",
                metadata={"source": "runbook_incidentes.md", "Header 1": "Runbook", "Header 2": "Incidente 1"},
            ),
            Document(page_content="frag 2", metadata={"source": "notas.txt"}),
        ]
        vectorstore = FakeVectorstore(docs)  # distancias 0.0 y 0.1
        monkeypatch.setattr(
            chain_module,
            "build_chain",
            lambda provider="openai", model=None: FakeChain(
                RespuestaLLM(respuesta="No lo sé.", contexto_encontrado=False)
            ),
        )
        resultado = await chain_module.answer_question("algo", vectorstore=vectorstore)
        assert [f.model_dump() for f in resultado.fragmentos_recuperados] == [
            {"fuente": "runbook_incidentes.md", "seccion": "Runbook > Incidente 1", "similitud": 1.0},
            {"fuente": "notas.txt", "seccion": None, "similitud": 0.9},
        ]
        assert resultado.fuentes == []

    async def test_sin_documentos_recuperados(self, monkeypatch):
        vectorstore = FakeVectorstore([])
        monkeypatch.setattr(
            chain_module,
            "build_chain",
            lambda provider="openai", model=None: FakeChain(
                RespuestaLLM(respuesta="No lo sé.", contexto_encontrado=False)
            ),
        )
        resultado = await chain_module.answer_question("pregunta sin resultados", vectorstore=vectorstore)
        assert resultado.fuentes == []

    async def test_propaga_error_de_retrieval_con_log(self, caplog):
        """Si ChromaDB/el modelo de embeddings falla durante el retrieval (no
        durante la generación), el error se loguea con contexto propio -- no
        debe quedar como un traceback sin explicación -- y se propaga (no se
        traga en silencio)."""
        with pytest.raises(ConnectionError):
            await chain_module.answer_question("algo", vectorstore=FakeVectorstoreRoto())
        assert "Fallo recuperando contexto" in caplog.text
