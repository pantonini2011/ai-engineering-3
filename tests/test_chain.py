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

        self.filtro_recibido = None

    async def asimilarity_search_with_score(self, query, k=None, filter=None):
        self.filtro_recibido = filter
        return self._docs_con_score


class FakeChain:
    def __init__(self, resultado):
        self._resultado = resultado

    async def ainvoke(self, inputs):
        return self._resultado


class FakeVectorstoreRoto:
    async def asimilarity_search_with_score(self, query, k=None, filter=None):
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
        # fallback_provider == provider desactiva el fallback, para aislar el retry puro.
        chain = chain_module.build_chain(provider="anthropic", fallback_provider="anthropic")
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
        chain = chain_module.build_chain(provider="anthropic", fallback_provider="anthropic")
        with pytest.raises(ValueError):
            await chain.ainvoke({"contexto": "contexto", "pregunta": "algo"})
        assert len(llamadas) == 1



class TestFallbackEntreProveedores:
    """Mismo criterio de resiliencia que el Módulo 2: si el proveedor principal
    falla, `.with_fallbacks()` reintenta la generación completa con otro."""

    VALIDA = '{"respuesta": "El pool de conexiones a PostgreSQL.", "contexto_encontrado": true}'

    async def test_recurre_al_fallback_si_el_principal_agota_reintentos(self, monkeypatch):
        llamadas = {"anthropic": 0, "openai": 0}
        respuestas = {"anthropic": "esto no es JSON", "openai": self.VALIDA}

        def _modelo(provider="anthropic", model=None):
            def _responder(_input):
                llamadas[provider] += 1
                return AIMessage(content=respuestas[provider], response_metadata={})

            return RunnableLambda(_responder)

        monkeypatch.setattr(chain_module, "_build_model", _modelo)
        chain = chain_module.build_chain(provider="anthropic")
        resultado = await chain.ainvoke({"contexto": "contexto", "pregunta": "algo"})
        assert resultado.respuesta == "El pool de conexiones a PostgreSQL."
        assert llamadas == {"anthropic": chain_module.MAX_RETRY_ATTEMPTS, "openai": 1}

    async def test_recurre_al_fallback_si_el_principal_esta_caido(self, monkeypatch, caplog):
        """Un error no reintentable (credenciales inválidas, proveedor caído) no
        gasta reintentos en el principal: pasa directo al fallback, y queda
        logueado qué proveedor falló y cuál tomó su lugar."""

        async def _caido(_input):
            raise ConnectionError("proveedor caído")

        def _modelo(provider="anthropic", model=None):
            return RunnableLambda(_caido) if provider == "anthropic" else _fake_llm([self.VALIDA])

        monkeypatch.setattr(chain_module, "_build_model", _modelo)
        chain = chain_module.build_chain(provider="anthropic")
        resultado = await chain.ainvoke({"contexto": "contexto", "pregunta": "algo"})
        assert resultado.contexto_encontrado is True
        assert "Falló el proveedor 'anthropic'" in caplog.text
        assert "fallback 'openai'" in caplog.text

    async def test_sin_api_key_del_fallback_no_se_agrega(self, monkeypatch):
        """Si el fallback no tiene API key, la cadena queda solo con el principal
        (y su error se propaga) en vez de romper al construir un cliente sin key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        construidos = []

        def _modelo(provider="anthropic", model=None):
            construidos.append(provider)
            return _fake_llm([self.VALIDA])

        monkeypatch.setattr(chain_module, "_build_model", _modelo)
        chain_module.build_chain(provider="anthropic")
        assert construidos == ["anthropic"]

    def test_fallback_por_defecto_cruza_proveedores(self):
        assert chain_module.fallback_por_defecto("anthropic") == "openai"
        assert chain_module.fallback_por_defecto("openai") == "anthropic"
        assert chain_module.fallback_por_defecto("ollama") == "anthropic"

    @pytest.mark.parametrize("valor", ["", "tu_openai_api_key", "sk-..."])
    def test_placeholder_o_vacia_no_cuenta_como_disponible(self, monkeypatch, valor):
        monkeypatch.setenv("OPENAI_API_KEY", valor)
        assert chain_module.proveedor_disponible("openai") is False

    def test_ollama_siempre_disponible(self):
        assert chain_module.proveedor_disponible("ollama") is True

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
            lambda provider="openai", model=None, fallback_provider=None: FakeChain(
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
            lambda provider="openai", model=None, fallback_provider=None: FakeChain(
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
            lambda provider="openai", model=None, fallback_provider=None: FakeChain(
                RespuestaLLM(respuesta="No lo sé.", contexto_encontrado=False)
            ),
        )
        resultado = await chain_module.answer_question("algo", vectorstore=vectorstore)
        assert [f.model_dump() for f in resultado.fragmentos_recuperados] == [
            {
                "fuente": "runbook_incidentes.md",
                "seccion": "Runbook > Incidente 1",
                "similitud": 1.0,
                "extracto": "frag 1",
                "created_at": None,
                "env": None,
            },
            {
                "fuente": "notas.txt",
                "seccion": None,
                "similitud": 0.9,
                "extracto": "frag 2",
                "created_at": None,
                "env": None,
            },
        ]
        assert resultado.fuentes == []

    async def test_pasa_el_filtro_de_metadata_al_vectorstore(self, monkeypatch):
        vectorstore = FakeVectorstore([Document(page_content="frag", metadata={"source": "runbook_incidentes.md"})])
        monkeypatch.setattr(
            chain_module,
            "build_chain",
            lambda provider="openai", model=None, fallback_provider=None: FakeChain(RespuestaLLM(respuesta="Ok.", contexto_encontrado=True)),
        )
        await chain_module.answer_question(
            "algo", vectorstore=vectorstore, filtro={"source": "runbook_incidentes.md"}
        )
        assert vectorstore.filtro_recibido == {"source": "runbook_incidentes.md"}

    async def test_sin_documentos_recuperados(self, monkeypatch):
        vectorstore = FakeVectorstore([])
        monkeypatch.setattr(
            chain_module,
            "build_chain",
            lambda provider="openai", model=None, fallback_provider=None: FakeChain(
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
