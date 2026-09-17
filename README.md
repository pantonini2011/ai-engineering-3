# Sistema de recuperación semántica local (RAG)

Pre-entrega 3 · Módulo 3 — AI Engineering (Coderhouse). Sistema RAG end-to-end: ingesta de
documentos en ChromaDB, recuperación semántica y generación de respuestas *grounded* (ancladas
al contexto recuperado) vía LCEL (LangChain).

El "cerebro" documental (`rag/docs/`) son 4 manuales técnicos ficticios sobre una plataforma de
pedidos (FastAPI + PostgreSQL + Redis + Celery) — el mismo sistema descripto en los ejemplos del
[Módulo 2](https://github.com/pantonini2011/ai-engineering-2), para mantener continuidad entre
entregas: arquitectura, runbook de incidentes, normativa de despliegue y monitoreo/alertas.

## Estructura

```
rag/
├── docs/                     # El "cerebro": 4 manuales técnicos (.md) sobre la plataforma de pedidos
├── schemas.py                 # RespuestaLLM (lo que decide el modelo) y RespuestaRAG (contrato final)
├── ingest.py                   # Módulo de ingesta: chunking + persistencia en ChromaDB
├── chain.py                     # Retriever + cadena LCEL (PROMPT | llm | PydanticOutputParser)
├── main.py                       # Script de demo: preguntas con y sin respuesta en el contexto
└── tests/                        # Suite de tests con pytest (embeddings/LLM mockeados, sin red)
vectorstore/                  # Colección persistida de ChromaDB (generada al correr, gitignored)
```

## Requisitos

- Python 3.12 (mismo criterio que los Módulos 1 y 2). En Windows, con el `py launcher`:
  `py -3.12 -m venv .venv`.
- Una API key de Anthropic (proveedor de generación por default) y/o de OpenAI.
- Conexión a internet la primera vez que se corre: `sentence-transformers/all-MiniLM-L6-v2`
  (modelo de embeddings) se descarga una vez desde Hugging Face Hub y queda cacheado en disco;
  no hace falta ninguna API key para usarlo. [Ollama](https://ollama.com) es opcional, solo si se
  usa `provider="ollama"` para la generación.

## Instalación

```bash
py -3.12 -m venv .venv          # o `python3.12 -m venv .venv` fuera de Windows
.venv\Scripts\activate           # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # y completá tus credenciales
```

## Cómo correrlo

```bash
python -m rag.main
```

Esto ingesta `rag/docs/` (si `vectorstore/` no existe aún) y corre 5 preguntas: 4 con
respuesta en el contexto y 1 deliberadamente fuera de él, para verificar el comportamiento
"No lo sé".

## Qué es "local" acá, y qué no

"Local" en el título de la consigna describe a la **base vectorial**: ChromaDB persiste su
colección en disco (`./vectorstore`), no en un servicio hosteado. No implica que el LLM de
generación también deba serlo — de hecho el default de `answer_question()`/`build_chain()` es
`provider="anthropic"` (Claude), un modelo hosteado, y es una elección completamente válida: el
requisito de "local" de la consigna es sobre dónde vive el índice semántico, no sobre quién
redacta la respuesta final.

Los **embeddings** sí corren local, pero eso es una decisión de diseño aparte (ver abajo), no un
requisito de la consigna.

## Diseño: embeddings locales (Hugging Face)

`rag/ingest.py::_build_embeddings()` usa **`HuggingFaceEmbeddings`** con el modelo
`sentence-transformers/all-MiniLM-L6-v2`: corre 100% local (se descarga una vez desde el Hub y
queda cacheado en disco, sin API key ni servidor externo corriendo). Ninguno de los proveedores de
generación configurados (Anthropic, OpenAI) ofrece un modelo de embeddings propio sin sumar una
API extra (ej. Voyage AI) sólo para eso — `sentence-transformers` evita esa dependencia adicional.

`_build_embeddings()` está decorada con `@lru_cache(maxsize=1)`: a diferencia de un proveedor
hosteado (una llamada HTTP liviana) o de Ollama (el modelo ya vive cargado en el proceso de
`ollama serve`), `HuggingFaceEmbeddings` carga el modelo de `sentence-transformers` en memoria del
propio proceso Python (~2s) cada vez que se instancia. Sin cachear, cada pregunta de
`answer_question()` volvía a cargar el modelo desde cero — evidencia real en el log de una corrida
sin cachear: `Load pretrained SentenceTransformer` se repetía una vez por pregunta. Con el cache,
se carga una única vez por proceso.

La **generación** (el LLM que redacta la respuesta final) reutiliza el mismo criterio de
proveedor intercambiable de los Módulos 1 y 2 (`provider="anthropic"|"openai"|"ollama"` en
`answer_question()`/`build_chain()`) — son dos roles distintos (embeddings vs. generación) y no
hace falta que compartan proveedor.

`_build_embeddings()` es un único punto de construcción, importado tanto por `ingest.py`
(indexación) como por `chain.py` (a través de `ingest_documentos()`, para construir el
retriever de consulta). Esto evita a propósito el "error #1" que señala la consigna: indexar con
un modelo de embeddings y consultar con otro distinto, lo que vuelve la distancia vectorial
inútil sin que nada lo avise en tiempo de ejecución — acá es estructuralmente imposible que
ocurra, porque solo hay una función que construye embeddings en todo el proyecto.

## Módulo de ingesta (`ingest.py`)

`ingest_documentos()`:

1. Carga todos los `.txt`/`.md` de `rag/docs/` como `Document` de LangChain, con
   `metadata={"source": <nombre de archivo>}` — esa metadata es la que después permite reportar
   `fuentes` reales sin depender de que el LLM las recuerde (ver más abajo).
2. Los fragmenta con `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=150`).
3. Persiste los fragmentos + sus embeddings en una colección de ChromaDB (`./vectorstore`,
   colección `manuales_tecnicos`).

**Persistencia real, no solo teórica**: antes de indexar, `_coleccion_ya_poblada()` abre la
colección persistida y chequea `._collection.count() > 0`. Si ya tiene documentos, `ingest_documentos()`
devuelve directamente el `Chroma` existente sin volver a fragmentar ni re-embeddear nada (salvo
`force_reindex=True`) — evidencia en el log de una corrida real, segunda vez que se llama:

```
INFO rag.ingest: Colección 'manuales_tecnicos' ya poblada en D:\...\vectorstore: se omite la re-indexación.
```

## Cadena LCEL de generación grounded (`chain.py`)

```python
PROMPT = ChatPromptTemplate.from_messages([...]).partial(format_instructions=parser.get_format_instructions())
pipeline = PROMPT | llm | RunnableLambda(_validar_salida)
cadena_generacion = pipeline.with_retry(
    retry_if_exception_type=(RespuestaIncompletaError,),
    stop_after_attempt=MAX_RETRY_ATTEMPTS,
    wait_exponential_jitter=True,
)
```

- **`parser`** = `PydanticOutputParser(pydantic_object=RespuestaLLM)`, como pide explícitamente
  la consigna (a diferencia del Módulo 2, que usaba `with_structured_output` vía tool-calling).
  `parser.get_format_instructions()` se inyecta una sola vez en el prompt vía `.partial()`, no a
  mano.
- **El prompt de sistema** actúa como "filtro de veracidad": instruye a responder EXCLUSIVAMENTE
  con el CONTEXTO recibido y a decir explícitamente "No lo sé..." (con `contexto_encontrado=false`)
  si el contexto no alcanza, en vez de completar con conocimiento propio del modelo.
- **Prompt enriquecido con criterio de contenido + ejemplo de referencia** (mismo enfoque que pidió
  la corrección del Módulo 2: no alcanza con la regla general, hace falta un criterio concreto y un
  ejemplo anclado en un caso real). El agregado clave es explicitar que el CONTEXTO "alcanza" solo
  si contiene el **dato puntual** que la pregunta pide, no simplemente el mismo tema — con un
  ejemplo de referencia tomado del hallazgo real documentado más abajo (un CONTEXTO que describe un
  incidente pero no sus pasos de resolución no alcanza para responder "cómo se resuelve"). También
  instruye a no mezclar datos de fragmentos de documentos/incidentes distintos como si fueran una
  sola respuesta.
- **`_validar_salida`** (mismo criterio que `_validar_salida` del Módulo 2) chequea, en este
  orden:
  1. El `finish_reason` (OpenAI/Ollama) / `stop_reason` (Anthropic) del mensaje crudo del LLM. Si
     es `length`/`max_tokens`, la respuesta se cortó por límite de tokens — se rechaza *antes* de
     intentar parsearla, aunque el JSON parcial llegue a "parecer" válido.
  2. Si `PydanticOutputParser` no logra parsear el contenido (`OutputParserException`).

  En ambos casos levanta `RespuestaIncompletaError` (y lo loguea) — es el único punto con
  visibilidad de intento por intento, ya que `answer_question` sólo ve el resultado final de toda
  la cadena.
- **`.with_retry(retry_if_exception_type=(RespuestaIncompletaError,), ...)`**: ante una respuesta
  incompleta o mal formada, reintenta la cadena completa (un nuevo pedido al LLM, no un
  re-parseo del mismo texto) hasta `MAX_RETRY_ATTEMPTS` (3) veces, con backoff exponencial +
  jitter. Acotar el reintento a `RespuestaIncompletaError` en vez de reintentar ante cualquier
  excepción evita gastar los 3 reintentos ante un error que no se va a resolver solo (ej.
  credenciales inválidas, proveedor caído) — ese tipo de error se propaga directo, en el primer
  intento.
- **La frase canónica de "No lo sé" se garantiza en código, no solo en el prompt**: el prompt le
  pide al modelo responder "literalmente" `NO_CONTEXTO_MENSAJE` (constante en `rag/schemas.py`)
  cuando no hay contexto suficiente, pero un LLM no siempre respeta esa instrucción al pie de la
  letra — se observó a Claude, en una corrida real, agregando una explicación extra después de un
  `\n\n` en vez de responder solo la frase pedida. `RespuestaLLM` tiene un `model_validator` que
  fuerza `respuesta` al mensaje canónico exacto cuando `contexto_encontrado=False`, sin importar
  qué texto haya generado el modelo — la consistencia de esa respuesta no depende de que el LLM
  cumpla la instrucción del prompt.

`answer_question()` es el punto de entrada end-to-end:

1. **Retrieval**: `vectorstore.as_retriever(search_kwargs={"k": 4})` — `top_k=4`, dentro del
   rango 3-5 que recomienda la consigna para evitar el "contexto infinito" (degradación por
   *lost in the middle* y gasto de tokens de más).
2. Arma el string de contexto con `_format_docs()`, marcando la fuente de cada fragmento.
3. Corre `cadena_generacion` con `{"contexto": ..., "pregunta": ...}`.
4. **Las `fuentes` finales se calculan en código**, no las decide el LLM: se toman de la
   metadata `source` de los documentos que el retriever efectivamente trajo (`RespuestaLLM` no
   tiene ni siquiera un campo `fuentes`). Pedirle al LLM que liste nombres de archivo de memoria
   es una invitación a que invente uno plausible pero incorrecto, y Pydantic no puede validar
   contra la realidad si un nombre de archivo existe o no — el mismo tipo de problema que el
   Módulo 2 documentó con alucinaciones de contenido. Si `contexto_encontrado=false`, `fuentes`
   queda vacía aunque el retriever haya traído fragmentos (poco relevantes) igual.

## Evidencia real de ejecución

Corrida real (`python -m rag.main`, embeddings `sentence-transformers/all-MiniLM-L6-v2` local,
generación `provider="anthropic"`):

**Pregunta con respuesta clara en el contexto:**

```json
{
  "pregunta": "¿Qué componente es el cuello de botella histórico del sistema bajo carga alta, y por qué?",
  "respuesta": "El cuello de botella histórico del sistema bajo carga alta es el **pool de conexiones a PostgreSQL**. Específicamente, cuando hay más de 500 pedidos concurrentes, el límite de 100 conexiones de PgBouncer se satura, lo que causa que las nuevas requests queden esperando una conexión libre, generando timeouts intermitentes en el endpoint `/v1/pedidos`.",
  "contexto_encontrado": true,
  "fuentes": ["arquitectura_sistema.md", "monitoreo_alertas.md", "normativa_despliegue.md"]
}
```

**Pregunta deliberadamente fuera de todo el "cerebro" documental:**

```json
{
  "pregunta": "¿Cuál es la política de vacaciones del equipo de guardia?",
  "respuesta": "No lo sé, no tengo información sobre eso en el contexto disponible.",
  "contexto_encontrado": false,
  "fuentes": []
}
```

### Hallazgo: un `top_k` correcto no garantiza traer el fragmento *correcto*

Para la pregunta "¿Qué pasos hay que seguir para resolver un agotamiento del pool de conexiones a
PostgreSQL?", el retriever trajo 4 fragmentos relevantes por *tema* (arquitectura del sistema y el
runbook de incidentes), pero el chunking (`chunk_size=1000`) separó, dentro de
`runbook_incidentes.md`, la descripción del síntoma/causa raíz del Incidente 1 de su lista de
**pasos de resolución** en fragmentos distintos — y el fragmento con los pasos concretos no entró
en el top-4 recuperado. El modelo no inventó una respuesta: reconoció que el contexto describía el
problema pero no los pasos, y respondió "No lo sé" en vez de alucinar un procedimiento plausible:

```json
{
  "pregunta": "¿Qué pasos hay que seguir para resolver un agotamiento del pool de conexiones a PostgreSQL?",
  "respuesta": "No lo sé, no tengo información sobre eso en el contexto disponible.",
  "contexto_encontrado": false,
  "fuentes": []
}
```

*Nota sobre esta evidencia*: en la primera corrida real, Claude marcó `contexto_encontrado=false`
correctamente pero además agregó, después de la frase canónica, una explicación de por qué no
podía responder (separada por un `\n\n`) — desviándose de la instrucción del prompt de responder
"literalmente" sólo esa frase. Ese hallazgo fue el que motivó normalizar `respuesta` en código (ver
sección anterior); el JSON de arriba ya refleja el comportamiento corregido.

Esto no es un bug del pipeline ni del prompt: es el trade-off real de la consigna entre "no pasar
contexto infinito" (`top_k` bajo) y la posibilidad de que la información relevante quede repartida
en un fragmento que el `top_k` elegido no llega a traer. Un `chunk_size` mayor (que mantenga la
sección "Resolución" en el mismo fragmento que su incidente) o un `top_k` más alto habrían traído
esa información — a costa de más tokens por consulta. El comportamiento correcto y observado acá
es que el sistema prefiere decir "No lo sé" antes que inventar pasos de resolución no verificados,
que es exactamente lo que pide la consigna.

## Errores comunes evitados (según la consigna)

- **Contexto infinito**: `top_k=4` (rango 3-5 recomendado), no se pasan decenas de fragmentos.
- **Embeddings no coincidentes**: `_build_embeddings()` es la única función que construye
  embeddings en todo el proyecto; la usan tanto la ingesta como la consulta.
- **Falta de persistencia**: `_coleccion_ya_poblada()` evita re-indexar si la colección ya existe
  y tiene documentos (ver log de ejemplo más arriba).

## Tests

```bash
pytest rag/tests/ -v
```

33 tests, sin llamadas de red reales (mismo criterio que los Módulos 1 y 2):

- `test_schemas.py`: validación de `RespuestaLLM` (respuesta no vacía, limpieza de espacios,
  campos requeridos, tipos, normalización de `respuesta` a `NO_CONTEXTO_MENSAJE` cuando
  `contexto_encontrado=False`) y `RespuestaRAG` (fuentes default vacía, pregunta/respuesta no
  vacías).
- `test_ingest.py`: usa un `FakeEmbeddings` determinístico (mismo texto → mismo vector, sin red)
  inyectado vía monkeypatch de `_build_embeddings`, con ChromaDB real apuntando a un directorio
  temporal. Cubre indexación real, filtrado de archivos no `.txt`/`.md`, no-reindexación si la
  colección ya está poblada, `force_reindex=True`, y error si el directorio de documentos está
  vacío.
- `test_chain.py`: mockea `_build_model` con `FakeListChatModel` (devuelve, en orden, las
  respuestas configuradas por cada test) para probar `_format_docs` y `_build_model` (selección de
  proveedor). `_validar_salida` se prueba en aislamiento: acepta una salida completa, rechaza
  `finish_reason`/`stop_reason` de truncamiento (OpenAI/Ollama y Anthropic) y rechaza JSON
  inválido, en los tres casos levantando `RespuestaIncompletaError`. La cadena LCEL completa se
  prueba con los **casos críticos** de reintentos: respuesta válida al primer intento, reintento
  ante salida mal formada con recuperación en un intento posterior, agotamiento de los
  `MAX_RETRY_ATTEMPTS` reintentos, y que un error *no relacionado* (ej. `ValueError` de
  credenciales inválidas) se propaga directo, sin gastar reintentos, porque
  `retry_if_exception_type` lo acota a `RespuestaIncompletaError`. `answer_question()` se prueba
  con un `FakeVectorstore`/`FakeRetriever` y una `FakeChain`, verificando que las `fuentes` se
  calculan a partir de los documentos recuperados (deduplicadas y ordenadas) y quedan vacías
  cuando `contexto_encontrado=False`.

La fixture `fast_retries` (en `rag/tests/conftest.py`) parchea `asyncio.sleep` para neutralizar
el backoff real de `.with_retry()` durante los tests, mismo criterio que el Módulo 2.
