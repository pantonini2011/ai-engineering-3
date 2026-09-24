# Sistema de recuperación semántica local (RAG)

Pre-entrega 3 · Módulo 3 — AI Engineering (Coderhouse).

**Entregable**: el script `src/main.py` (se corre con `python -m src.main`, ver
[Guía de ejecución](#guía-de-ejecución)). La consigna original está en
[`preentrega3.md`](preentrega3.md).

## Índice

- [Descripción técnica](#descripción-técnica)
- [Checklist de la consigna](#checklist-de-la-consigna)
- [Decisiones de diseño](#decisiones-de-diseño)
- [Guía de ejecución](#guía-de-ejecución)
- [Ejemplo de entrada/salida](#ejemplo-de-entradasalida)
- [Estructura](#estructura)
- [Diseño en detalle](#diseño)
  - [Qué es "local" acá, y qué no](#qué-es-local-acá-y-qué-no)
  - [Embeddings locales (Hugging Face)](#embeddings-locales-hugging-face)
- [Arquitectura interna](#arquitectura-interna)
  - [Módulo de ingesta (`ingest.py`)](#módulo-de-ingesta-ingestpy)
  - [Cadena LCEL de generación grounded (`chain.py`)](#cadena-lcel-de-generación-grounded-chainpy)
- [Evidencia real de ejecución](#evidencia-real-de-ejecución)
- [Errores comunes evitados (según la consigna)](#errores-comunes-evitados-según-la-consigna)
- [Tests](#tests)

## Descripción técnica

Sistema de **recuperación semántica local (RAG)** que responde preguntas sobre un conjunto de
manuales técnicos usando **solo** la información de esos manuales. Si la respuesta no está en
ellos, dice "No lo sé" en vez de inventarla.

El flujo end-to-end tiene tres etapas:

1. **Ingesta** (`src/ingest.py`): lee los `.md`/`.txt` de `data/`, los fragmenta, calcula un
   embedding por fragmento con un modelo local de Hugging Face y los persiste en una base
   vectorial **ChromaDB local** (`./vectorstore`). Si la base ya existe, no re-indexa.
2. **Recuperación** (`src/chain.py`): convierte la pregunta en embedding con el mismo modelo y
   trae de ChromaDB los 4 fragmentos más parecidos por similitud coseno.
3. **Generación grounded** (`src/chain.py`): una cadena LCEL
   (`prompt | LLM | PydanticOutputParser`) redacta la respuesta usando solo esos fragmentos y la
   devuelve como JSON validado, con las fuentes y los scores de similitud.

El "cerebro" documental (`data/`) son 4 manuales técnicos ficticios sobre una plataforma de
pedidos (FastAPI + PostgreSQL + Redis + Celery) — el mismo sistema descripto en los ejemplos del
[Módulo 2](https://github.com/pantonini2011/ai-engineering-2), para mantener continuidad entre
entregas: arquitectura, runbook de incidentes, normativa de despliegue y monitoreo/alertas.

## Checklist de la consigna

Cada requisito de [`preentrega3.md`](preentrega3.md), dónde está implementado y con qué se
verifica. Los logs de [`evidencia/`](evidencia/) son salidas reales de `python -m src.main` y
`pytest`, sin editar (ver [`evidencia/README.md`](evidencia/README.md)).

### Componentes a entregar

| Requisito | Implementación | Evidencia verificable |
|---|---|---|
| Script/notebook con flujo RAG end-to-end | [`src/main.py`](src/main.py) → `main()`: ingesta, 4 preguntas con contexto y 1 sin contexto | [`evidencia/corrida_1_indexacion.txt`](evidencia/corrida_1_indexacion.txt) |
| **Módulo de ingesta**: toma documentos `.txt`/`.md`, los fragmenta y los persiste en ChromaDB | [`src/ingest.py`](src/ingest.py) → `ingest_documentos()` (usa `_cargar_documentos()` y `_fragmentar_documento()`) | Log: `Indexando 24 fragmentos de 4 documento(s) en la colección 'manuales_tecnicos_dev'` · tests `test_ingesta_indexa_y_persiste`, `test_ingesta_ignora_archivos_no_txt_md` |
| **Retriever**: convierte la pregunta en embedding y trae los fragmentos más relevantes | [`src/chain.py`](src/chain.py) → `answer_question()`: `vectorstore.asimilarity_search_with_score(pregunta, k=TOP_K)` | Log: `Recuperados 4 fragmento(s): ['arquitectura_sistema.md (similitud=54.26%)', ...]` |
| **Generación grounded** con una cadena LCEL | [`src/chain.py`](src/chain.py) → `build_chain()`: `PROMPT \| llm \| RunnableLambda(_validar_salida)` + `.with_retry()` | Respuestas JSON en la corrida 1 · tests `TestBuildChain` en [`tests/test_chain.py`](tests/test_chain.py) |
| El prompt instruye a decir **"No lo sé"** si la respuesta no está en el contexto | [`src/chain.py`](src/chain.py) → `PROMPT`; frase fija `NO_CONTEXTO_MENSAJE` en [`src/schemas.py`](src/schemas.py) | Pregunta sobre vacaciones → `"No lo sé, no tengo información sobre eso en el contexto disponible."`, `contexto_encontrado: false` · test `test_normaliza_respuesta_cuando_no_hay_contexto` |

### Pasos sugeridos

| Paso | Implementación | Evidencia verificable |
|---|---|---|
| 3 o 4 archivos de texto sobre un tema específico | 4 manuales `.md` en [`data/`](data/) | Log: `... de 4 documento(s)` |
| Fragmentar con `RecursiveCharacterTextSplitter` | [`src/ingest.py`](src/ingest.py) → `_fragmentar_documento()`: `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)`, aplicado dentro de cada sección Markdown (ver [por qué](#módulo-de-ingesta-ingestpy)) | Tests `TestFragmentarDocumento` en [`tests/test_ingest.py`](tests/test_ingest.py) |
| Cliente ChromaDB persistente en una carpeta local (`./vectorstore`) | [`src/ingest.py`](src/ingest.py) → `PERSIST_DIRECTORY = <raíz>/vectorstore`, colección `manuales_tecnicos_<RAG_ENV>` (`manuales_tecnicos_dev` por default) | Log: `... (D:\...\vectorstore)`; la carpeta queda creada tras correr |
| Mismo modelo de embeddings para indexar y consultar | [`src/ingest.py`](src/ingest.py) → `_build_embeddings()`, única función que crea embeddings en el proyecto | Las similitudes de la corrida 2 (base ya persistida) son idénticas a las de la corrida 1 |
| Prompt de sistema como "filtro de veracidad" | [`src/chain.py`](src/chain.py) → `PROMPT`: "responde EXCLUSIVAMENTE en base al CONTEXTO..." | Ver texto completo del prompt en el código |
| Salida por un `PydanticOutputParser` | [`src/chain.py`](src/chain.py) → `parser = PydanticOutputParser(pydantic_object=RespuestaLLM)`, usado en `_validar_salida()` | Tests `test_acepta_salida_completa`, `test_rechaza_json_invalido` |

### Errores comunes a evitar

| Error | Cómo se evita | Evidencia verificable |
|---|---|---|
| Contexto infinito | `TOP_K = 4` en [`src/chain.py`](src/chain.py) (rango 3-5) | Log: cada pregunta recupera exactamente `4 fragmento(s)` |
| Embeddings no coincidentes | Un solo `_build_embeddings()` para ingesta y consulta | Corrida 2: mismo ranking y mismas similitudes que la corrida 1 |
| Falta de persistencia (re-indexar siempre) | `_coleccion_ya_poblada()` en [`src/ingest.py`](src/ingest.py): si la colección ya tiene documentos, no re-indexa | [`evidencia/corrida_2_persistencia.txt`](evidencia/corrida_2_persistencia.txt): `Colección 'manuales_tecnicos_dev' ya poblada ... se omite la re-indexación.` · test `test_ingesta_no_reindexa_si_ya_esta_poblada` |

## Decisiones de diseño

Resumen de las decisiones técnicas. El detalle y los hallazgos que las motivaron están en
[Diseño en detalle](#diseño) y [Arquitectura interna](#arquitectura-interna).

| Decisión | Elección | Justificación |
|---|---|---|
| **Base vectorial** | ChromaDB en modo persistente local (`./vectorstore`) | La consigna pide un sistema local. No requiere servidor ni cuenta: la colección vive en disco y sobrevive entre corridas. |
| **Modelo de embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` vía `HuggingFaceEmbeddings` | Corre 100% local, sin API key. Es **multilingüe**: los documentos y las preguntas están en español, y un modelo entrenado casi solo en inglés (ej. `all-MiniLM-L6-v2`) mide peor la similitud ante sinónimos en español. |
| **Dimensión del embedding** | **384** | Es la dimensión de salida del modelo elegido. Se mide sobre un vector real y se loguea al arrancar: `Modelo de embeddings: ... (dimensión 384)`. Como referencia, `text-embedding-3-small` de OpenAI usa 1536: 384 alcanza para un corpus chico y hace la indexación rápida en CPU. |
| **Métrica de similitud** | Coseno, fijada explícitamente con `collection_metadata={"hnsw:space": "cosine"}` | En embeddings de texto importa la **dirección** del vector (el tema), no su magnitud (el largo del texto); el coseno mide justamente eso. Además, sin especificarla Chroma usa L2² por default. Con coseno explícito, el score que devuelve Chroma es `1 - similitud_coseno`, que se puede leer directamente como similitud (`1 - distancia`). Los vectores se normalizan (`normalize_embeddings=True`), que es lo que este modelo espera. |
| **Chunking** | `MarkdownHeaderTextSplitter` (por `#`/`##`/`###`) + `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)` como fallback dentro de secciones largas | Mantiene cada sección del manual (ej. un incidente completo, con síntoma, causa y resolución) en un solo fragmento. El splitter plano original cortaba secciones por la mitad y hacía perder datos puntuales (ver [Hallazgo](#hallazgo-un-top_k-correcto-no-garantiza-traer-el-fragmento-correcto)). Resultado: 24 fragmentos para 4 documentos. |
| **Metadatos por fragmento** | `source` (archivo) + `Header 1`/`Header 2`/`Header 3` (sección) | Las `fuentes` y la `seccion` de cada fragmento en la salida se calculan desde la metadata, en código, sin depender de que el LLM las recuerde. |
| **Segmentación por entorno (dev/prod)** | Una colección por entorno: `manuales_tecnicos_<RAG_ENV>`, con `RAG_ENV=dev` por default | Chroma no tiene *namespaces* como Pinecone; el equivalente es una colección separada dentro del mismo `./vectorstore`. Así una re-indexación de prueba en `dev` no pisa los vectores que consulta `prod`. Test: `test_colecciones_por_entorno_no_se_pisan`. |
| **`top_k`** | 4 | Dentro del rango 3-5 de la consigna: suficiente contexto sin caer en "contexto infinito" (*lost in the middle*, tokens de más). |
| **LLM de generación** | Claude (`claude-haiku-4-5-20251001`) por default; OpenAI u Ollama intercambiables con `provider=` | "Local" en la consigna describe a la base vectorial, no al LLM (ver [Qué es "local" acá](#qué-es-local-acá-y-qué-no)). |
| **Salida estructurada** | `PydanticOutputParser(RespuestaLLM)` + reintento automático si la salida llega truncada o mal formada | Lo pide la consigna; el reintento evita que una respuesta cortada llegue como válida. |

## Guía de ejecución

### Requisitos

- Python 3.12. En Windows, con el `py launcher`: `py -3.12`.
- Una API key de Anthropic (proveedor de generación por default). Opcional: OpenAI u
  [Ollama](https://ollama.com) si se usa `provider="openai"` / `provider="ollama"`.
- Conexión a internet la primera vez: el modelo de embeddings
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` se descarga una vez desde
  Hugging Face Hub y queda cacheado en disco. No necesita API key.

### Paso a paso

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/pantonini2011/ai-engineering-3.git
   cd ai-engineering-3
   ```

2. **Crear y activar un entorno virtual**

   ```bash
   py -3.12 -m venv .venv          # Linux/Mac: python3.12 -m venv .venv
   .venv\Scripts\activate           # Linux/Mac: source .venv/bin/activate
   ```

3. **Instalar dependencias**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**: copiar `.env.example` a `.env` y completar
   `ANTHROPIC_API_KEY` (la única obligatoria). Opcional: `RAG_ENV=dev|prod` elige la colección
   de ChromaDB (ver [Decisiones de diseño](#decisiones-de-diseño)).

   ```bash
   copy .env.example .env           # Linux/Mac: cp .env.example .env
   ```

5. **Correr el flujo RAG end-to-end**

   ```bash
   python -m src.main
   ```

   Qué hace:
   - Si `./vectorstore` no existe, fragmenta los 4 documentos de `data/` y los persiste en
     ChromaDB. Log esperado: `Indexando 24 fragmentos de 4 documento(s) ...` y
     `Indexación completa: 24 fragmentos persistidos.`
   - Hace 5 preguntas: 4 con respuesta en los documentos y 1 deliberadamente fuera de ellos.
     Cada respuesta se imprime como JSON (`pregunta`, `respuesta`, `contexto_encontrado`,
     `fuentes`). La última tiene que ser `"No lo sé, no tengo información sobre eso en el
     contexto disponible."` con `contexto_encontrado: false`.
   - El log también queda guardado en `rag.log`.

   Salida completa de referencia: [`evidencia/corrida_1_indexacion.txt`](evidencia/corrida_1_indexacion.txt).

6. **Verificar la persistencia**: correr `python -m src.main` una segunda vez. Ahora el log
   tiene que decir `Colección 'manuales_tecnicos_dev' ya poblada en ...: se omite la re-indexación.`
   Referencia: [`evidencia/corrida_2_persistencia.txt`](evidencia/corrida_2_persistencia.txt).

7. **Correr los tests** (no necesitan API key ni red: embeddings y LLM están mockeados)

   ```bash
   pytest -v
   ```

   Resultado esperado: `39 passed`. Referencia: [`evidencia/tests_pytest.txt`](evidencia/tests_pytest.txt).

Para re-indexar desde cero (por ejemplo, después de cambiar `EMBEDDING_MODEL` en `.env`),
borrar la carpeta `vectorstore/` y volver a correr el paso 5.

## Ejemplo de entrada/salida

`python -m src.main` hace 5 preguntas fijas (definidas en `src/main.py`) y por cada una imprime
un JSON `RespuestaRAG`:

- `respuesta` y `contexto_encontrado` los decide el LLM.
- `fuentes` y `fragmentos_recuperados` se calculan en código a partir de lo que devolvió
  ChromaDB. Cada fragmento trae su metadata (`fuente`, `seccion`) y su score (`similitud` =
  `1 - distancia coseno`, donde 1.0 es idéntico).

Copiado sin editar de [`evidencia/corrida_1_indexacion.txt`](evidencia/corrida_1_indexacion.txt).

**Entrada** (pregunta con respuesta en los documentos):

```
¿Qué pasos hay que seguir para resolver un agotamiento del pool de conexiones a PostgreSQL?
```

**Salida:**

```json
{
  "pregunta": "¿Qué pasos hay que seguir para resolver un agotamiento del pool de conexiones a PostgreSQL?",
  "respuesta": "Según el runbook de incidentes, los pasos para resolver un agotamiento del pool de conexiones a PostgreSQL son:\n\n1. Verificar en el dashboard de PgBouncer cuántas conexiones están activas vs. el límite (100).\n\n2. Si hay una query \"colgada\" reteniendo conexiones, identificarla con:\n   ```\n   SELECT * FROM pg_stat_activity WHERE state = 'active' ORDER BY query_start;\n   ```\n   y, si corresponde, cancelarla con `pg_cancel_backend(pid)`.\n\n3. Si el problema es puramente de volumen (no hay queries colgadas), escalar horizontalmente el número de réplicas de PgBouncer o subir el límite de conexiones a 150.",
  "contexto_encontrado": true,
  "fuentes": [
    "arquitectura_sistema.md",
    "monitoreo_alertas.md",
    "runbook_incidentes.md"
  ],
  "fragmentos_recuperados": [
    {
      "fuente": "runbook_incidentes.md",
      "seccion": "Runbook de incidentes — Plataforma de Pedidos > Incidente 1: Agotamiento del pool de conexiones a PostgreSQL",
      "similitud": 0.6701
    },
    {
      "fuente": "monitoreo_alertas.md",
      "seccion": "Monitoreo y alertas — Plataforma de Pedidos > Métricas clave por componente > PostgreSQL / PgBouncer",
      "similitud": 0.5825
    },
    {
      "fuente": "arquitectura_sistema.md",
      "seccion": "Arquitectura del sistema — Plataforma de Pedidos > Consideraciones de escalabilidad",
      "similitud": 0.581
    },
    {
      "fuente": "arquitectura_sistema.md",
      "seccion": "Arquitectura del sistema — Plataforma de Pedidos > Flujo de un pedido",
      "similitud": 0.5237
    }
  ]
}
```

**Entrada** (pregunta deliberadamente fuera de los documentos):

```
¿Cuál es la política de vacaciones del equipo de guardia?
```

**Salida:** el retriever igual trae 4 fragmentos, pero con similitud baja (máximo 0.3592)
y ninguno contiene el dato. El modelo responde la frase canónica de "No lo sé" y `fuentes`
queda vacía:

```json
{
  "pregunta": "¿Cuál es la política de vacaciones del equipo de guardia?",
  "respuesta": "No lo sé, no tengo información sobre eso en el contexto disponible.",
  "contexto_encontrado": false,
  "fuentes": [],
  "fragmentos_recuperados": [
    {
      "fuente": "normativa_despliegue.md",
      "seccion": "Normativa de despliegue — Plataforma de Pedidos > Capacitación y checklist de guardia",
      "similitud": 0.3592
    },
    {
      "fuente": "runbook_incidentes.md",
      "seccion": "Runbook de incidentes — Plataforma de Pedidos",
      "similitud": 0.2492
    },
    {
      "fuente": "normativa_despliegue.md",
      "seccion": "Normativa de despliegue — Plataforma de Pedidos > Proceso de despliegue a producción",
      "similitud": 0.2265
    },
    {
      "fuente": "monitoreo_alertas.md",
      "seccion": "Monitoreo y alertas — Plataforma de Pedidos > Canales de alerta",
      "similitud": 0.1926
    }
  ]
}
```

Log de la misma corrida (modelo y dimensión, indexación y scores de la primera pregunta):

```
INFO src.ingest: Modelo de embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (dimensión 384)
INFO src.ingest: Indexando 24 fragmentos de 4 documento(s) en la colección 'manuales_tecnicos_dev' (...\vectorstore)...
INFO src.chain: Recuperados 4 fragmento(s): ['runbook_incidentes.md (similitud=67.01%)', 'monitoreo_alertas.md (similitud=58.25%)', 'arquitectura_sistema.md (similitud=58.10%)', 'arquitectura_sistema.md (similitud=52.37%)']
```

## Estructura

```
src/               # Código fuente
├── ingest.py      # Módulo de ingesta: embeddings, chunking y persistencia en ChromaDB
├── chain.py       # Retriever + cadena LCEL (PROMPT | llm | PydanticOutputParser)
├── schemas.py     # RespuestaLLM (lo que decide el modelo), RespuestaRAG y FragmentoRecuperado (salida final)
└── main.py        # Script entregable: ingesta + preguntas con y sin respuesta en el contexto
data/              # El "cerebro": 4 manuales técnicos (.md) usados para la ingesta
tests/             # Suite de pytest (embeddings/LLM mockeados, sin red ni API keys)
evidencia/         # Salidas reales de `python -m src.main` (2 corridas) y de `pytest -v`
preentrega3.md     # Consigna
.env.example       # Plantilla de variables de entorno (sin valores reales)
.gitignore         # Ignora .env, .venv/, __pycache__/, vectorstore/, logs
requirements.txt   # Dependencias con versiones fijadas
pytest.ini         # Configuración de pytest
vectorstore/       # Colección persistida de ChromaDB (se genera al correr, gitignored)
```

No hay configuración oculta: todo lo que el código lee del entorno está listado en
[`.env.example`](.env.example), y lo único obligatorio es `ANTHROPIC_API_KEY`.

## Diseño

### Qué es "local" acá, y qué no

"Local" en el título de la consigna describe a la **base vectorial**: ChromaDB persiste su
colección en disco (`./vectorstore`), no en un servicio hosteado. No implica que el LLM de
generación también deba serlo — de hecho el default de `answer_question()`/`build_chain()` es
`provider="anthropic"` (Claude), un modelo hosteado, y es una elección completamente válida: el
requisito de "local" de la consigna es sobre dónde vive el índice semántico, no sobre quién
redacta la respuesta final.

Los **embeddings** sí corren local, pero eso es una decisión de diseño aparte (ver abajo), no un
requisito de la consigna.

### Embeddings locales (Hugging Face)

`src/ingest.py::_build_embeddings()` usa **`HuggingFaceEmbeddings`** con el modelo
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: corre 100% local (se descarga una
vez desde el Hub y queda cacheado en disco, sin API key ni servidor externo corriendo). Ninguno
de los proveedores de generación configurados (Anthropic, OpenAI) ofrece un modelo de embeddings
propio sin sumar una API extra (ej. Voyage AI) sólo para eso — `sentence-transformers` evita esa
dependencia adicional.

Se eligió la variante **multilingüe** en vez de `all-MiniLM-L6-v2` (entrenado casi
exclusivamente en inglés) porque todo el corpus documental (`data/`) y las preguntas de
prueba están en español — un modelo solo-inglés degrada la similitud coseno ante sinonimia o
lenguaje coloquial en español.

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

## Arquitectura interna

### Módulo de ingesta (`ingest.py`)

`ingest_documentos()`:

1. Carga todos los `.txt`/`.md` de `data/` como `Document` de LangChain, con
   `metadata={"source": <nombre de archivo>}` — esa metadata es la que después permite reportar
   `fuentes` reales sin depender de que el LLM las recuerde (ver más abajo).
2. Los fragmenta con un **splitter jerárquico** (`_fragmentar_documento()`): primero
   `MarkdownHeaderTextSplitter` por encabezado (`#`/`##`/`###`), y recién para las secciones que
   sigan superando `chunk_size=1000` caracteres, `RecursiveCharacterTextSplitter` como *fallback*
   dentro de esa sección puntual (`chunk_overlap=150`).
3. Persiste los fragmentos + sus embeddings en una colección de ChromaDB (`./vectorstore`,
   colección `manuales_tecnicos_<RAG_ENV>` (`manuales_tecnicos_dev` por default)), creada con `collection_metadata={"hnsw:space": "cosine"}`.

**Por qué jerárquico y no un splitter plano**: el splitter plano original (una sola pasada de
`RecursiveCharacterTextSplitter` sobre el documento completo) cortaba por cantidad de caracteres,
ciego al contenido — causa raíz de un hallazgo real, documentado en detalle en
[Evidencia real de ejecución](#evidencia-real-de-ejecución). `MarkdownHeaderTextSplitter` evita
eso de raíz: mantiene cada `##` (ej. "Incidente 1" completo, con síntoma + causa + resolución) en
un solo fragmento siempre que entre en `chunk_size`; solo recurre al fallback de caracteres para
las 3 secciones del corpus que efectivamente lo superan (`Componentes principales`,
`Proceso de despliegue a producción`, `Incidente 1`) — y ahí corta *dentro* de esa sección
puntual, nunca mezclando contenido de dos secciones o documentos distintos en el mismo fragmento
(a diferencia del splitter plano).

**Métrica de distancia explícita, no la default implícita**: sin especificar `hnsw:space`, Chroma
usa L2 al cuadrado por default, no coseno. Como los embeddings ya están normalizados
(`normalize_embeddings=True` en `_build_embeddings()`), el *orden* de similitud es idéntico con
cualquiera de las dos métricas (L2² sobre vectores unitarios es `2 - 2·cos_sim`, una
transformación monótona de la similitud coseno) — pero fijar `"cosine"` explícitamente hace que el
score que devuelva Chroma sea literalmente `1 - similitud_coseno`, más interpretable si en algún
momento se loguea o expone. Verificado a mano: con `hnsw:space` sin especificar, el score de
Chroma coincidía con `2 - 2·cos_sim` calculado por separado; tras el cambio, coincide exacto con
`1 - cos_sim`.

**Persistencia real, no solo teórica**: antes de indexar, `_coleccion_ya_poblada()` abre la
colección persistida y chequea `._collection.count() > 0`. Si ya tiene documentos, `ingest_documentos()`
devuelve directamente el `Chroma` existente sin volver a fragmentar ni re-embeddear nada (salvo
`force_reindex=True`) — evidencia en el log de una corrida real, segunda vez que se llama:

```
INFO src.ingest: Colección 'manuales_tecnicos_dev' ya poblada en D:\...\vectorstore: se omite la re-indexación.
```

**Una sola conexión a Chroma por corrida, no una por pregunta**: `src/main.py` llama a
`ingest_documentos()` una única vez al principio y pasa el `Chroma` devuelto explícitamente a cada
`answer_question(..., vectorstore=vectorstore)`. Sin esto, cada pregunta reabriría su propia
conexión al `chroma.sqlite3` persistido (`answer_question` hace `vectorstore or
ingest_documentos()` -- si no se pasa `vectorstore`, reconecta desde cero) -- innecesario cuando ya
tenés la instancia a mano, y un riesgo real de contención si en el futuro esto corriera con
requests concurrentes sobre el mismo archivo SQLite.

### Cadena LCEL de generación grounded (`chain.py`)

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
- **Instrucción explícita de fidelidad técnica**: una vez resuelto el chunking, apareció un
  problema distinto — el modelo **sí** encontraba el contexto correcto, pero lo parafraseaba
  perdiendo datos puntuales (comandos, valores numéricos, nombres de parámetros) al redactar la
  respuesta. El prompt original no decía nada sobre *cómo* redactar cuando el contexto sí alcanza,
  solo sobre qué hacer cuando no alcanza. Se agregó una instrucción de "fidelidad técnica" —
  reproducir esos datos tal cual aparecen en el CONTEXTO, no resumirlos ni parafrasearlos aunque
  conserven el sentido general — con un ejemplo concreto anclado en un caso real (ver
  [Evidencia real de ejecución](#evidencia-real-de-ejecución)).
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
  pide al modelo responder "literalmente" `NO_CONTEXTO_MENSAJE` (constante en `src/schemas.py`)
  cuando no hay contexto suficiente, pero un LLM no siempre respeta esa instrucción al pie de la
  letra — se observó a Claude, en una corrida real, agregando una explicación extra después de un
  `\n\n` en vez de responder solo la frase pedida. `RespuestaLLM` tiene un `model_validator` que
  fuerza `respuesta` al mensaje canónico exacto cuando `contexto_encontrado=False`, sin importar
  qué texto haya generado el modelo — la consistencia de esa respuesta no depende de que el LLM
  cumpla la instrucción del prompt.
- **El paso de *retrieval* también está protegido, no solo la generación**: `answer_question()`
  envuelve `ingest_documentos()` + la búsqueda en Chroma en su propio `try/except` (loguea con
  contexto propio -- "Fallo recuperando contexto..." -- y re-lanza, mismo criterio de "logueá y
  propagá" que ya usaba el paso de generación). Antes de este cambio, una falla de ChromaDB o del
  modelo de embeddings durante el retrieval salía como un traceback sin explicación, distinto del
  path de error ya cubierto de la generación.
- **El score de similitud queda visible en el log**: se usa
  `vectorstore.asimilarity_search_with_score()` en vez de `as_retriever().ainvoke()` -- este último
  descarta el score, el primero lo devuelve junto con cada `Document`. Chroma devuelve una
  **distancia** (con la colección configurada a `hnsw:space="cosine"`, ver sección de ingesta, es
  `1 - similitud_coseno`; `0.0` es lo más parecido). En el log se muestra invertido, como
  **similitud** (`1 - distancia`, en porcentaje) porque es más intuitivo de leer: `100%` es lo más
  parecido, no `0.0`. Ej.: `arquitectura_sistema.md (similitud=54.26%)`.
- **El contenido completo de cada chunk recuperado se loguea a nivel `DEBUG`** (no `INFO`, para no
  ensuciar el log por default): fue justamente inspeccionando esto durante el desarrollo que se
  detectó el problema real que motivó el splitter jerárquico (ver
  [Módulo de ingesta](#módulo-de-ingesta-ingestpy) y
  [Evidencia real de ejecución](#evidencia-real-de-ejecución)) -- se dejó como herramienta de
  diagnóstico permanente, no como algo puntual. Para verlo, subí el nivel del logger a `DEBUG` en
  `src/main.py` (o el que corresponda).

`answer_question()` es el punto de entrada end-to-end:

1. **Retrieval**: `vectorstore.asimilarity_search_with_score(pregunta, k=top_k)` — `top_k=4`,
   dentro del rango 3-5 que recomienda la consigna para evitar el "contexto infinito" (degradación
   por *lost in the middle* y gasto de tokens de más).
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

Corrida real (`python -m src.main`, embeddings
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` local, generación
`provider="anthropic"`). Los JSON de esta sección son de corridas anteriores a que la salida
incluyera `fragmentos_recuperados`; se dejan como registro de cómo evolucionó el sistema. La
salida actual completa está en [Ejemplo de entrada/salida](#ejemplo-de-entradasalida) y en
[`evidencia/`](evidencia/).

**Pregunta con respuesta clara en el contexto:**

```json
{
  "pregunta": "¿Qué componente es el cuello de botella histórico del sistema bajo carga alta, y por qué?",
  "respuesta": "El cuello de botella histórico del sistema bajo carga alta es el **pool de conexiones a PostgreSQL**. Específicamente, cuando hay más de 500 pedidos concurrentes, el límite de 100 conexiones de PgBouncer se satura, lo que causa que las nuevas requests queden esperando una conexión libre, generando timeouts intermitentes en el endpoint `/v1/pedidos`.",
  "contexto_encontrado": true,
  "fuentes": ["arquitectura_sistema.md", "monitoreo_alertas.md"]
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

**Segunda instancia del mismo trade-off, tras cambiar a un modelo de embeddings multilingüe**: al
pasar de `all-MiniLM-L6-v2` a `paraphrase-multilingual-MiniLM-L12-v2` (ver sección de
[Diseño](#embeddings-locales-hugging-face)), el *ranking* de similitud cambia -- son modelos
distintos, con espacios semánticos distintos -- y una pregunta que antes se respondía bien puede
dejar de estarlo, y viceversa. Con el modelo multilingüe (y todavía con el splitter plano), la
pregunta "¿Qué umbral de uso de conexiones de PgBouncer dispara una alerta?" recuperó un fragmento
distinto de `monitoreo_alertas.md` (la sección de "Dashboards de referencia", que no menciona
ningún umbral) en vez del fragmento con "Alerta al superar el 80% de uso sostenido durante más de
2 minutos" -- y el sistema, correctamente, volvió a responder "No lo sé" en vez de inventar un
número. No fue una regresión del cambio de modelo: fue evidencia de que el *ranking* de similitud
es sensible al modelo de embeddings usado, y de que el sistema se comporta de forma consistente
(honesto ante la falta del dato puntual) sin importar cuál sea la causa concreta de que el
fragmento correcto no entre en el `top_k`.

**Ambos casos, resueltos con el splitter jerárquico** (`MarkdownHeaderTextSplitter` +
`RecursiveCharacterTextSplitter` como fallback, ver
[Módulo de ingesta](#módulo-de-ingesta-ingestpy)): al mantener cada sección de encabezado completa
en un solo fragmento en vez de cortar ciegamente por cantidad de caracteres, las dos preguntas que
antes daban "No lo sé" ahora recuperan la sección correcta completa y responden con el dato
puntual real:

```json
{
  "pregunta": "¿Qué pasos hay que seguir para resolver un agotamiento del pool de conexiones a PostgreSQL?",
  "respuesta": "Según el runbook de incidentes, los pasos para resolver un agotamiento del pool de conexiones a PostgreSQL son:\n\n1. Verificar en el dashboard de PgBouncer cuántas conexiones están activas vs. el límite (100).\n\n2. Si hay una query \"colgada\" reteniendo conexiones, identificarla con:\n   ```\n   SELECT * FROM pg_stat_activity WHERE state = 'active' ORDER BY query_start;\n   ```\n   y, si corresponde, cancelarla con `pg_cancel_backend(pid)`.\n\n3. Si el problema es puramente de volumen (no hay queries colgadas), escalar horizontalmente el número de réplicas de PgBouncer o subir el límite de conexiones a 150.",
  "contexto_encontrado": true,
  "fuentes": ["arquitectura_sistema.md", "monitoreo_alertas.md", "runbook_incidentes.md"]
}
```

*Nota sobre esta evidencia*: en la primera corrida tras el splitter jerárquico, el paso 3 de esta
respuesta decía "escalar horizontalmente el sistema" -- el modelo sí había encontrado el contexto
correcto, pero al redactar perdió los datos puntuales ("réplicas de PgBouncer", "150") en una
paráfrasis genérica. Ese fue el hallazgo que motivó agregar la instrucción de "fidelidad técnica"
al prompt (ver [Cadena LCEL](#cadena-lcel-de-generación-grounded-chainpy)); el JSON de arriba ya
refleja el comportamiento corregido.

```json
{
  "pregunta": "¿Qué umbral de uso de conexiones de PgBouncer dispara una alerta?",
  "respuesta": "El umbral de uso de conexiones de PgBouncer que dispara una alerta es el 80% de uso sostenido durante más de 2 minutos. Dado que el límite configurado es de 100 conexiones, esto significa que se genera una alerta cuando se superan 80 conexiones activas de forma sostenida durante ese período, ya que este es el precursor directo del agotamiento del pool de conexiones.",
  "contexto_encontrado": true,
  "fuentes": ["arquitectura_sistema.md", "monitoreo_alertas.md", "runbook_incidentes.md"]
}
```

Importante: esto **no invalida** el trade-off explicado arriba -- con un documento lo bastante
grande, o una pregunta cuyo dato puntual quede en una sub-sección `###` distinta de la que el
`top_k` elegido trae, el mismo tipo de "No lo sé" correcto puede volver a aparecer. Lo que cambió
es que ahora el corte de chunking respeta los límites semánticos del documento (nunca parte una
sección a la mitad *por casualidad* de dónde cae el caracter 1000), así que cuando falla, falla por
una razón real de cobertura de `top_k`/relevancia -- no por un accidente de dónde cortó un splitter
ciego al contenido.

## Errores comunes evitados (según la consigna)

- **Contexto infinito**: `top_k=4` (rango 3-5 recomendado), no se pasan decenas de fragmentos.
- **Embeddings no coincidentes**: `_build_embeddings()` es la única función que construye
  embeddings en todo el proyecto; la usan tanto la ingesta como la consulta.
- **Falta de persistencia**: `_coleccion_ya_poblada()` evita re-indexar si la colección ya existe
  y tiene documentos (ver log de ejemplo más arriba).

## Tests

```bash
pytest -v
```

39 tests, sin llamadas de red reales (mismo criterio que los Módulos 1 y 2):

- `test_schemas.py`: validación de `RespuestaLLM` (respuesta no vacía, limpieza de espacios,
  campos requeridos, tipos, normalización de `respuesta` a `NO_CONTEXTO_MENSAJE` cuando
  `contexto_encontrado=False`) y `RespuestaRAG` (fuentes default vacía, pregunta/respuesta no
  vacías).
- `test_ingest.py`: usa un `FakeEmbeddings` determinístico (mismo texto → mismo vector, sin red)
  inyectado vía monkeypatch de `_build_embeddings`, con ChromaDB real apuntando a un directorio
  temporal. Cubre indexación real, filtrado de archivos no `.txt`/`.md`, no-reindexación si la
  colección ya está poblada, `force_reindex=True`, colecciones `dev`/`prod` independientes
  dentro del mismo directorio, y error si el directorio de documentos está
  vacío. `TestFragmentarDocumento` prueba el splitter jerárquico en aislamiento: una sección corta
  mantiene junto su encabezado y contenido, una sección larga activa el fallback de
  `RecursiveCharacterTextSplitter` sin mezclar la sección vecina, y un texto sin encabezados
  Markdown cae directo al fallback.
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
  cuando `contexto_encontrado=False`, y que `fragmentos_recuperados` expone fuente, sección y
  similitud (`1 - distancia`) de cada fragmento. También cubre que una falla durante el retrieval (ej.
  ChromaDB caído) se loguea con el mensaje "Fallo recuperando contexto..." y se propaga, en vez
  de tragarse en silencio.

La fixture `fast_retries` (en `tests/conftest.py`) parchea `asyncio.sleep` para neutralizar
el backoff real de `.with_retry()` durante los tests, mismo criterio que el Módulo 2.
