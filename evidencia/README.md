# Evidencia de ejecución

Salidas reales, sin editar, generadas el 2026-09-24 con Python 3.12 en Windows
(embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` locales,
generación con `provider="anthropic"`, modelo `claude-haiku-4-5-20251001`).

| Archivo | Comando | Qué demuestra |
|---|---|---|
| [`corrida_1_indexacion.txt`](corrida_1_indexacion.txt) | `python -m rag.main` con `./vectorstore` borrado | Ingesta: 4 documentos → 24 fragmentos persistidos en ChromaDB; retrieval con `top_k=4` y similitud por fragmento; 4 respuestas con contexto y 1 "No lo sé" |
| [`corrida_2_persistencia.txt`](corrida_2_persistencia.txt) | `python -m rag.main` otra vez, con `./vectorstore` ya creado | Persistencia: `Colección 'manuales_tecnicos' ya poblada ... se omite la re-indexación`; mismos fragmentos y mismas similitudes que en la corrida 1 (mismo modelo de embeddings para indexar y consultar) |
| [`tests_pytest.txt`](tests_pytest.txt) | `pytest rag/tests/ -v` | 37 tests en verde, sin llamadas de red |

Notas para leer los logs:

- Las líneas de log (`INFO ...`) salen por stderr y las respuestas JSON por stdout. Como
  stdout queda en buffer al redirigirse a un archivo, en el `.txt` los logs aparecen todos
  primero y las respuestas después. El orden real de ejecución es el de los timestamps.
- La similitud del log es `1 - distancia coseno` de Chroma (100% = idéntico).
- En la pregunta sobre el pool de conexiones, el paso 3 de la respuesta dice "subir el
  límite de conexiones" sin el valor puntual ("a 150") que sí figura en
  `rag/docs/runbook_incidentes.md`. Es una paráfrasis del LLM que la instrucción de
  fidelidad técnica del prompt reduce pero no elimina del todo. Queda registrada tal cual
  salió, sin retocar.
