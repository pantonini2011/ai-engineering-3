# Evidencia de ejecución

Salidas reales, sin editar, generadas el 2026-09-24 con Python 3.12 en Windows
(embeddings `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` locales, dimensión 384;
generación con `provider="anthropic"`, modelo `claude-haiku-4-5-20251001`; `RAG_ENV=dev`).

| Archivo | Comando | Qué demuestra |
|---|---|---|
| [`corrida_1_indexacion.txt`](corrida_1_indexacion.txt) | `python -m src.main` con `./vectorstore` borrado | Ingesta: 4 documentos → 24 fragmentos persistidos en la colección `manuales_tecnicos_dev`; retrieval con `top_k=4`; 4 respuestas con contexto, 1 "No lo sé" y 1 consulta con filtro de metadata (`source = runbook_incidentes.md`); cada respuesta con sus `fragmentos_recuperados` (similitud, extracto y metadata) |
| [`corrida_2_persistencia.txt`](corrida_2_persistencia.txt) | `python -m src.main` otra vez, con `./vectorstore` ya creado | Persistencia: `Colección 'manuales_tecnicos_dev' ya poblada ... se omite la re-indexación`; mismos fragmentos y mismas similitudes que en la corrida 1 (mismo modelo de embeddings para indexar y consultar) |
| [`tests_pytest.txt`](tests_pytest.txt) | `pytest -v` | 45 tests en verde, sin llamadas de red ni API keys |

Notas para leer los logs:

- Las líneas de log (`INFO ...`) salen por stderr y las respuestas JSON por stdout. Como
  stdout queda en buffer al redirigirse a un archivo, en el `.txt` los logs aparecen todos
  primero y las respuestas después. El orden real de ejecución es el de los timestamps.
- La similitud es `1 - distancia coseno` de Chroma: en el log se muestra en porcentaje
  (`similitud=67.01%`) y en el JSON como fracción (`"similitud": 0.6701`).
- Las respuestas del LLM pueden variar levemente de una corrida a otra (redacción, nivel de
  detalle). Los fragmentos recuperados y sus similitudes no: dependen solo de los embeddings,
  que son determinísticos.
