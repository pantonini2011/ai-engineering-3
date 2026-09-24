import pytest
from langchain_core.documents import Document

from src import client as client_module
from src import ingestion as ingest_module
from src import retriever as retriever_module
from tests.test_ingestion import FakeEmbeddings


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(client_module, "build_embeddings", lambda: FakeEmbeddings())


@pytest.fixture
def vectorstore(tmp_path):
    """ChromaDB real en un directorio temporal, con dos documentos."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "runbook.md").write_text("# Runbook\n\n## Incidente 1\n\nPool de conexiones agotado.", encoding="utf-8")
    (docs / "normativa.md").write_text("# Normativa\n\n## Despliegues\n\nNo se despliega los viernes.", encoding="utf-8")
    return ingest_module.ingest_documentos(directorio=str(docs), persist_directory=str(tmp_path / "vs"))


async def test_filtro_por_source_restringe_la_busqueda(vectorstore):
    resultados = await retriever_module.buscar_fragmentos(
        vectorstore, "conexiones", top_k=4, filtro={"source": "runbook.md"}
    )
    assert resultados
    assert {doc.metadata["source"] for doc, _ in resultados} == {"runbook.md"}


async def test_sin_filtro_busca_en_toda_la_coleccion(vectorstore):
    resultados = await retriever_module.buscar_fragmentos(vectorstore, "conexiones", top_k=4)
    assert {doc.metadata["source"] for doc, _ in resultados} == {"runbook.md", "normativa.md"}


async def test_filtro_por_env(vectorstore):
    """El filtro por `env` trae todo lo indexado en el entorno actual, y nada
    si se pide otro entorno."""
    del_entorno = await retriever_module.buscar_fragmentos(
        vectorstore, "conexiones", top_k=4, filtro={"env": client_module.RAG_ENV}
    )
    de_otro = await retriever_module.buscar_fragmentos(vectorstore, "conexiones", top_k=4, filtro={"env": "otro"})
    assert len(del_entorno) == 2
    assert de_otro == []


def test_extracto_se_corta_y_normaliza_espacios():
    texto = "linea uno\n\n" + "x" * 200
    fragmentos = retriever_module.a_fragmentos_recuperados(
        [(Document(page_content=texto, metadata={"source": "a.md"}), 0.5)]
    )
    assert fragmentos[0].extracto.startswith("linea uno xxx")
    assert fragmentos[0].extracto.endswith("...")
    assert len(fragmentos[0].extracto) == retriever_module.LARGO_EXTRACTO + 3
