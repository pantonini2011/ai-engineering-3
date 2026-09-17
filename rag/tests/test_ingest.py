import pytest
from langchain_core.embeddings import Embeddings

from rag import ingest as ingest_module


class FakeEmbeddings(Embeddings):
    """Embeddings determinísticos y locales: mismo texto -> mismo vector,
    sin llamar a ninguna API. Alcanza para probar que el pipeline de ingesta
    fragmenta, persiste y detecta una colección ya poblada, sin depender de
    la red ni gastar créditos."""

    def _vector(self, text: str):
        h = abs(hash(text))
        return [float((h >> (8 * i)) % 100) for i in range(8)]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(ingest_module, "_build_embeddings", lambda: FakeEmbeddings())


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "uno.md").write_text("Contenido del primer documento de prueba. " * 20, encoding="utf-8")
    (d / "dos.txt").write_text(
        "Contenido del segundo documento de prueba, distinto del primero. " * 20, encoding="utf-8"
    )
    (d / "ignorar.json").write_text("{}", encoding="utf-8")
    return d


@pytest.fixture
def persist_dir(tmp_path):
    return str(tmp_path / "vectorstore_test")


def test_ingesta_indexa_y_persiste(docs_dir, persist_dir):
    vectorstore = ingest_module.ingest_documentos(directorio=str(docs_dir), persist_directory=persist_dir)
    assert vectorstore._collection.count() > 0


def test_ingesta_ignora_archivos_no_txt_md(docs_dir, persist_dir):
    vectorstore = ingest_module.ingest_documentos(directorio=str(docs_dir), persist_directory=persist_dir)
    fuentes = {m["source"] for m in vectorstore.get()["metadatas"]}
    assert "ignorar.json" not in fuentes
    assert "uno.md" in fuentes
    assert "dos.txt" in fuentes


def test_ingesta_no_reindexa_si_ya_esta_poblada(docs_dir, persist_dir, monkeypatch):
    ingest_module.ingest_documentos(directorio=str(docs_dir), persist_directory=persist_dir)

    llamadas = []
    original = ingest_module.Chroma.from_documents

    def spy(*args, **kwargs):
        llamadas.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest_module.Chroma, "from_documents", spy)
    ingest_module.ingest_documentos(directorio=str(docs_dir), persist_directory=persist_dir)
    assert len(llamadas) == 0


def test_ingesta_force_reindex_vuelve_a_indexar(docs_dir, persist_dir, monkeypatch):
    ingest_module.ingest_documentos(directorio=str(docs_dir), persist_directory=persist_dir)

    llamadas = []
    original = ingest_module.Chroma.from_documents

    def spy(*args, **kwargs):
        llamadas.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest_module.Chroma, "from_documents", spy)
    ingest_module.ingest_documentos(directorio=str(docs_dir), persist_directory=persist_dir, force_reindex=True)
    assert len(llamadas) == 1


def test_ingesta_sin_documentos_lanza_error(tmp_path, persist_dir):
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    with pytest.raises(ValueError):
        ingest_module.ingest_documentos(directorio=str(vacio), persist_directory=persist_dir)
