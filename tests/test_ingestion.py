from datetime import datetime

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src import client as client_module
from src import ingestion as ingest_module


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
    monkeypatch.setattr(client_module, "build_embeddings", lambda: FakeEmbeddings())


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


def test_ingesta_guarda_esquema_de_metadatos(docs_dir, persist_dir):
    """Cada fragmento persistido lleva source, env y created_at (ISO 8601)."""
    vectorstore = ingest_module.ingest_documentos(directorio=str(docs_dir), persist_directory=persist_dir)
    metadatas = vectorstore.get()["metadatas"]
    assert metadatas
    for m in metadatas:
        assert m["source"] in ("uno.md", "dos.txt")
        assert m["env"] == client_module.RAG_ENV
        datetime.fromisoformat(m["created_at"])


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
    original_vs = ingest_module.ingest_documentos(directorio=str(docs_dir), persist_directory=persist_dir)
    cantidad_original = original_vs._collection.count()

    llamadas = []
    original = ingest_module.Chroma.from_documents

    def spy(*args, **kwargs):
        llamadas.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingest_module.Chroma, "from_documents", spy)
    vectorstore = ingest_module.ingest_documentos(
        directorio=str(docs_dir), persist_directory=persist_dir, force_reindex=True
    )
    assert len(llamadas) == 1
    # Re-indexar reemplaza la colección, no le suma una segunda copia.
    assert vectorstore._collection.count() == cantidad_original


def test_colecciones_por_entorno_no_se_pisan(docs_dir, persist_dir):
    """Segmentación dev/prod: dos colecciones en el mismo persist_directory
    son independientes -- poblar la de dev no hace que prod se considere
    poblada (y viceversa)."""
    ingest_module.ingest_documentos(
        directorio=str(docs_dir), persist_directory=persist_dir, collection_name="manuales_tecnicos_dev"
    )
    embeddings = client_module.build_embeddings()
    assert client_module.coleccion_ya_poblada(persist_dir, "manuales_tecnicos_dev", embeddings)
    assert not client_module.coleccion_ya_poblada(persist_dir, "manuales_tecnicos_prod", embeddings)


def test_ingesta_sin_documentos_lanza_error(tmp_path, persist_dir):
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    with pytest.raises(ValueError):
        ingest_module.ingest_documentos(directorio=str(vacio), persist_directory=persist_dir)


class TestFragmentarDocumento:
    def test_mantiene_junta_una_seccion_corta_con_su_encabezado(self):
        texto = (
            "# Título\n\n"
            "## Incidente 1\n\n"
            "**Síntoma**: algo falla.\n\n"
            "**Resolución**: reiniciar el servicio.\n"
        )
        documento = Document(page_content=texto, metadata={"source": "runbook.md"})
        fragmentos = ingest_module._fragmentar_documento(documento)

        # Una sola sección (cabe entera dentro de CHUNK_SIZE): sintoma y
        # resolución quedan en el MISMO fragmento, no separados.
        assert len(fragmentos) == 1
        assert "Síntoma" in fragmentos[0].page_content
        assert "Resolución" in fragmentos[0].page_content
        assert fragmentos[0].metadata["source"] == "runbook.md"
        assert fragmentos[0].metadata["Header 2"] == "Incidente 1"

    def test_aplica_fallback_dentro_de_una_seccion_larga_sin_mezclar_otra(self):
        seccion_larga = "Contenido de relleno para forzar el fallback. " * 40  # > CHUNK_SIZE
        texto = f"## Sección larga\n\n{seccion_larga}\n\n## Sección corta\n\nTexto breve.\n"
        documento = Document(page_content=texto, metadata={"source": "doc.md"})
        fragmentos = ingest_module._fragmentar_documento(documento)

        de_la_larga = [f for f in fragmentos if f.metadata.get("Header 2") == "Sección larga"]
        de_la_corta = [f for f in fragmentos if f.metadata.get("Header 2") == "Sección corta"]

        # La sección larga se partió en más de un fragmento (fallback)...
        assert len(de_la_larga) > 1
        # ...pero ninguno de esos fragmentos contiene texto de la otra sección.
        for f in de_la_larga:
            assert "Texto breve" not in f.page_content
        # La sección corta, en cambio, quedó en un solo fragmento.
        assert len(de_la_corta) == 1

    def test_texto_sin_encabezados_cae_directo_al_fallback(self):
        texto = "Contenido plano sin ningún encabezado Markdown. " * 40
        documento = Document(page_content=texto, metadata={"source": "plano.txt"})
        fragmentos = ingest_module._fragmentar_documento(documento)

        assert len(fragmentos) > 1
        assert all(f.metadata["source"] == "plano.txt" for f in fragmentos)
