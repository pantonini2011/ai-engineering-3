import pytest


@pytest.fixture(autouse=True)
def dummy_api_keys(monkeypatch):
    """API keys dummy para que construir los clientes (ChatOpenAI, etc.) no
    falle por falta de credenciales -- ningún test pega a la red real."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Neutraliza el backoff real de `.with_retry()` (tenacity) durante los
    tests -- mismo criterio que el Módulo 2."""
    import asyncio

    async def _no_sleep(*args, **kwargs):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


@pytest.fixture(autouse=True)
def conteo_de_tokens_sin_red(monkeypatch):
    """El chunking mide en tokens del modelo de embeddings, cuyo tokenizer se
    descarga del Hub. En los tests se cuenta una palabra = un token, sin red."""
    from src import ingestion

    monkeypatch.setattr(ingestion, "_contar_tokens", lambda texto: len(texto.split()))
