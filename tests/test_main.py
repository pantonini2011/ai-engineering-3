import pytest

from src import main as main_module


def test_default_es_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert main_module.proveedor_configurado() == "anthropic"


def test_openai_con_su_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
    assert main_module.proveedor_configurado() == "openai"


def test_openai_sin_api_key_corta_con_mensaje_claro(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        main_module.proveedor_configurado()


@pytest.mark.parametrize("placeholder", ["tu_openai_api_key", "sk-..."])
def test_api_key_placeholder_cuenta_como_faltante(monkeypatch, placeholder):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", placeholder)
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        main_module.proveedor_configurado()


def test_ollama_no_requiere_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert main_module.proveedor_configurado() == "ollama"


def test_proveedor_desconocido(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "cohere")
    with pytest.raises(SystemExit, match="no soportado"):
        main_module.proveedor_configurado()
