import pytest

from src import main as main_module


@pytest.fixture(autouse=True)
def sin_config_de_proveedor(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)


def test_default_anthropic_con_fallback_openai():
    assert main_module.resolver_proveedores() == ("anthropic", "openai")


def test_openai_como_principal_con_fallback_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "OpenAI")
    assert main_module.resolver_proveedores() == ("openai", "anthropic")


def test_fallback_configurable(monkeypatch):
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "ollama")
    assert main_module.resolver_proveedores() == ("anthropic", "ollama")


@pytest.mark.parametrize("valor", [None, "tu_anthropic_api_key", "sk-ant-..."])
def test_si_el_principal_no_tiene_key_usa_el_fallback(monkeypatch, valor):
    """No se ata el sistema a un solo proveedor: sin key de Anthropic (o con el
    placeholder de .env.example), corre con OpenAI en vez de cortar."""
    if valor is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", valor)
    assert main_module.resolver_proveedores() == ("openai", "anthropic")


def test_sin_ningun_proveedor_disponible_corta_con_mensaje_claro(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "tu_openai_api_key")
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY.*OPENAI_API_KEY"):
        main_module.resolver_proveedores()


def test_ollama_no_requiere_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main_module.resolver_proveedores() == ("ollama", "anthropic")


def test_proveedor_desconocido(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "cohere")
    with pytest.raises(SystemExit, match="no soportado"):
        main_module.resolver_proveedores()
