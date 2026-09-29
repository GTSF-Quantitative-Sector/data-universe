import pytest

from data_universe import config


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def test_polygon_key_defaults_to_none_and_is_settable():
    assert config.get_polygon_key() is None
    config.set_polygon_key("secret")
    assert config.get_polygon_key() == "secret"


def test_polygon_base_url_default():
    assert config.get_polygon_base_url() == "https://api.polygon.io"


def test_polygon_base_url_settable():
    config.set_polygon_base_url("https://example.test")
    assert config.get_polygon_base_url() == "https://example.test"


def test_project_defaults_to_none_and_is_settable():
    assert config.get_project() is None
    config.set_project("vrp-research")
    assert config.get_project() == "vrp-research"


def test_cache_size_default_is_256():
    assert config.get_cache_size() == 256


def test_cache_size_settable():
    config.set_cache_size(64)
    assert config.get_cache_size() == 64


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("POLYGON_BASE_URL", "https://from-env.test")
    config.reset()
    assert config.get_polygon_base_url() == "https://from-env.test"


def test_load_yaml_sets_values_not_set_by_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    config.reset()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("fred_api_key: from-yaml\n")
    config.load_yaml(str(yaml_path))
    assert config.get_fred_key() == "from-yaml"


def test_load_yaml_does_not_override_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    config.reset()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("fred_api_key: from-yaml\n")
    config.load_yaml(str(yaml_path))
    assert config.get_fred_key() == "from-env"


def test_load_yaml_missing_file_is_a_noop(tmp_path):
    config.reset()
    config.load_yaml(str(tmp_path / "does-not-exist.yaml"))
    assert config.get_fred_key() is None


def test_data_dir_default_is_under_package_parent():
    assert config.get_data_dir().endswith("data")


def test_massive_defaults(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_BASE_URL", raising=False)
    config.reset()
    assert config.get_massive_key() is None
    assert config.get_massive_base_url() == "https://api.massive.com"


def test_massive_setters():
    config.set_massive_key("k")
    config.set_massive_base_url("https://example.test")
    assert config.get_massive_key() == "k"
    assert config.get_massive_base_url() == "https://example.test"


def test_load_yaml_sets_massive_values(tmp_path, monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("MASSIVE_BASE_URL", raising=False)
    config.reset()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("massive_api_key: from-yaml\nmassive_base_url: https://yaml.test\n")
    config.load_yaml(str(yaml_path))
    assert config.get_massive_key() == "from-yaml"
    assert config.get_massive_base_url() == "https://yaml.test"
