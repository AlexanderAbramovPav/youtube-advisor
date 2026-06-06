import pathlib
import pytest

from youtube_advisor.search import _get_model


@pytest.fixture
def fixtures_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "fixtures"

@pytest.fixture
def cassettes_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent / "cassettes"


@pytest.fixture(autouse=True)
def _clear_embed_model_cache():
    """Clear the cached TextEmbedding model between tests so per-test mocks
    of `youtube_advisor.search.TextEmbedding` take effect."""
    _get_model.cache_clear()
    yield
    _get_model.cache_clear()


@pytest.fixture(autouse=True)
def _no_polite_sleep(monkeypatch):
    """The orchestrator inserts 0.5-2s random sleeps between captions calls
    when no cookies are set (anti-bot mitigation). Tests mock the network so
    those sleeps would only slow CI — patch them to no-op everywhere."""
    monkeypatch.setattr("youtube_advisor.ingest.orchestrator.time.sleep",
                        lambda *_a, **_kw: None)
