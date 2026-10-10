import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.fakes import FakeLLM, InMemoryStore

KEY = "test-key"
HEADERS = {"X-Internal-Api-Key": KEY}


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(internal_api_key=KEY, google_cloud_project="test", knowledge_dir=str(tmp_path / "nutrition"),
                    rag_min_score=0.2, _env_file=None)


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def store(settings) -> InMemoryStore:
    return InMemoryStore(settings)


@pytest.fixture
def client(settings, llm, store):
    with TestClient(create_app(settings, llm=llm, store=store)) as c:
        yield c
