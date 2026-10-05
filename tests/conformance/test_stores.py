from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentic_rag.adapters.milvus import MilvusSettings, MilvusStore
from agentic_rag.testing import FakeVectorStore, VectorStoreContract


class TestFakeVectorStore(VectorStoreContract):
    async def create(self) -> Any:
        return FakeVectorStore()

    async def create_failing(self) -> Any:
        return FakeVectorStore(unavailable=True)


@pytest.mark.milvus
class TestMilvusStore(VectorStoreContract):
    @pytest.fixture(autouse=True)
    def _wire(self, milvus_uri: str, collection_name: str) -> None:
        self._uri, self._collection = milvus_uri, collection_name

    async def create(self) -> Any:
        return MilvusStore(MilvusSettings(uri=self._uri, collection=self._collection))

    async def create_failing(self) -> Any:
        return MilvusStore(MilvusSettings(uri="http://127.0.0.1:1", collection=self._collection))


class TestChromaStore(VectorStoreContract):
    @pytest.fixture(autouse=True)
    def _wire(self, tmp_path: Path) -> None:
        pytest.importorskip("chromadb")
        self._path = str(tmp_path / "chroma")

    async def create(self) -> Any:
        from agentic_rag.adapters.chroma import ChromaSettings, ChromaStore

        return ChromaStore(ChromaSettings(path=self._path, collection="conformance"))

    async def create_failing(self) -> Any:
        from agentic_rag.adapters.chroma import ChromaSettings, ChromaStore

        return ChromaStore(ChromaSettings(url="http://127.0.0.1:1", collection="conformance"))


class TestQdrantStore(VectorStoreContract):
    @pytest.fixture(autouse=True)
    def _wire(self, tmp_path: Path) -> None:
        pytest.importorskip("qdrant_client")
        self._path = str(tmp_path / "qdrant")

    async def create(self) -> Any:
        from agentic_rag.adapters.qdrant import QdrantSettings, QdrantStore

        return QdrantStore(QdrantSettings(location=self._path, collection="conformance"))

    async def create_failing(self) -> Any:
        from agentic_rag.adapters.qdrant import QdrantSettings, QdrantStore

        return QdrantStore(QdrantSettings(location="http://127.0.0.1:1", collection="conformance"))
