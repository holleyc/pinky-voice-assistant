# tests/test_chroma_utils.py
import pytest
from utils import chroma_utils

class DummyCollection:
    def __init__(self):
        self.last_add = None
    def add(self, **kwargs):
        self.last_add = kwargs

@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    dummy = DummyCollection()
    monkeypatch.setattr(chroma_utils, "chat_collection", dummy)
    return dummy

def test_save_user_name_args(fake_client):
    chroma_utils.save_user_name("u1", "X")
    # ensure add() was called with exactly:
    assert fake_client.last_add["documents"] == ["X"]
    assert fake_client.last_add["metadatas"] == [{"type":"user_name","user_id":"u1"}]
    assert fake_client.last_add["ids"] == ["user_name_u1"]
    # embeddings list has same length as documents
    assert len(fake_client.last_add["embeddings"]) == 1

def test_save_message_to_chroma_args(fake_client):
    chroma_utils.save_message_to_chroma("u2", "user", "hello")
    last = fake_client.last_add
    assert last["metadatas"][0]["role"] == "user"
    assert last["metadatas"][0]["user_id"] == "u2"
    assert last["documents"] == ["hello"]
