# tests/test_memory_profile.py

import os
import memory  # or: from memory import get_user_profile, update_user_fact, etc.

def test_profile_roundtrip(tmp_path, monkeypatch):
    # Override PROFILE_DIR to the test temp directory
    monkeypatch.setenv("PROFILE_DIR", str(tmp_path))

    uid = "testuser"

    # Initially blank profile
    prof = memory.get_user_profile(uid)
    assert prof == {"facts": {}}

    # Update a fact and verify file saved
    memory.update_user_fact(uid, "name", "Charlie")

    # Ensure file was saved to the test path
    pfile = tmp_path / f"{uid}.json"
    assert pfile.exists()
