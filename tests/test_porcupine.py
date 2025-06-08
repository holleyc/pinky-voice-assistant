import os
import pytest

# If you haven’t exported a real key, skip all tests in this module:
ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
if not ACCESS_KEY:
    pytest.skip(
        "PICOVOICE_ACCESS_KEY not set, skipping Porcupine keyword-detector tests",
        allow_module_level=True
    )

import pvporcupine

# Create the detector once for all tests
porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keywords=["picovoice"]
)

def test_process_exists_and_callable():
    """Smoke-test that .process() exists and is callable."""
    assert hasattr(porcupine, "process"), "Porcupine instance has no .process()"
    assert callable(porcupine.process), ".process should be callable"

def test_frame_length_and_sample_rate():
    """
    The pvporcupine object should expose frame_length and sample_rate,
    which are needed by your audio loop.
    """
    assert isinstance(porcupine.frame_length, int) and porcupine.frame_length > 0
    assert isinstance(porcupine.sample_rate, int) and porcupine.sample_rate > 0
