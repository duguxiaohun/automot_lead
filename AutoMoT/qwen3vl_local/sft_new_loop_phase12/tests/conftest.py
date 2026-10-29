import pytest
from qwen3vl_local.sft_new_loop_phase12 import history

@pytest.fixture(autouse=True)
def fixture_history_release(tmp_path,monkeypatch):
    # Synthetic bundles have an explicitly separate trusted release descriptor.
    # Production remains pinned to the shipped real-history SHA.
    monkeypatch.setattr(history,'RELEASE',tmp_path/'fixture_history_release.json')
