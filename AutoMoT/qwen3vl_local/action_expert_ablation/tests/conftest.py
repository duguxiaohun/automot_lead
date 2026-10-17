"""Explicit isolated source trees for contract-only tests, never runtime substitutes."""
import pytest
from qwen3vl_local.action_expert_ablation import common


@pytest.fixture
def ablation_source_tree(tmp_path, monkeypatch):
    root = tmp_path / 'contract_sources'
    for variant in common.VARIANTS:
        for name in common.contract_source_paths(variant):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# isolated test-only source: '+name+'\n')
    monkeypatch.setattr(common, 'AUTOMOT_ROOT', root)
    return root
