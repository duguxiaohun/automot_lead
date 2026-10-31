import pytest
from qwen3vl_local.qwen35 import gpu_selection as gpu


@pytest.fixture
def cards(monkeypatch):
    monkeypatch.delenv('GPU_IDS',raising=False)
    rows={str(i):[0,0] for i in range(8)}
    monkeypatch.setattr(gpu.subprocess,'check_output',lambda *a,**k:
        '\n'.join(f'{i},{v[0]},{v[1]}' for i,v in reversed(list(rows.items()))))
    return rows


def test_training_uses_fixed_ids_despite_inventory_order(cards):
    assert gpu.select()==['0','1','2','3']


@pytest.mark.parametrize('usage',[[256,0],[0,6]])
def test_busy_default_refuses_other_idle_cards(cards,usage):
    cards['0']=usage
    with pytest.raises(ValueError,match='busy'):gpu.select()
    assert gpu.select(4,'4,5,6,7')==['4','5','6','7']


def test_missing_default_refuses_other_idle_cards(cards):
    del cards['0']
    with pytest.raises(ValueError,match='distinct'):gpu.select()


def test_single_card_probe_still_automatically_selects_idle(cards):
    cards['7']=[256,0]
    assert gpu.select(1)==['6']


@pytest.mark.parametrize('value',['0','0,1,2,2','0,1,2,9'])
def test_explicit_invalid_mask_is_not_silently_overridden(cards,monkeypatch,value):
    monkeypatch.setenv('GPU_IDS',value)
    with pytest.raises(ValueError,match='distinct'):gpu.select()
