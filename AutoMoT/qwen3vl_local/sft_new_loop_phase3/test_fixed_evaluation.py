from dataclasses import replace
from collections import Counter
import pytest
from qwen3vl_local.sft_new_loop_phase3.test_validation_balance import candidate_rows
from qwen3vl_local.sft_new_loop_phase3 import train
from qwen3vl_local.sft_new_loop_phase3.sequence_evaluation import select_work
from qwen3vl_local.sft_new_loop_phase3.invalid_balance import case_identity


@pytest.mark.parametrize('mode',['binary','choice'])
def test_production_periodic_validation_equal_after_actual_dedup(mode):
    rows=candidate_rows();rows.append(replace(rows[0],frame_id=10000))
    work,report=train._validation_work(rows,target_per_bin=64,seed=17,action_output_mode=mode,equal_events=True)
    valid=[i for i in work if not i.row.answers.get('INVALID_ACTION_CONTEXT',False)]
    assert len(set(Counter(i.row.context_id for i in valid).values()))==1
    assert len(work)==len({case_identity(i) for i in work})
    assert report['per_event']==1
    assert select_work(work,seed=17,limit=64)[0]==work
