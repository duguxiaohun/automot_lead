from ...phase3_single_v2.tests.test_single import record
from .. import objective,regions

def prediction(r,ids):
    return dict(token_ids=ids,format_valid=[True]*4,contract=regions.contract('six'),
        sample_binding=objective.observation_binding(r),source='predicted')
