"""Action's import namespace for the single explicitly reviewed Phase3 release."""
from qwen3vl_local.action_prior.phase3_release import active_release

_directory, _manifest, _selection = active_release()
__path__ = [str(_directory)]
DATASET_NAME = _manifest['dataset_name']
