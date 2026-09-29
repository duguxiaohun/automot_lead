"""Reviewed evaluation-only transitions; no blanket source/package mismatch switch.

The six Qwen migration diffs are from 59a5d7fe2 to e4558406e. BEV-only
never calls the changed adapter/base/cache branches. Its generated-token RoPE
must retain the original full, non-interleaved convention. Numerical tests use
the actual old implementation; all other payload fields remain exact.
"""
from copy import deepcopy
from dataclasses import fields
from pathlib import Path


BEV_PROFILE = "bev_qwen35_environment_v1"
ENTRY_PROFILE = "ablation_evaluation_entry_v1"
# A pair authorizes only these exact reviewed file contents, never a pathname.
QWEN_SOURCE_PAIRS = {
    "qwen3vl_local/action_prior/config.py": (
        "768826e2d2ca4407049bfe04670b3fa10cf63edd47b28143fa5f56523de9b721",
        "699260049aababf800200f1568d96bfcd8534808ee35ade800d72fe0f4a6349a"),
    "qwen3vl_local/action_prior/contracts.py": (
        "45e2f2f28c3a3dc81c640644ebe7df874f6f4529820618c249ef24639ab67ff1",
        "40d129d91329f4441bcdc3206b33f69b21c7967ed69d01d87c621f4e4b499e53"),
    "qwen3vl_local/leadmot/config.py": (
        "f1cb5cf237587cdb399eb86880d8324dfadcb134afa1dfe37d87739ea66f4dd8",
        "acead33c29242439075b24bb50cc7b15aaca6e5217d7ca7d01aa055eb78a4500"),
    "qwen3vl_local/leadmot/decoder.py": (
        "1a1d42eda396931d2afe5a7a65e11230713036fe014d6e8bbaa68e8a09cbc145",
        "1fa351e3e3dd2ac25d38d88431377390c746fb32f117ec325cd9ed7d333e85bd"),
    "qwen3vl_local/leadmot/mot_block.py": (
        "8a9ea55126222207f884a80a7e7bb8509aca4382350ba8272211b5fad6e386e9",
        "a9ab8c9be772209ae7f427f3a8f6b6bda344e53aa3270e3997840ff26c4fce45"),
    "qwen3vl_local/leadmot/train.py": (
        "03499c8c3f519dcf1b1a6f9acfd0016bb20a2a6cd5684747a3d69672536cb77b",
        "9659593631d31476b52c9d877894746711ee76bb5e10dec7268156db61e1dff3"),
}
# This patch changes only the eval loader and its metrics provenance. Resume's
# require_contract remains strict. The pair also preserves current Qwen evals.
ENTRY_SOURCE_PAIRS = {
    "qwen3vl_local/action_expert_ablation/common.py": (
        "744866ede5b4d776222950f213e48da158bc4000b9a36d8a3da37b2b42366541",
        "dc7b48abf0284d472b342bad2e1c9f2198b73694704beab25d1377adc6f447c5"),
}
NEW_CONFIG_FIELDS = {"partial_rotary_factor", "mrope_interleaved", "qwen_full_attention_layers"}


def legacy_bev_config(state):
    """Recover old BEV semantics without changing any saved field or weight."""
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
    if (state.get("ablation_variant") != "bev_only"
            or state.get("schema") != "action_expert_ablation_checkpoint_v1"
            or state.get("trajectory_decoder") != "conditional_joint_trajectory_flow_matching_v2"):
        raise ValueError("legacy configuration supports BEV-only joint FM checkpoints only")
    saved = dict(state["decoder_config"])
    required = {field.name for field in fields(LeadMoTPlanningDecoderConfig)} - NEW_CONFIG_FIELDS
    if set(saved) != required:
        raise ValueError("legacy probe requires the complete pre-Qwen3.5 decoder config, without new fields")
    saved.update(partial_rotary_factor=1.0, mrope_interleaved=False, qwen_full_attention_layers=())
    config = LeadMoTPlanningDecoderConfig(**saved)
    config.validate_qwen_kv_shape()
    return config


def _reject(message):
    raise ValueError("ablation condition contract mismatch: " + message)


def require_evaluation_contract(state, actual):
    """Require exact identity, or an explicit reviewed evaluation transition.

Neither a user report nor a checkpoint flag grants compatibility. Callers build
the current contract from real assets, and still verify dataset and EMA hashes.
The returned record documents differing identities; it never rewrites them.
"""
    from qwen3vl_local.action_prior.contracts import digest, file_hash
    expected = state["condition_contract"]
    if expected.get("schema") == actual.get("schema") and expected.get("identity") == actual.get("identity"):
        return None
    for name, contract in (("saved", expected), ("current", actual)):
        if (contract.get("schema") != "action_expert_ablation_condition_v1"
                or not isinstance(contract.get("identity_payload"), dict)
                or digest(contract["identity_payload"]) != contract.get("identity")):
            _reject(f"{name} contract is missing or has inconsistent identity")
    variant = state.get("ablation_variant")
    if (state.get("schema") != "action_expert_ablation_checkpoint_v1"
            or state.get("trajectory_decoder") != "conditional_joint_trajectory_flow_matching_v2"
            or variant not in ("bev_only", "qwen_simple")):
        _reject("unsupported checkpoint schema/variant")
    for key in ("variant", "prior_source", "adapter_enabled", "final_cache_model"):
        if expected.get(key) != actual.get(key):
            _reject(f"changed {key}")
    before, after = expected["identity_payload"], actual["identity_payload"]
    if before.get("variant") != variant or after.get("variant") != variant:
        _reject("checkpoint and condition variants differ")
    is_legacy = not NEW_CONFIG_FIELDS.intersection(state["decoder_config"])
    bev_transition = variant == "bev_only" and is_legacy
    if bev_transition:
        if before.get("base") is not None or before.get("condition", {}).get("uses_qwen") is not False:
            _reject("legacy BEV must have no Qwen base or prefix")
        legacy_bev_config(state)  # Reject incomplete configs before GPU allocation.
    allowed = dict(ENTRY_SOURCE_PAIRS)
    if bev_transition:
        allowed.update(QWEN_SOURCE_PAIRS)
    normalized = deepcopy(before)
    changes = []
    try:
        old_code, new_code = before["execution"]["code"], after["execution"]["code"]
        old_packages, new_packages = before["execution"]["packages"], after["execution"]["packages"]
    except (KeyError, TypeError):
        _reject("missing execution source/package fingerprints")
    if not old_code or old_code.keys() != new_code.keys() or old_packages.keys() != new_packages.keys():
        _reject("execution source/package inventory changed")
    # Require all reviewed source anchors, even when only one differs. Prevents
    # an incomplete legacy fingerprint from masquerading as the reviewed path.
    if not set(allowed) <= old_code.keys():
        _reject("missing reviewed source anchors")
    for name, saved in old_code.items():
        current = new_code[name]
        if saved != current:
            if allowed.get(name) != (saved, current):
                _reject(f"unreviewed source change: {name}")
            normalized["execution"]["code"][name] = current
            changes.append(dict(path=["execution", "code", name], saved=saved, current=current))
    for name, saved in old_packages.items():
        current = new_packages[name]
        if saved != current:
            if not (bev_transition and name == "transformers" and (saved, current) == ("4.57.3", "5.3.0")):
                _reject(f"unreviewed package change: {name} {saved} -> {current}")
            normalized["execution"]["packages"][name] = current
            changes.append(dict(path=["execution", "packages", name], saved=saved, current=current))
    if normalized != after:
        _reject("model assets, conditioning, precision, labels or sampling changed")
    if not changes:
        _reject("identity differs outside the reviewed payload")
    return dict(profile=BEV_PROFILE if bev_transition else ENTRY_PROFILE,
                policy_sha256=file_hash(Path(__file__)),
                saved_identity=expected["identity"], current_identity=actual["identity"],
                changes=changes, restore_legacy_rope=bev_transition,
                scope="evaluation only; original weights/configuration; no optimizer resume migration",
                validation="old-source CPU numerical replay and strict loading; GPU/end-to-end results remain to be measured")


def evaluation_decoder_config(state, compatibility=None):
    """Use saved dimensions; restore missing old RoPE only after reviewed checks."""
    from qwen3vl_local.leadmot import LeadMoTPlanningDecoderConfig
    if compatibility and compatibility.get("restore_legacy_rope"):
        if compatibility.get("profile") != BEV_PROFILE:
            _reject("unknown legacy decoder compatibility profile")
        return legacy_bev_config(state)
    return LeadMoTPlanningDecoderConfig(**state["decoder_config"])
