"""离线权重选择与可复现合同；此模块不加载 torch 或模型。"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from qwen3vl_local.sft_new_loop_phase1 import prompts as p1
from qwen3vl_local.sft_new_loop_phase2 import prompts as p2
from qwen3vl_local.sft_new_loop_phase1.history_rgb import history_rgb_indices

SCHEMA = "action_prior_base_kv_v1"


def digest(value):
    """计算规范 JSON 的稳定指纹。"""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_hash(path):
    """流式计算文件指纹。"""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    """读取本地 JSON。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_adapter(path, phase, model_dir):
    """硬校验模型、提示词和 RGB 合同；显式选择也不能绕过。"""
    path = Path(path).expanduser().resolve()
    cfg_name = f"sft_new_loop_phase{phase}_adapter_config.json"
    cfg = read_json(path / cfg_name)
    peft = read_json(path / "adapter_config.json")
    module = p1 if phase == 1 else p2
    mode = cfg["history_rgb_mode"]
    indices = list(history_rgb_indices(mode))
    expected = (p1.phase1_prompt_sha256 if phase == 1 else p2.event_prompt_sha256)(
        history_rgb_mode=mode
    )
    if (
        cfg.get("prompt_name") != module.PROMPT_NAME
        or cfg.get("production_prompt_sha256") != expected
    ):
        raise ValueError(
            f"{path}: prompt version/hash differs from current Phase{phase}"
        )
    if not cfg.get("git", {}).get("commit"):
        raise ValueError(f"{path}: missing training git.commit")
    if (
        Path(cfg["base_model_dir"]).expanduser().resolve()
        != Path(model_dir).expanduser().resolve()
    ):
        raise ValueError(
            f"{path}: base_model_dir mismatch; restore training base path/symlink"
        )
    if cfg.get("history_rgb_selected_indices") != indices or cfg.get(
        "history_rgb_count"
    ) != len(indices):
        raise ValueError(f"{path}: inconsistent RGB metadata")
    # 单 base 多 adapter 要保持禁用后严格等于原始模型，拒绝额外可训练 bias/保存模块。
    if (
        peft.get("bias", "none") != "none"
        or peft.get("modules_to_save")
        or peft.get("peft_type") != "LORA"
    ):
        raise ValueError(f"{path}: requires plain LoRA, bias=none, no modules_to_save")
    weights = [
        path / n
        for n in ("adapter_model.safetensors", "adapter_model.bin")
        if (path / n).is_file()
    ]
    if len(weights) != 1:
        raise ValueError(f"{path}: expected exactly one adapter weight file")
    files = {
        p.name: file_hash(p)
        for p in [path / cfg_name, path / "adapter_config.json", *weights]
    }
    return dict(
        path=str(path),
        phase=phase,
        metadata=cfg,
        file_sha256=files,
        fingerprint=digest(files),
    )


def selection_score(path, cfg, phase):
    """只读取被保存 best 的 validation generation 分数，不借用测试集评分。"""
    if phase == 2:
        record = read_json(path.parent / "best_generation.json")
        if record.get("generation_guards_ok") is not True:
            raise ValueError("best_generation validation guards missing/failed")
        if int(record["step"]) != int(cfg["global_step"]):
            raise ValueError("selection record step differs from adapter")
        score = record["generation_exact_accuracy"]
    else:
        # Phase1 保存 adapter 时没有 best_generation.json，按该 adapter 的 step 精确回查。
        matches = []
        with (path.parent / "train_eval_metrics.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("type") == "generation" and int(r["step"]) == int(
                    cfg["global_step"]
                ):
                    matches.append(r)
        if len(matches) != 1:
            raise ValueError("missing/ambiguous generation record for saved step")
        r = matches[0]
        if r["format_valid_rate"] < cfg["generation_format_valid_gate"]:
            raise ValueError("saved generation record fails format guard")
        score = r["exact_accuracy"]
    if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
        raise ValueError("invalid generation exact score")
    return float(score)


def select_adapter(root, phase, model_dir, explicit=""):
    """显式路径优先；自动只在 best_generation 内比较兼容候选。"""
    if explicit:
        item = inspect_adapter(explicit, phase, model_dir)
        return {
            **item,
            "selection": "explicit",
            "is_best_generation": Path(item["path"]).name == "best_generation",
        }
    root = Path(root).expanduser().resolve()
    paths = {
        p.parent.resolve()
        for p in root.rglob(f"sft_new_loop_phase{phase}_adapter_config.json")
        if p.parent.name == "best_generation"
    }
    eligible, rejected = [], []
    for path in sorted(paths):
        try:
            item = inspect_adapter(path, phase, model_dir)
            item["generation_exact"] = selection_score(path, item["metadata"], phase)
            eligible.append(item)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            rejected.append({"path": str(path), "reason": str(exc)})
    if not eligible:
        raise ValueError(
            f"Phase{phase}: no compatible best_generation weights in {root}; no final fallback. Rejections={rejected}"
        )
    eligible.sort(
        key=lambda x: (
            x["generation_exact"],
            x["metadata"].get("saved_at", ""),
            x["path"],
        ),
        reverse=True,
    )
    return {
        **eligible[0],
        "selection": "best_generation_validation_exact",
        "is_best_generation": True,
        "candidates": [
            {k: x[k] for k in ("path", "generation_exact")} for x in eligible
        ],
        "rejected": rejected,
        "comparison_note": "validation scores across runs may use different sampled cases; not a common holdout ranking",
    }


def require_contract(expected, actual):
    """resume/eval 必须恢复同样的 base、先验权重与分析协议。"""
    if expected.get("schema") != SCHEMA or expected.get("identity") != actual.get(
        "identity"
    ):
        raise ValueError(
            "action prior checkpoint contract mismatch; do not cross-load old LeadMoT or different priors"
        )
