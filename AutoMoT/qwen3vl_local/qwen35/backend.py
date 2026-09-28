"""Explicit local Qwen3.5 loading. No Hub resolution or dynamic remote code.

The model, config, tokenizer and visual processors are vendored next to this
file. Transformers supplies shared infrastructure, pinned in requirements.txt.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DEFAULT_MODEL_NAME = "Qwen3.5-4B"
BACKEND_VERSION = "qwen35_local_v2"
TRANSFORMERS_VERSION = "5.3.0"


def assistant_header_text(processor):
    header = "<|im_start|>assistant\n"
    if getattr(processor, "qwen35_non_thinking", False):
        header += "<think>\n\n</think>\n\n"
    return header


def offline():
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY"):
        os.environ[name] = "1"


def local_model_dir(path):
    offline()
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Missing local Qwen3.5 model directory: {root}; runtime never downloads files")
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"Expected Qwen3.5 (qwen3_5), got {config.get('model_type')!r}; old runs require their original source/base")
    return root


def require_runtime():
    offline()
    import transformers
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Local Qwen3.5 requires transformers=={TRANSFORMERS_VERSION}, found {transformers.__version__}; see qwen35/requirements.txt")


def backend_contract():
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*')) if p.suffix in {'.py', '.json', '.txt', '.jinja'}
             and '__pycache__' not in p.parts and 'tests' not in p.parts}
    return dict(version=BACKEND_VERSION, model_type="qwen3_5", transformers=TRANSFORMERS_VERSION,
                enable_thinking=False, source_sha256=files)


class LocalModel:
    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        root = local_model_dir(path)
        require_runtime()
        from .vendor.configuration_qwen3_5 import Qwen3_5Config
        from .vendor.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("local_files_only", None)
        if kwargs.pop("use_kernels", False):
            raise ValueError("Hub kernel downloads are disabled; use locally installed CUDA kernels")
        kwargs.setdefault("config", Qwen3_5Config.from_pretrained(str(root), local_files_only=True))
        from .identity import base_asset_hashes
        if (kwargs.get("use_safetensors") is False or kwargs.get("variant") is not None
                or kwargs.get("state_dict") is not None):
            raise ValueError("Local base identity requires the default safetensors weights")
        kwargs["use_safetensors"] = True
        assets = base_asset_hashes(root)
        # Avoid changing globally installed Transformers auto registries.
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(root), *args, local_files_only=True, **kwargs)
        model.config.local_backend_contract = backend_contract()
        model.config.local_asset_sha256 = assets
        return model


class LocalProcessor:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        root = local_model_dir(path)
        require_runtime()
        from .processor import Qwen35Processor
        from .vendor.tokenization_qwen3_5 import Qwen3_5Tokenizer
        from .vendor.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
        from .vendor.video_processing_qwen3_vl import Qwen3VLVideoProcessor
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("local_files_only", None)
        kwargs.pop("use_fast", None)
        tokenizer = Qwen3_5Tokenizer.from_pretrained(str(root), local_files_only=True, **kwargs)
        image = Qwen2VLImageProcessorFast.from_pretrained(str(root), local_files_only=True)
        video = Qwen3VLVideoProcessor.from_pretrained(str(root), local_files_only=True)
        template_file = root / "chat_template.jinja"
        template = template_file.read_text() if template_file.is_file() else tokenizer.chat_template
        if not template:
            raise ValueError(f"Missing local official chat template: {root}")
        if 'enable_thinking' not in template or '{%- if loop.index0 > ns.last_query_index %}' not in template:
            raise ValueError("Unrecognized Qwen3.5 chat template; review local processor compatibility before use")
        return Qwen35Processor(image_processor=image, tokenizer=tokenizer,
                               video_processor=video, chat_template=template)


# Compatibility import names used by existing experiment entry points.
AutoProcessor = LocalProcessor
AutoModelForImageTextToText = LocalModel
