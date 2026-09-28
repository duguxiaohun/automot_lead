"""Task-specific chat policy on the locally copied visual processor."""
from .vendor.processing_qwen3_vl import Qwen3VLProcessor


class Qwen35Processor(Qwen3VLProcessor):
    qwen35_non_thinking = True

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        # DeltaNet recurrent state cannot be rolled back past right padding.
        tokenizer.padding_side = "left"
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template, **kwargs)

    def apply_chat_template(self, conversation, *args, **kwargs):
        # Short supervised answers and KV condition prefixes use the same mode.
        # Explicit callers may request thinking for separate free-form experiments.
        kwargs.setdefault("enable_thinking", False)
        self._reject_remote_media(conversation)
        if kwargs["enable_thinking"] is False:
            # Keep completed assistant turns identical when extending a KV cache.
            # The official template otherwise removes their empty think block
            # once a newer user turn appears, invalidating cached-prefix equality.
            template = kwargs.get("chat_template") or self.chat_template
            kwargs["chat_template"] = template.replace(
                "{%- if loop.index0 > ns.last_query_index %}",
                "{%- if (enable_thinking is defined and enable_thinking is false) or loop.index0 > ns.last_query_index %}")
        if (conversation and isinstance(conversation[0], dict)
                and len(conversation) == 1 and conversation[0].get("role") == "system"
                and kwargs.get("tokenize", True) is False and not kwargs.get("add_generation_prompt", False)):
            # Official template requires a user. Render with a sentinel user,
            # then retain exactly the system prefix for the prefix-cache helper.
            text = super().apply_chat_template(
                conversation + [{"role": "user", "content": ""}], *args, **kwargs)
            return text.rsplit("<|im_start|>user\n", 1)[0]
        return super().apply_chat_template(conversation, *args, **kwargs)

    def __call__(self, images=None, text=None, videos=None, **kwargs):
        self._reject_remote_media({"image": images, "video": videos})
        return super().__call__(images=images, text=text, videos=videos, **kwargs)

    @staticmethod
    def _reject_remote_media(value, media=False):
        if isinstance(value, dict):
            for key, item in value.items():
                Qwen35Processor._reject_remote_media(item, media or key in {"image", "image_url", "video", "url"})
        elif media and isinstance(value, str) and "://" in value:
            raise ValueError("Qwen3.5 runtime accepts local media only")
        elif isinstance(value, (list, tuple)):
            for item in value:
                Qwen35Processor._reject_remote_media(item, media)
