"""Ai2 Molmo2 adapter using the pinned remote processor."""

from .common import NativeVLM


class Molmo2(NativeVLM):
    def __init__(self, config, cache_dir):
        from transformers import AutoModelForImageTextToText
        super().__init__(config, cache_dir, AutoModelForImageTextToText)

    def visual_token_count(self, inputs, image_count):
        if image_count == 0:
            return 0
        token_id = self.processor.tokenizer.convert_tokens_to_ids("<im_patch>")
        return int((inputs["input_ids"] == token_id).sum())
