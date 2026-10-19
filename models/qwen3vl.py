"""Qwen3-VL native image placeholders and ordered PIL images; no video helper required."""

from .common import NativeVLM


class Qwen3VL(NativeVLM):
    def __init__(self, config, cache_dir):
        from transformers import Qwen3VLForConditionalGeneration
        super().__init__(config, cache_dir, Qwen3VLForConditionalGeneration)
