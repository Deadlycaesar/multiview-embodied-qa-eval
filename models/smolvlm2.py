"""Hugging Face SmolVLM2 native adapter."""

from .common import NativeVLM


class SmolVLM2(NativeVLM):
    def __init__(self, config, cache_dir):
        from transformers import AutoModelForImageTextToText
        super().__init__(config, cache_dir, AutoModelForImageTextToText)
