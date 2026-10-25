"""Liquid AI LFM2.5-VL native adapter."""

from .common import NativeVLM


class LFM25(NativeVLM):
    def __init__(self, config, cache_dir):
        from transformers import AutoModelForImageTextToText
        super().__init__(config, cache_dir, AutoModelForImageTextToText)
