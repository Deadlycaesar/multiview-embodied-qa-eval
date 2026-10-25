"""Google Gemma 4 native multimodal adapter."""

from .common import NativeVLM


class Gemma4(NativeVLM):
    def __init__(self, config, cache_dir):
        from transformers import AutoModelForMultimodalLM
        super().__init__(config, cache_dir, AutoModelForMultimodalLM)
