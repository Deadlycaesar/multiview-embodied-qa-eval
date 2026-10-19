"""Qwen3.5 unified vision-language model.

Requires transformers from git main (the qwen3_5 architecture is not in the
pinned 4.57.x release). Thinking mode is disabled through the model config's
chat_template_kwargs so answers fit the benchmark token budget.
"""

from .common import NativeVLM


class Qwen35(NativeVLM):
    def __init__(self, config, cache_dir):
        from transformers import Qwen3_5ForConditionalGeneration
        super().__init__(config, cache_dir, Qwen3_5ForConditionalGeneration)
