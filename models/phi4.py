"""Microsoft Phi-4-multimodal manual prompt protocol."""

from .common import NativeVLM


class Phi4(NativeVLM):
    def __init__(self, config, cache_dir):
        from transformers import AutoModelForCausalLM
        super().__init__(config, cache_dir, AutoModelForCausalLM)

    def render_prompt(self, request):
        if request.system:
            raise ValueError("Phi-4 multimodal protocol does not use a system prompt")
        images = "".join(f"<|image_{index}|>" for index in range(1, len(request.images) + 1))
        return f"<|user|>{images}{request.text}<|end|><|assistant|>"
