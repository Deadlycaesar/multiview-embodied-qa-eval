"""Shared single-device inference and measurement for native VLM adapters."""

import math
from copy import deepcopy
from time import perf_counter

from ..core.schemas import GenerationResult


OPTION_PROBS_VERSION = "first-token-label-softmax-v1"


def messages_for(request):
    messages = []
    if request.system:
        messages.append({"role": "system", "content": [{"type": "text", "text": request.system}]})
    messages.append({"role": "user", "content": [
        *[{"type": "image"} for _ in request.images], {"type": "text", "text": request.text},
    ]})
    return messages


def option_probabilities(first_logits, labels, tokenizer):
    """Normalize first-generation-position logits over single-token option labels."""
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("MCQ option labels must be nonempty and unique")
    token_ids = []
    for label in labels:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Option label {label!r} is not a single token for this model")
        token_ids.append(encoded[0])
    logits = first_logits[0, token_ids].float()
    probabilities = logits.softmax(dim=-1).tolist()
    if any(not math.isfinite(value) for value in probabilities):
        raise ValueError("Non-finite option probability")
    return dict(zip(labels, probabilities))


class NativeVLM:
    def __init__(self, config, cache_dir, model_class):
        import torch
        from transformers import AutoProcessor, set_seed

        self.torch, self.config = torch, config
        self.device = config.get("device", "cuda")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda or mps")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS is unavailable")
        dtype = config["dtype"]
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")
        if self.device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16; set both model configs explicitly")
        self.dtype = getattr(torch, dtype)
        set_seed(config["seed"])
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        common = {"revision": config["revision"], "cache_dir": str(cache_dir),
                  "trust_remote_code": config.get("trust_remote_code", False)}
        self.processor = AutoProcessor.from_pretrained(
            config["repo_id"], use_fast=config.get("use_fast", False),
            **config.get("processor", {}), **common,
        )
        self.model = model_class.from_pretrained(
            config["repo_id"], torch_dtype=self.dtype,
            attn_implementation=config["attn_implementation"], **common,
        ).to(self.device).eval()

    def metadata(self):
        import platform
        torch = self.torch
        hardware = {"machine": platform.machine(), "processor": platform.processor(), "host": platform.node()}
        if self.device == "cuda":
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            hardware.update(name=properties.name, total_memory=properties.total_memory,
                            capability=list(torch.cuda.get_device_capability()), cuda_version=torch.version.cuda,
                            uuid=str(getattr(properties, "uuid", "unavailable")))
        else:
            hardware["name"] = platform.platform()
        image_processor = getattr(self.processor, "image_processor", None)
        return {
            "device": self.device, "dtype": str(self.dtype), "hardware": hardware,
            "model_class": type(self.model).__name__, "repo_id": self.config["repo_id"],
            "revision": self.config["revision"], "chat_template": getattr(self.processor, "chat_template", None),
            "image_processor": image_processor.to_dict() if image_processor is not None else None,
            "attn_implementation": self.config["attn_implementation"],
            "measurement_version": "single-device-v2",
            "timing_scope": "image decode + template + preprocessing + transfer + generate + output decode",
            "memory_scope": "CUDA allocated/reserved peak including resident model; unused cache cleared before each request",
            "visual_tokens_scope": "expanded visual placeholder token IDs in input_ids; excludes text and structural tokens",
            "option_probs_version": OPTION_PROBS_VERSION,
        }

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def visual_token_count(self, inputs, image_count):
        if image_count == 0:
            return 0
        token_ids = self.config.get("visual_token_ids")
        if token_ids is None:
            token_id = getattr(self.model.config, "image_token_id", None)
            if token_id is None:
                raise ValueError("Adapter must define visual_token_ids for this model")
            token_ids = [token_id]
        input_ids = inputs["input_ids"]
        return sum(int((input_ids == token_id).sum()) for token_id in token_ids)

    def render_prompt(self, request):
        return self.processor.apply_chat_template(
            messages_for(request), tokenize=False, add_generation_prompt=True,
            **self.config.get("chat_template_kwargs", {}))

    def generate(self, request):
        from PIL import Image
        torch = self.torch
        self._sync()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        images = []
        for path in request.images:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        prompt = self.render_prompt(request)
        processor_kwargs = {"text": [prompt], "return_tensors": "pt", "padding": True}
        if images:
            processor_kwargs["images"] = images
        inputs = self.processor(**processor_kwargs)
        # Image grids and token IDs stay integer; only floating tensors adopt model dtype.
        inputs = {key: value.to(device=self.device, dtype=self.dtype) if value.is_floating_point()
                  else value.to(self.device) for key, value in inputs.items()}
        visual_tokens = self.visual_token_count(inputs, len(images))
        labels = request.option_labels
        generation = deepcopy(self.model.generation_config)
        generation.update(
            **request.generation, max_new_tokens=request.max_new_tokens,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
            no_repeat_ngram_size=0, return_dict_in_generate=bool(labels), output_scores=bool(labels),
        )
        self._sync()
        generation_start = perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=generation)
        self._sync()
        generation_seconds = perf_counter() - generation_start
        sequences = output.sequences if labels else output
        input_count = inputs["input_ids"].shape[-1]
        tokens = sequences[0, input_count:].tolist()
        text = self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if labels:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            probabilities = option_probabilities(output.scores[0], labels, tokenizer)
            probability_status = "ok"
        else:
            probabilities, probability_status = None, "not_applicable"
        self._sync()
        elapsed = perf_counter() - started
        eos = generation.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        truncated = len(tokens) >= request.max_new_tokens and (not tokens or tokens[-1] not in eos_ids)
        return GenerationResult(
            text=text, elapsed_seconds=elapsed, output_tokens=len(tokens), reached_token_limit=truncated,
            generation_seconds=generation_seconds, input_tokens=input_count, visual_tokens=visual_tokens,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device == "cuda" else None,
            rendered_prompt=prompt, effective_generation=generation.to_dict(),
            option_probs=probabilities, option_probs_status=probability_status,
            option_probs_version=OPTION_PROBS_VERSION if labels else None,
        )
