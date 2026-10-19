"""Shared single-device inference and measurement for both native VLM adapters."""

from copy import deepcopy
from time import perf_counter

from ..core.schemas import GenerationResult


def messages_for(request):
    messages = []
    if request.system:
        messages.append({"role": "system", "content": [{"type": "text", "text": request.system}]})
    messages.append({"role": "user", "content": [
        *[{"type": "image"} for _ in request.images], {"type": "text", "text": request.text},
    ]})
    return messages


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
        common = {"revision": config["revision"], "cache_dir": str(cache_dir), "trust_remote_code": False}
        self.processor = AutoProcessor.from_pretrained(config["repo_id"], use_fast=False, **config.get("processor", {}), **common)
        self.model = model_class.from_pretrained(
            config["repo_id"], dtype=self.dtype,
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
        return {
            "device": self.device, "dtype": str(self.dtype), "hardware": hardware,
            "model_class": type(self.model).__name__, "repo_id": self.config["repo_id"],
            "revision": self.config["revision"], "chat_template": self.processor.chat_template,
            "image_processor": self.processor.image_processor.to_dict(),
            "attn_implementation": self.config["attn_implementation"],
            "measurement_version": "single-device-v1",
            "timing_scope": "image decode + template + preprocessing + transfer + generate + output decode",
            "memory_scope": "CUDA allocated/reserved peak including resident model; unused cache cleared before each request",
        }

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

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
        if not images:
            raise ValueError("At least one image is required")
        prompt = self.processor.apply_chat_template(
            messages_for(request), tokenize=False, add_generation_prompt=True,
            **self.config.get("chat_template_kwargs", {}))
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt", padding=True)
        # Image grids and token IDs stay integer; only floating tensors adopt model dtype.
        inputs = {key: value.to(device=self.device, dtype=self.dtype) if value.is_floating_point()
                  else value.to(self.device) for key, value in inputs.items()}
        generation = deepcopy(self.model.generation_config)
        generation.update(
            **request.generation, max_new_tokens=request.max_new_tokens,
            temperature=None, top_p=None, top_k=None, repetition_penalty=1.0,
            no_repeat_ngram_size=0, return_dict_in_generate=False, output_scores=False,
        )
        self._sync()
        generation_start = perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=generation)
        self._sync()
        generation_seconds = perf_counter() - generation_start
        input_count = inputs["input_ids"].shape[-1]
        tokens = output[0, input_count:].tolist()
        text = self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self._sync()
        elapsed = perf_counter() - started
        eos = generation.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        truncated = len(tokens) >= request.max_new_tokens and (not tokens or tokens[-1] not in eos_ids)
        return GenerationResult(
            text=text, elapsed_seconds=elapsed, output_tokens=len(tokens), reached_token_limit=truncated,
            generation_seconds=generation_seconds, input_tokens=input_count,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device == "cuda" else None,
            rendered_prompt=prompt, effective_generation=generation.to_dict(),
        )
