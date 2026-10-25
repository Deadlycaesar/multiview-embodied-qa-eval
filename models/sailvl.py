"""SAIL-VL2 remote-code adapter for single-sample CUDA generation."""

from copy import deepcopy
from time import perf_counter

from ..core.schemas import GenerationResult
from .common import OPTION_PROBS_VERSION, option_probabilities


REPO_ID = "BytedanceDouyinContent/SAIL-VL2-2B"
REVISION = "3c8b3be75dd51ef77b55e5149e7dbdebd2bbe4c5"
IMG_CONTEXT_TOKEN_ID = 151671


class SAILVL:
    def __init__(self, config, cache_dir):
        import torch
        from transformers import AutoModel, AutoProcessor, set_seed

        self.torch, self.config = torch, config
        self.repo_id = config.get("repo_id", REPO_ID)
        self.revision = config.get("revision", REVISION)
        self.device = config.get("device", "cuda")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device != "cuda":
            raise ValueError("SAIL-VL2 pinned processor requires device='cuda'")
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        dtype = config["dtype"]
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unsupported dtype")
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16")
        self.dtype = getattr(torch, dtype)
        set_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        common = {
            "revision": self.revision,
            "cache_dir": str(cache_dir),
            "trust_remote_code": True,
        }
        self.processor = AutoProcessor.from_pretrained(self.repo_id, **common)
        self.model = AutoModel.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
            attn_implementation=config["attn_implementation"],
            **common,
        ).to(self.device).eval()

    def metadata(self):
        import platform

        torch = self.torch
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        hardware = {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "host": platform.node(),
            "name": properties.name,
            "total_memory": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability()),
            "cuda_version": torch.version.cuda,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
        }
        image_processor = getattr(self.processor, "image_processor", None)
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "hardware": hardware,
            "model_class": type(self.model).__name__,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "chat_template": getattr(self.processor, "chat_template", None),
            "image_processor": image_processor.to_dict() if image_processor is not None else None,
            "attn_implementation": self.config["attn_implementation"],
            "measurement_version": "single-device-v2",
            "timing_scope": "image decode + template + preprocessing + transfer + generate + output decode",
            "memory_scope": "CUDA allocated/reserved peak including resident model; unused cache cleared before each request",
            "visual_tokens_scope": "expanded IMG_CONTEXT token ID 151671 in input_ids; excludes <img> and </img>",
            "option_probs_version": OPTION_PROBS_VERSION,
        }

    def _sync(self):
        self.torch.cuda.synchronize()

    def generate(self, request):
        from PIL import Image

        torch = self.torch
        self._sync()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        images = []
        for path in request.images:
            with Image.open(path) as source:
                images.append(source.convert("RGB"))
        messages = []
        if request.system:
            messages.append({"role": "system", "content": [{"type": "text", "text": request.system}]})
        messages.append({"role": "user", "content": [
            *[{"type": "image"} for _ in images],
            {"type": "text", "text": request.text},
        ]})
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.config.get("chat_template_kwargs", {}),
        )
        inputs = self.processor(
            images=images if images else None,
            text=prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        eos_token_id = inputs.pop("eos_token_id")
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype)
            if hasattr(value, "is_floating_point") and value.is_floating_point()
            else value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        input_count = inputs["input_ids"].shape[-1]
        visual_tokens = int((inputs["input_ids"] == IMG_CONTEXT_TOKEN_ID).sum()) if images else 0
        labels = request.option_labels
        generation = deepcopy(self.model.generation_config)
        generation.update(
            **request.generation,
            max_new_tokens=request.max_new_tokens,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            eos_token_id=eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        self._sync()
        generation_start = perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=generation)
        self._sync()
        generation_seconds = perf_counter() - generation_start
        # The remote wrapper may prepend BOS; scores count exactly the generated positions.
        tokens = output.sequences[0, -len(output.scores):].tolist()
        text = self.processor.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        if labels:
            probabilities = option_probabilities(output.scores[0], labels, self.processor.tokenizer)
            probability_status = "ok"
        else:
            probabilities, probability_status = None, "not_applicable"
        self._sync()
        elapsed = perf_counter() - started
        eos = generation.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        truncated = len(tokens) >= request.max_new_tokens and (not tokens or tokens[-1] not in eos_ids)
        return GenerationResult(
            text=text,
            elapsed_seconds=elapsed,
            output_tokens=len(tokens),
            reached_token_limit=truncated,
            generation_seconds=generation_seconds,
            input_tokens=input_count,
            visual_tokens=visual_tokens,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            rendered_prompt=prompt,
            effective_generation=generation.to_dict(),
            option_probs=probabilities,
            option_probs_status=probability_status,
            option_probs_version=OPTION_PROBS_VERSION if labels else None,
        )
