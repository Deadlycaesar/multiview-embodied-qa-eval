"""MiniCPM-V 4 remote-code adapter using its processor and decode_text=False path."""

from copy import deepcopy
from time import perf_counter

from ..core.schemas import GenerationResult
from .common import OPTION_PROBS_VERSION, option_probabilities


REPO_ID = "openbmb/MiniCPM-V-4"
REVISION = "6f69a8235885be89608f3a71ffd40695b379e1e7"


class MiniCPM:
    def __init__(self, config, cache_dir):
        import torch
        from transformers import AutoModel, AutoProcessor, set_seed

        self.torch, self.config = torch, config
        self.repo_id = config.get("repo_id", REPO_ID)
        self.revision = config.get("revision", REVISION)
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
            raise ValueError("This CUDA device does not support bfloat16")
        self.dtype = getattr(torch, dtype)
        if config["attn_implementation"] != "sdpa":
            raise ValueError("MiniCPM-V 4 requires attn_implementation='sdpa'; eager is unsupported")
        set_seed(config["seed"])
        if self.device == "cuda":
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
        hardware = {"machine": platform.machine(), "processor": platform.processor(), "host": platform.node()}
        if self.device == "cuda":
            properties = torch.cuda.get_device_properties(torch.cuda.current_device())
            hardware.update(
                name=properties.name,
                total_memory=properties.total_memory,
                capability=list(torch.cuda.get_device_capability()),
                cuda_version=torch.version.cuda,
                uuid=str(getattr(properties, "uuid", "unavailable")),
            )
        else:
            hardware["name"] = platform.platform()
        image_processor = getattr(self.processor, "image_processor", None)
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "hardware": hardware,
            "model_class": type(self.model).__name__,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "chat_template": getattr(self.processor.tokenizer, "chat_template", None),
            "image_processor": image_processor.to_dict() if image_processor is not None else None,
            "attn_implementation": self.config["attn_implementation"],
            "measurement_version": "single-device-v2",
            "timing_scope": "image decode + template + preprocessing + transfer + generate + output decode",
            "memory_scope": "CUDA allocated/reserved peak including resident model; unused cache cleared before each request",
            "visual_tokens_scope": "sum of half-open image_bound intervals; excludes image and slice delimiters",
            "option_probs_version": OPTION_PROBS_VERSION,
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
        content = ["(<image>./</image>)" for _ in images]
        content.append(request.text)
        messages = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.append({"role": "user", "content": "\n".join(content)})
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.config.get("chat_template_kwargs", {}),
        )
        # [[]] is intentional: the pinned processor preserves empty pixel_values,
        # image_bound and tgt_sizes structures required by MiniCPMV.generate.
        inputs = self.processor(
            [prompt],
            [images],
            max_slice_nums=self.config.get("max_slice_nums"),
            use_image_id=self.config.get("use_image_id"),
            return_tensors="pt",
            max_length=self.config.get("max_inp_length", 32768),
        ).to(self.device)
        inputs.pop("image_sizes")
        input_count = inputs["input_ids"].shape[-1]
        visual_tokens = sum(int(end) - int(start) for start, end in inputs["image_bound"][0])
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
            pad_token_id=0,
            eos_token_id=[
                self.processor.tokenizer.convert_tokens_to_ids(token) for token in self.model.terminators
            ],
            return_dict_in_generate=True,
            output_scores=True,
        )
        self._sync()
        generation_start = perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                tokenizer=self.processor.tokenizer,
                decode_text=False,
                generation_config=generation,
            )
        self._sync()
        generation_seconds = perf_counter() - generation_start
        # The pinned LLM may prepend BOS; scores count exactly the generated positions.
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
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if self.device == "cuda" else None,
            rendered_prompt=prompt,
            effective_generation=generation.to_dict(),
            option_probs=probabilities,
            option_probs_status=probability_status,
            option_probs_version=OPTION_PROBS_VERSION if labels else None,
        )
