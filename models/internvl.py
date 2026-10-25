"""InternVL 3.5 remote-code adapter with dynamic tiles and raw generation scores."""

from copy import deepcopy
from time import perf_counter

from ..core.schemas import GenerationResult
from .common import OPTION_PROBS_VERSION, option_probabilities


REPO_ID = "OpenGVLab/InternVL3_5-4B"
REVISION = "481f6e32467eab4e922ccd7fd6cf420441a62331"
IMG_CONTEXT = "<IMG_CONTEXT>"


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num, max_num, image_size, use_thumbnail):
    """Pinned InternVL dynamic_preprocess, including its deterministic tie-breaking."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


class InternVL:
    def __init__(self, config, cache_dir):
        import torch
        from transformers import AutoModel, AutoTokenizer, set_seed

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
        set_seed(config["seed"])
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        common = {
            "revision": self.revision,
            "cache_dir": str(cache_dir),
            "trust_remote_code": True,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.repo_id, use_fast=config.get("use_fast", False), **common,
        )
        self.model = AutoModel.from_pretrained(
            self.repo_id,
            torch_dtype=self.dtype,
            attn_implementation=config["attn_implementation"],
            use_flash_attn=False,
            **common,
        ).to(self.device).eval()
        self.image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        self.min_num = self.model.config.min_dynamic_patch
        self.max_num = self.model.config.max_dynamic_patch
        self.use_thumbnail = self.model.config.use_thumbnail

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
        return {
            "device": self.device,
            "dtype": str(self.dtype),
            "hardware": hardware,
            "model_class": type(self.model).__name__,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "chat_template": getattr(self.tokenizer, "chat_template", None),
            "image_processor": {
                "type": "InternVL-dynamic-tile-v1",
                "image_size": self.image_size,
                "min_num": self.min_num,
                "max_num": self.max_num,
                "use_thumbnail": self.use_thumbnail,
            },
            "attn_implementation": getattr(
                self.model.config.llm_config, "_attn_implementation", self.config["attn_implementation"],
            ),
            "measurement_version": "single-device-v2",
            "timing_scope": "image decode + template + preprocessing + transfer + generate + output decode",
            "memory_scope": "CUDA allocated/reserved peak including resident model; unused cache cleared before each request",
            "visual_tokens_scope": "expanded IMG_CONTEXT token IDs in input_ids; excludes <img> and </img>",
            "option_probs_version": OPTION_PROBS_VERSION,
        }

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def _load_tiles(self, paths):
        import torchvision.transforms as T
        from PIL import Image
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        images = []
        num_patches = []
        for path in paths:
            with Image.open(path) as source:
                image = source.convert("RGB")
            tiles = _dynamic_preprocess(
                image, self.min_num, self.max_num, self.image_size, self.use_thumbnail,
            )
            images.extend(transform(tile) for tile in tiles)
            num_patches.append(len(tiles))
        if not images:
            return None, []
        return self.torch.stack(images).to(device=self.device, dtype=self.dtype), num_patches

    def generate(self, request):
        torch = self.torch
        self._sync()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        pixel_values, num_patches = self._load_tiles(request.images)
        question = "".join("<image>\n" for _ in request.images) + request.text
        system = request.system or self.model.system_message
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        expanded_prompt = prompt
        for count in num_patches:
            image_tokens = "<img>" + IMG_CONTEXT * self.model.num_image_token * count + "</img>"
            expanded_prompt = expanded_prompt.replace("<image>", image_tokens, 1)
        model_inputs = self.tokenizer(expanded_prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in model_inputs.items()}
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT)
        visual_tokens = int((inputs["input_ids"] == self.model.img_context_token_id).sum()) if num_patches else 0
        input_count = inputs["input_ids"].shape[-1]
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
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            return_dict_in_generate=True,
            output_scores=True,
        )
        self._sync()
        generation_start = perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation,
            )
        self._sync()
        generation_seconds = perf_counter() - generation_start
        # inputs_embeds generation may include BOS; scores count exactly the generated positions.
        tokens = output.sequences[0, -len(output.scores):].tolist()
        text = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if labels:
            probabilities = option_probabilities(output.scores[0], labels, self.tokenizer)
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
