"""Build answer-free requests independently of model chat templates."""

from ..core.schemas import InferenceRequest
from .views import select_views


def build_request(sample, experiment, config, assets_dir):
    images, view_info = select_views(sample, experiment["images"], config["white_image"], assets_dir)
    text = sample.question
    if experiment["mode"] == "mcq":
        if not sample.choices:
            raise ValueError("MCQ requires labeled options")
        text += "\n" + "\n".join(f"{key}. {value}" for key, value in sample.choices.items())
    instruction = config["prompts"][experiment["prompt"]]
    system = ""
    if instruction:
        if config["prompts"]["placement"] == "system":
            system = instruction
        else:
            text += "\n\n" + instruction
    return InferenceRequest(
        sample.id, system, text, images, experiment["max_new_tokens"], dict(config["generation"]),
        {**view_info, "mode": experiment["mode"], "prompt_variant": experiment["prompt"],
         "prompt_version": config["prompts"]["version"], "instruction": instruction,
         "prompt_placement": config["prompts"]["placement"]},
    )
