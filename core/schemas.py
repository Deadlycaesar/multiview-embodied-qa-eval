"""Reference answers never enter InferenceRequest."""

from dataclasses import dataclass, field


@dataclass
class Sample:
    id: str
    question: str
    images: list[str]
    choices: dict[str, str]
    answer: str
    references: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class InferenceRequest:
    sample_id: str
    system: str
    text: str
    images: list[str]
    max_new_tokens: int
    generation: dict
    metadata: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    elapsed_seconds: float
    output_tokens: int
    reached_token_limit: bool
    generation_seconds: float
    input_tokens: int
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    rendered_prompt: str
    effective_generation: dict
