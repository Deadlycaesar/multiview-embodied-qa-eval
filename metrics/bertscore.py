"""Pinned local snapshot for BERTScore; loaded only by score --bertscore."""

from ..core.config import pinned


def validate_config(config):
    pinned(config["revision"])
    if config.get("idf") is not False or config.get("rescale_with_baseline") is not False:
        raise ValueError("This protocol fixes IDF and baseline rescaling to false")
    if type(config.get("num_layers")) is not int or config["num_layers"] < 1:
        raise ValueError("BERTScore needs an explicit positive layer count")
    if type(config.get("batch_size")) is not int or config["batch_size"] < 1:
        raise ValueError("BERTScore batch_size must be positive")


class BertScore:
    def __init__(self, config, cache_dir):
        validate_config(config)
        from bert_score import BERTScorer
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            config["repo_id"], revision=config["revision"], cache_dir=str(cache_dir),
            allow_patterns=["*.json", "merges.txt", "*.safetensors"],
        )
        self.scorer = BERTScorer(
            model_type=path, num_layers=config["num_layers"], lang=config["lang"],
            idf=False, rescale_with_baseline=False, device=config["device"],
            batch_size=config["batch_size"], use_fast_tokenizer=False,
        )
        self.metadata = {**config, "resolved_path": path, "implementation_hash": self.scorer.hash}

    def score(self, prediction, references):
        if not prediction.strip():
            return 0.0
        # Score each reference and choose the greatest F1 explicitly.
        _, _, f1 = self.scorer.score([prediction] * len(references), references)
        return float(f1.max().item())
