"""Resolve immutable camera plans without exposing answer annotations to requests."""


def select_views(sample, strategy, plan_row=None):
    if strategy == "all":
        if plan_row is not None:
            raise ValueError("All-view conditions do not consume a view-plan row")
        indices = list(range(len(sample.images)))
        plan = {"experiment": "OPEN", "condition": "open_all_views", "k": len(indices), "seed": None}
    elif strategy == "plan":
        if plan_row is None or plan_row["qid"] != sample.id:
            raise ValueError(f"{sample.id}: missing or mismatched view-plan row")
        indices = plan_row["cameras"]
        if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                or indices != sorted(set(indices)) or any(i < 0 or i >= len(sample.images) for i in indices)):
            raise ValueError(f"{sample.id}: invalid camera indices")
        if plan_row["k"] != len(indices):
            raise ValueError(f"{sample.id}: view-plan k does not match cameras")
        plan = {key: plan_row[key] for key in ("experiment", "condition", "k", "seed")}
    else:
        raise ValueError(f"Unknown view strategy: {strategy}")
    images = [sample.images[i] for i in indices]
    if len(set(images)) != len(images):
        raise ValueError(f"{sample.id}: repeated view paths")
    required = sample.metadata.get("necessary_view_indices")
    return images, {
        **plan,
        "original_image_count": len(sample.images),
        "input_image_count": len(images),
        "selected_view_indices": indices,
        "cameras": indices,
        "removed_view_indices": [i for i in range(len(sample.images)) if i not in indices],
        "necessary_view_missing": None if required is None else bool(set(required) - set(indices)),
        "image_strategy": strategy,
    }
