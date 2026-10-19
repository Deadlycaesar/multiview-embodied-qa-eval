"""Preserve source order; required-view judgments need annotations."""

from ..core.paths import project_path


def select_views(sample, strategy, white, assets_dir):
    if strategy == "white":
        from PIL import Image
        if white["mode"] == "file":
            path = project_path(white["path"])
        else:
            size = white["size"]
            path = assets_dir / f"white-{size}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                Image.new("RGB", (size, size), "white").save(path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if any(extrema != (255, 255) for extrema in rgb.getextrema()):
                raise ValueError("White baseline must contain only RGB white pixels")
            if white["mode"] == "generated" and rgb.size != (white["size"], white["size"]):
                raise ValueError("Existing white image has the wrong size")
        indices, images = [], [str(path.resolve())]
    else:
        count = len(sample.images) if strategy == "all" else {"first1": 1, "first2": 2}[strategy]
        if len(sample.images) < count:
            raise ValueError(f"{sample.id}: fewer images than requested")
        indices = list(range(count))
        images = [sample.images[i] for i in indices]
        if len(set(images)) != len(images):
            raise ValueError(f"{sample.id}: repeated view paths")
    required = sample.metadata.get("necessary_view_indices")
    return images, {
        "original_image_count": len(sample.images), "input_image_count": len(images),
        "selected_view_indices": indices,
        "removed_view_indices": [i for i in range(len(sample.images)) if i not in indices],
        "necessary_view_missing": None if required is None else bool(set(required) - set(indices)),
        "dual_equals_multi": len(sample.images) == 2, "image_strategy": strategy,
    }
