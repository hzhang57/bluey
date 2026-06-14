PROMPT_TEMPLATE = """Completely cover {object_text} with solid bright white paint in every frame.
The visible parts of {object_text} must be pure white, with no original color or texture remaining.
Keep the white paint attached to the same object through motion and occlusion.
Keep the scene, camera motion, and all non-target objects unchanged.
Do not produce a mask-only image, black background, or blank frame."""


def build_silhouette_prompt(object_text: str) -> str:
    object_text = object_text.strip()
    if not object_text:
        raise ValueError("--object must contain a non-empty text description")
    return PROMPT_TEMPLATE.format(object_text=object_text)
