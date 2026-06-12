PROMPT_TEMPLATE = """Turn {object_text} into a solid, pure white silhouette.
Keep everything else unchanged.
Preserve the original motion, camera movement, timing, and occlusions.
Only paint the visible parts of {object_text}.
The white silhouette must remain attached to the same object in every frame."""


def build_silhouette_prompt(object_text: str) -> str:
    object_text = object_text.strip()
    if not object_text:
        raise ValueError("--object must contain a non-empty text description")
    return PROMPT_TEMPLATE.format(object_text=object_text)
