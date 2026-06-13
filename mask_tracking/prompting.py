PROMPT_TEMPLATE = """Paint only {object_text} solid pure white.
Keep every background pixel and every other object unchanged.
Do not replace the scene with a silhouette, mask image, black background, or blank frame.
Preserve the original motion, camera movement, timing, and occlusions.
Only paint the visible parts of {object_text}, and keep the white paint attached to the same object."""


def build_silhouette_prompt(object_text: str) -> str:
    object_text = object_text.strip()
    if not object_text:
        raise ValueError("--object must contain a non-empty text description")
    return PROMPT_TEMPLATE.format(object_text=object_text)
