import numpy as np
from scipy import ndimage


def count_components(binary_image):
    """
    Count connected foreground regions in a binary image.
    """
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled_image, component_count = ndimage.label(binary_image,structure=structure,)
    return int(component_count)


def count_horizontal_regions(binary_image):
    """
    Estimate how many separate drawing regions exist
    from left to right.

    This is a simple heuristic for detecting multiple
    digits drawn side by side.
    """
    column_has_foreground = np.any(binary_image, axis=0)

    regions = 0
    inside_region = False

    for has_foreground in column_has_foreground:
        if has_foreground and not inside_region:
            regions += 1
            inside_region = True

        elif not has_foreground:
            inside_region = False

    return regions


def validate_prediction(
    confidence,
    margin,
    component_count,
    horizontal_regions=1,
):
    """
    Decide whether a prediction should be accepted.
    """

    # Model confidence must be reasonably high.
    if confidence < 0.70:
        return False

    # Best prediction should clearly beat second-best.
    if margin < 0.20:
        return False

    # Too many disconnected components are suspicious.
    if component_count > 3:
        return False

    # Multiple separated horizontal regions suggest
    # that more than one digit was drawn.
    if horizontal_regions > 1:
        return False

    return True