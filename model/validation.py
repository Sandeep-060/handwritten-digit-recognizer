import numpy as np
from scipy import ndimage


def count_components(binary_image):
    """
    Count connected foreground regions in a binary image.
    Returns:
        Number of connected components.
    """

    structure = np.ones((3, 3), dtype=np.uint8)

    labeled_image, component_count = ndimage.label(
        binary_image,
        structure=structure,
    )

    return int(component_count)

def validate_prediction(
    confidence,
    margin,
    component_count,
):
    """
    Decide whether a prediction should be accepted.

    This combines model confidence, probability margin,
    and simple structural information from the drawing.
    """

    # Model must be reasonably confident.
    if confidence < 0.70:
        return False

    # The best class should be clearly ahead of the second-best class.
    if margin < 0.20:
        return False

    # A large number of disconnected regions is suspicious.
    # We use this only as a heuristic, not as absolute proof
    # that multiple digits were drawn.
    if component_count > 3:
        return False

    return True