import numpy as np
from PIL import Image


def preprocess_image(image):
   
    # Convert image to a NumPy array
    image = np.asarray(image)

    # Remove alpha channel if the image has RGBA channels
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    # Convert RGB image to grayscale
    if image.ndim == 3:
        image = np.mean(image, axis=-1)

    image = image.astype("float32")

    # Normalize pixel values temporarily to 0–1
    if image.max() > 1.0:
        image = image / 255.0

    # Make sure the digit is white on a black background
    if image.mean() > 0.5:
        image = 1.0 - image

    # Find pixels that belong to the drawing
    threshold = 0.1
    mask = image > threshold

    # Empty canvas
    if not np.any(mask):
        return None

    # Find bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])
    digit = image[top:bottom, left:right]

    # Resize while preserving aspect ratio
    height, width = digit.shape
    max_size = 20
    scale = max_size / max(height, width)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    digit_image = Image.fromarray((digit * 255).astype(np.uint8))

    digit_image = digit_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    digit = np.asarray(digit_image).astype("float32") / 255.0

    # Place the digit in the center of a 28×28 canvas
    canvas = np.zeros((28, 28), dtype="float32")

    top_offset = (28 - new_height) // 2
    left_offset = (28 - new_width) // 2

    canvas[
        top_offset:top_offset + new_height,
        left_offset:left_offset + new_width,
    ] = digit

    # Add batch dimension
    canvas = np.expand_dims(canvas, axis=0)

    return canvas