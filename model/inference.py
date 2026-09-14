import numpy as np
from tensorflow import keras


MODEL_PATH = "model/saved/digit_model.keras"
CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.20

def load_model():
    """Load the trained digit-recognition model."""
    return keras.models.load_model(MODEL_PATH)

def predict_digit(model, processed_image, component_count=1):
    """
    Predict a digit from an already-preprocessed image.

    Expected input shape:
        (1, 28, 28)

    Returns:
        Dictionary containing prediction and validation information.
    """

    probabilities = model.predict(
        processed_image,
        verbose=0,
    )[0]

    sorted_probabilities = np.sort(probabilities)

    confidence = float(sorted_probabilities[-1])
    second_highest = float(sorted_probabilities[-2])

    margin = confidence - second_highest

    predicted_digit = int(np.argmax(probabilities))

    from .validation import validate_prediction

    valid = validate_prediction(
        confidence=confidence,
        margin=margin,
        component_count=component_count,
    )

    if valid:
        message = None
    else:
        message = (
            "Couldn't recognize a single digit. "
            "Please draw one clear digit (0–9)."
        )

    return {
    "valid": valid,
    "digit": predicted_digit,
    "confidence": confidence,
    "margin": margin,
    "probabilities": probabilities,
    "message": message,
    }
