import sys
from pathlib import Path

import numpy as np
import gradio as gr


# ---------------------------------------------------------
# Make project root available
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

from model.preprocess import preprocess_image, analyze_drawing
from model.inference import load_model, predict_digit


# ---------------------------------------------------------
# Load model once
# ---------------------------------------------------------

model = load_model()


# ---------------------------------------------------------
# Canvas configuration
# ---------------------------------------------------------

CANVAS_SIZE = 520

# White blank canvas.
# User draws black digits on white background.
BLANK_CANVAS = np.full(
    (CANVAS_SIZE, CANVAS_SIZE, 3),
    255,
    dtype=np.uint8,
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

def predict(image_editor_value):

    if image_editor_value is None:
        return "Please draw one digit (0–9)."

    # ImageEditor normally gives us a dictionary.
    # We want the final combined image.
    if isinstance(image_editor_value, dict):

        image = image_editor_value.get("composite")

        if image is None:
            image = image_editor_value.get("background")

        if image is None:
            return "Please draw one digit (0–9)."

    else:
        image = image_editor_value

    # -----------------------------------------------------
    # Analyze drawing before sending it to the model
    # -----------------------------------------------------

    analysis = analyze_drawing(image)

    if analysis["empty"]:
        return "Please draw one digit (0–9)."

    if analysis["component_count"] > 3:
        return "Please draw only one digit at a time."

    # -----------------------------------------------------
    # Convert drawing into MNIST-like 28x28 input
    # -----------------------------------------------------

    processed_image = preprocess_image(image)

    if processed_image is None:
        return "Please draw one digit (0–9)."

    # -----------------------------------------------------
    # Ask the trained Keras model for prediction
    # -----------------------------------------------------

    result = predict_digit(
        model,
        processed_image,
        component_count=analysis["component_count"],
    )

    # -----------------------------------------------------
    # Apply validation rules
    # -----------------------------------------------------

    if not result["valid"]:
        return result["message"]

    return (
        f"Predicted digit: {result['digit']}\n"
        f"Confidence: {result['confidence']:.2%}"
    )


# ---------------------------------------------------------
# Clear canvas
# ---------------------------------------------------------

def clear_canvas():
    """
    Reset the drawing to a blank white canvas.

    Returning a real blank image instead of None keeps the
    ImageEditor initialized instead of putting it into an
    empty/uninitialized state.
    """

    return BLANK_CANVAS.copy(), ""


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

css = """
#app-container {
    max-width: 1250px;
    margin: 0 auto;
}

#title {
    text-align: center;
    margin-bottom: 6px;
}

#subtitle {
    text-align: center;
    margin-bottom: 25px;
}

#canvas-column {
    min-width: 0;
}

#result-column {
    min-width: 320px;
}

#canvas-box {
    display: flex;
    justify-content: center;
}

#result-box {
    min-height: 180px;
}

#button-row {
    margin-top: 18px;
}

#predict-button,
#clear-button {
    min-height: 52px;
    font-size: 18px;
}
"""


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------

with gr.Blocks(
    title="Handwritten Digit Recognizer",
    css=css,
) as demo:

    with gr.Column(elem_id="app-container"):

        # -------------------------------------------------
        # Header
        # -------------------------------------------------

        gr.Markdown(
            "# Handwritten Digit Recognizer",
            elem_id="title",
        )

        gr.Markdown(
            "Draw one handwritten digit (0–9) and click Predict.",
            elem_id="subtitle",
        )

        # -------------------------------------------------
        # Main layout
        # -------------------------------------------------

        with gr.Row(equal_height=True):

            # ---------------------------------------------
            # LEFT: CANVAS
            # ---------------------------------------------

            with gr.Column(
                scale=3,
                elem_id="canvas-column",
            ):

                gr.Markdown("### Draw your digit")

                with gr.Column(elem_id="canvas-box"):

                    canvas = gr.ImageEditor(
                        value=BLANK_CANVAS,
                        sources=(),
                        type="numpy",
                        image_mode="RGB",

                        canvas_size=(
                            CANVAS_SIZE,
                            CANVAS_SIZE,
                        ),

                        fixed_canvas=True,

                        # Hide layers.
                        layers=False,

                        # No crop/resize tools.
                        transforms=[],

                        # Simple black drawing brush.
                        brush=gr.Brush(
                            colors=["#000000"],
                            default_color="#000000",
                            color_mode="fixed",
                            default_size=18,
                        ),

                        width=CANVAS_SIZE,
                        height=CANVAS_SIZE,

                        show_label=False,
                    )

            # ---------------------------------------------
            # RIGHT: RESULT
            # ---------------------------------------------

            with gr.Column(
                scale=2,
                elem_id="result-column",
            ):

                gr.Markdown("### Prediction")

                result_box = gr.Textbox(
                    label=None,
                    placeholder=(
                        "Draw a digit on the left "
                        "and click Predict."
                    ),
                    lines=5,
                    interactive=False,
                    elem_id="result-box",
                )

                # -----------------------------------------
                # Buttons
                # -----------------------------------------

                with gr.Row(elem_id="button-row"):

                    clear_button = gr.Button(
                        "Clear",
                        variant="secondary",
                        elem_id="clear-button",
                    )

                    predict_button = gr.Button(
                        "Predict",
                        variant="primary",
                        elem_id="predict-button",
                    )


        # -------------------------------------------------
        # Events
        # -------------------------------------------------

        predict_button.click(
            fn=predict,
            inputs=canvas,
            outputs=result_box,
            show_progress="hidden",
        )

        clear_button.click(
            fn=clear_canvas,
            inputs=None,
            outputs=[canvas, result_box],
            show_progress="hidden",
        )


# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":
    demo.launch()