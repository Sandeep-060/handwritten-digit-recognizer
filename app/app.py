import sys
from pathlib import Path
import gradio as gr
import numpy as np

# Make project root available
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Project imports
from model.preprocess import preprocess_image, analyze_drawing
from model.inference import load_model, predict_digit

# Load model once
model = load_model()

# Canvas configuration
CANVAS_SIZE = 520
BLANK_CANVAS = np.full(
    (CANVAS_SIZE, CANVAS_SIZE, 3),
    255,
    dtype=np.uint8,
)

# Extract image from ImageEditor
def extract_image(image_editor_value):
    if image_editor_value is None:
        return None

    if isinstance(image_editor_value, dict):
        image = image_editor_value.get("composite")

        if image is None:
            image = image_editor_value.get("background")

        return image

    return image_editor_value


# Prediction
def predict(image_editor_value):

    # Get the current image directly from ImageEditor
    image = extract_image(image_editor_value)

    if image is None:
        return "Draw a digit first."

    # Analyze drawing
    analysis = analyze_drawing(image)

    if analysis["empty"]:
        return "Draw a digit first."

    # Detect multiple separated regions
    gray = np.asarray(image)

    if gray.ndim == 3 and gray.shape[-1] == 4:
        gray = gray[..., :3]

    if gray.ndim == 3:
        gray = np.mean(gray, axis=-1)

    gray = gray.astype("float32")

    if gray.max() > 1.0:
        gray = gray / 255.0

    # Convert black drawing on white background
    # into white drawing on black background.
    if gray.mean() > 0.5:
        gray = 1.0 - gray

    mask = gray > 0.1

    # Find horizontal gaps between drawing regions
    columns = np.any(mask, axis=0)

    horizontal_regions = 0
    inside_region = False

    for value in columns:

        if value and not inside_region:
            horizontal_regions += 1
            inside_region = True

        elif not value:
            inside_region = False


    # More than one separate horizontal region
    # is treated as multiple digits.
    if horizontal_regions > 1:
        return "Please draw only one digit at a time."

    # Preprocess
    processed_image = preprocess_image(image)

    if processed_image is None:
        return "Draw a digit first."

    # CNN prediction
    result = predict_digit(
        model,
        processed_image,
        component_count=analysis["component_count"],
    )

    # Validation
    if not result["valid"]:
        return "Couldn't recognize a clear digit."

    # Display only the predicted digit
    return f"""
<div class="prediction-content">
    <div class="prediction-label">Predicted digit</div>
    <div class="prediction-digit">{result["digit"]}</div>
</div>
"""

# Clear canvas
def clear_canvas():
    return BLANK_CANVAS.copy(), ""


# CSS
css = """
#app-container {
    max-width: 1150px;
    margin: 0 auto;
}

#title {
    text-align: center;
    margin-bottom: 4px;
}

#subtitle {
    text-align: center;
    margin-bottom: 30px;
}

#canvas-column,
#result-column {
    min-width: 0;
}

#canvas-box {
    display: flex;
    justify-content: center;
}

#prediction-box {
    min-height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
}

.prediction-content {
    text-align: center;
    width: 100%;
}

.prediction-label {
    font-size: 20px;
    margin-bottom: 10px;
}

.prediction-digit {
    font-size: 120px;
    line-height: 1;
    font-weight: 700;
}

#button-row {
    margin-top: 22px;
}

#predict-button,
#clear-button {
    min-height: 54px;
    font-size: 18px;
    font-weight: 600;
}
"""

# UI
with gr.Blocks(
    title="Handwritten Digit Recognizer",
    css=css,
) as demo:

    with gr.Column(elem_id="app-container"):

        # Header
        gr.Markdown(
            "# Handwritten Digit Recognizer",
            elem_id="title",
        )

        gr.Markdown(
            "Draw one handwritten digit (0–9) and click Predict.",
            elem_id="subtitle",
        )

        # Main layout
        with gr.Row(equal_height=True):

            # LEFT: CANVAS
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
                        layers=False,
                        transforms=[],
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

            # RIGHT: PREDICTION
            with gr.Column(
                scale=2,
                elem_id="result-column",
            ):

                gr.Markdown("### Prediction")

                result_box = gr.Markdown(
                    value="""
<div class="prediction-content">
    <div class="prediction-label">Draw a digit</div>
</div>
""",
                    elem_id="prediction-box",
                )

                # Buttons
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


        # Predict
        predict_button.click(
            fn=predict,
            inputs=canvas,
            outputs=result_box,
            show_progress="hidden",
        )

        # Clear
        clear_button.click(
            fn=clear_canvas,
            inputs=None,
            outputs=[canvas, result_box],
            show_progress="hidden",
        )

# Start application
if __name__ == "__main__":
    demo.launch()