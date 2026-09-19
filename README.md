# Handwritten Digit Recognizer

A simple interactive handwritten digit recognition app built with **TensorFlow/Keras** and **Gradio**.

The app lets you draw a single digit (0–9), preprocesses the drawing into an MNIST-compatible image, and uses a CNN trained on MNIST to predict the digit.

## Features

- Draw a handwritten digit using a browser canvas
- Clear and redraw
- CNN-based digit classification
- Image preprocessing and centering
- Basic validation for empty or unclear/multiple drawings
- Local Gradio interface

## CNN Architecture

```text
28×28×1
   ↓
Conv2D (32 filters, 3×3) + ReLU
   ↓
MaxPooling2D (2×2)
   ↓
Conv2D (64 filters, 3×3) + ReLU
   ↓
MaxPooling2D (2×2)
   ↓
Flatten
   ↓
Dense (64) + ReLU
   ↓
Dense (10) + Softmax
```

## Model Result

The CNN achieved **99.07% test accuracy** on the MNIST test set.

This result is specific to MNIST. User-drawn digits can be more varied, so the model may still make incorrect predictions or classify unclear drawings as digits.

## Project Structure

```text
handwritten-digit-recognizer/
├── app/
│   └── app.py
├── model/
│   ├── data.py
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── validation.py
│   └── saved/
│       └── digit_model.keras
├── requirements.txt
└── .gitignore
```

## Run Locally

### 1. Clone the repository

```bash
git clone <https://github.com/Sandeep-060>
cd handwritten-digit-recognizer
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Script\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
python app/app.py
```

The Gradio interface will open locally in your browser.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- SciPy
- Gradio
