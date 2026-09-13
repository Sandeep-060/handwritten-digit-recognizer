from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from data import load_and_prepare_data


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    return model


def main():
    # Load and prepare MNIST
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    # Build the neural network
    model = build_model()

    # Configure the training process
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
    )

    # Evaluate on the test set
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=2,
    )

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Save the trained model
    model_path = Path("model") / "saved" / "digit_model.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()