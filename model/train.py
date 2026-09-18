from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from data import load_and_prepare_data


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32,kernel_size=(3, 3),activation="relu",),
        layers.MaxPooling2D(pool_size=(2, 2),),
        layers.Conv2D(64,kernel_size=(3, 3),activation="relu",),
        layers.MaxPooling2D(pool_size=(2, 2),),
        layers.Flatten(),
        layers.Dense(64,activation="relu",),
        layers.Dense(10,activation="softmax",),
    ])
    return model


def main():
    # Load and prepare MNIST
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    # Add the grayscale channel dimension for CNN
    x_train = x_train[..., None]
    x_test = x_test[..., None]

    # Build the CNN
    model = build_model()

    # Configure the training process
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Show the CNN architecture
    model.summary()

    # Train the CNN
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

    # Save the trained CNN
    model_path = Path("model") / "saved" / "digit_model.keras"
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save(model_path)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()