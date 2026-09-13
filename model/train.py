from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()