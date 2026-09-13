import numpy as np
from tensorflow import keras

def load_and_prepare_data():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Convert pixel values from [0, 255] to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_and_prepare_data()

    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Test images:", x_test.shape)
    print("Test labels:", y_test.shape)

    print("\nTraining pixel range:")
    print("Minimum:", np.min(x_train))
    print("Maximum:", np.max(x_train))

    print("\nFirst training label:", y_train[0])
    print("First training image shape:", x_train[0].shape)