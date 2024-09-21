import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the images
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encoding


# Define a function to create and compile the model
def create_model(activation):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=activation),
        Dense(64, activation=activation),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# List of activation functions to compare
activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu']

# Store results
results = {}

# Train and evaluate the model with different activation functions
for activation in activations:
    print(f"Training with {activation} activation function...")

    # Leaky ReLU needs a special handling since it's not directly in the list of activations
    if activation == 'leaky_relu':
        activation_layer = tf.keras.layers.LeakyReLU()
    else:
        activation_layer = activation

    # Create and train the model
    model = create_model(activation_layer)
    history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=0)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    results[activation] = {'accuracy': test_acc, 'loss': test_loss, 'history': history}
    print(f"{activation} Test Accuracy: {test_acc:.4f}")

# Plot the results
plt.figure(figsize=(12, 6))
for activation in activations:
    plt.plot(results[activation]['history'].history['val_accuracy'], label=f"{activation} val_accuracy")

plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()