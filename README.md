# Simple ANN Library

Version: 1.0

A Simple tiny Artificial Neural Network (ANN) Library for synthesizing and training basic neural networks. This library includes a Gradient Descent Optimization method and is designed for creating simple neural networks. It is not specifically tailored for Natural Language Processing (NLP) models.

## Introduction

This repository contains a compact implementation of an Artificial Neural Network (ANN) library for creating and training basic neural networks. The library supports gradient descent optimization and provides building blocks for creating neural network layers, various activation functions, loss functions, and training utilities.

## Usage

1. Import the `aurora.py` file into your project.
2. Define your neural network architecture by creating layers and activations.
3. Create a network by combining layers and activations as needed.
4. Use the `train()` function to train your network on your data.
5. Utilize the `predict()` function to obtain predictions from your trained network.
6. Save and load trained networks using `save_model()` and `deploy_model()` functions.

## Features

- **Layer Class**: The library provides a `Layer` class for defining layers of the neural network. Each layer includes weights and biases.
- **Activation Functions**: Different activation functions such as Tanh, Sigmoid, ReLU, and more are available through the provided `Activations` class.
- **Loss Functions**: The library includes Mean Squared Error (MSE) and Binary Cross-Entropy loss functions.
- **Training Utilities**: The `train()` function allows you to train your neural network using gradient descent optimization.
- **Prediction**: The `predict()` function enables obtaining predictions from your trained network.
- **Model Saving**: Save and load trained models using the `save_model()` and `deploy_model()` functions.

## Example

```python
# Import the library
import ann

# Define your neural network architecture
layer1 = ann.Layer(input_size, hidden_size)
activation1 = ann.Tanh()
layer2 = ann.Layer(hidden_size, output_size)
activation2 = ann.Sigmoid()

# Create the network
network = [layer1, activation1, layer2, activation2]

# Train the network
ann.train(network, ann.mse, ann.mse_prime, X_train, Y_train, batch_size, epochs, learning_rate)

# Get predictions
predictions = ann.predict(network, X_test)

# Calculate accuracy
accuracy = ann.get_accuracy(predictions, Y_test)

# Save and deploy trained models
ann.save_model(network, "my_trained_model")
loaded_network = ann.deploy_model("my_trained_model.dat")
```

## Limitations

- This library is intended for educational purposes and simple experiments. It may not be suitable for complex neural network architectures.
- The optimization method is limited to basic gradient descent.
- The library is not optimized for Natural Language Processing (NLP) tasks.

## License

This project is licensed under the [MIT License](LICENSE).
