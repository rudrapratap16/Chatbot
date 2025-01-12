# Chatbot Intent Classification Model

This repository contains a chatbot intent classification model built with Python, TensorFlow, and scikit-learn. The model uses **TF-IDF vectorization** to preprocess text and a **feed-forward neural network** to classify user inputs into predefined intents based on a JSON file. The responses are generated dynamically for each intent.

---

## Features
- **TF-IDF Vectorizer**: Converts input text into numerical vectors.
- **Neural Network Model**: Multi-layer perceptron with dense layers for intent classification.
- **Dynamic Responses**: Generates random responses from a predefined list for each intent.
- **Serialization**: Saves and loads the trained model using Python's `pickle`.
- **Customizable**: Modify the `intents.json` file to add your own intents, patterns, and responses.

---

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- scikit-learn
- numpy
- pickle
- json

You can install the required libraries using `pip`:

```bash
pip install tensorflow scikit-learn numpy
