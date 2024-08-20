# Universal Sentence Encoder with TensorFlow Lite Conversion

## Overview

This project showcases how to create a TensorFlow Keras model with a custom layer that utilizes the Universal Sentence Encoder (USE) from TensorFlow Hub. The model is saved in Keras format and then converted to TensorFlow Lite using the Flex converter, allowing for complex operations that are not natively supported by TensorFlow Lite.

## Project Structure

- **`convert_model.py`**: The main script that creates, saves, and converts the model.
- **`model.keras`**: The saved Keras model file.
- **`model_flex.tflite`**: The TensorFlow Lite model file with Flex ops.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- TensorFlow Hub

You can install the required packages using pip:

```bash
pip install tensorflow tensorflow-hub
