# Character-Level Text Generation with Deep RNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview
This project implements a **character-level Deep Recurrent Neural Network (Deep RNN)** trained on text from *Alice in Wonderland*. The model learns to predict the next character in a sequence given a context window.

The primary focus of this experiment is to analyze:
1.  **Deep Architecture:** The effect of stacking RNN layers on learning capability.
2.  **Context Awareness:** Using a longer sequence length (100 characters) to capture better semantic dependencies.
3.  **Temperature Sampling:** How the temperature parameter controls the trade-off between structural stability (determinism) and creativity (randomness).

## Key Features
- **Model Type:** Deep SimpleRNN (Stacked Architecture)
- **Task:** Character-level next-character prediction
- **Input Sequence:** 100 characters (Warm Start)
- **Regularization:** Dropout (0.2) to prevent overfitting
- **Dataset:** *Alice in Wonderland* (Project Gutenberg)

## Data Processing
1.  Text is converted to lowercase and cleaned using regex (non-alphabetic characters removed).
2.  Unique characters are mapped to integer indices (Character-to-Index mapping).
3.  Training samples are created using a sliding window approach with **seq_length=100**.
4.  One-hot encoding is applied to both inputs and target labels.

## Model Architecture
The architecture was upgraded from a shallow network to a **Deep RNN** to improve context learning:

- **Layer 1:** SimpleRNN (256 units, `return_sequences=True`)
- **Dropout:** 0.2
- **Layer 2:** SimpleRNN (128 units)
- **Dropout:** 0.2
- **Output:** Dense layer with Softmax activation
- **Total Parameters:** ~100,000+

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

model = Sequential()
# Layer 1: Returns sequences to feed the next RNN layer
model.add(SimpleRNN(256, return_sequences=True, input_shape=(100, vocab_size)))
model.add(Dropout(0.2))

# Layer 2: Compresses information into a single vector
model.add(SimpleRNN(128))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
