# Character-Level Text Generation with RNN

This project implements a character-level Recurrent Neural Network (SimpleRNN) trained on text from *Alice in Wonderland* in order to generate new sentences by predicting the next character in a sequence. The focus of the project is to examine how the temperature parameter affects the balance between determinism and creativity in generated text.

## Project Overview
- Model type: SimpleRNN
- Task: Character-level next-character prediction
- Input: 40-character sequences
- Dataset source: Alice in Wonderland text file
- Output: Generated text with controlled randomness (temperature)

## Data Processing
1. Text is converted to lowercase and cleaned using regex.
2. Unique characters are mapped to integer indices.
3. Training samples are created with a sliding window approach.
4. One-hot encoding is used for both inputs and labels.

## Model Architecture
- SimpleRNN layer with 128 units
- Dense output layer with softmax activation
- Loss: Categorical crossentropy
- Optimizer: Adam
- Total parameters: ~24,000

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential()
model.add(SimpleRNN(128, input_shape=(40, vocab_size)))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

## Temperature Parameter

Temperature controls sampling randomness:

| Temperature | Output Type                 |
|-------------|------------------------------|
| 0.2         | Deterministic, repetitive    |
| 0.5         | Balanced                     |
| 0.8 - 1.0   | Creative, unstable           |

Example output:
```
alice was sitting wout at irsceriaco verter , and sored.
```

## Project Structure
```
.
├── Alice_Text_Generation_RNN.pdf      # Full project report
├── data/                              # Dataset (alice.txt)
├── src/                               # Code scripts 
└── README.md
```

## Future Work
- Extension to time series anomaly detection
- LSTM Autoencoders for reconstruction-based anomaly scoring
- Application to sensor data (e.g., NASA Bearing Dataset)

## Author
Ecem Aydemir, 2025
