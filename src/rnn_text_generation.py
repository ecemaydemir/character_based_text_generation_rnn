import numpy as np                               # numerical operations and array handling
from tensorflow.keras.models import Sequential    # sequential model architecture
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding   # RNN layers and output layer
from tensorflow.keras.utils import to_categorical  # for one-hot encoding

import warnings
warnings.filterwarnings("ignore")

import os
import re

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "alice.txt")

# Read the text file
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().lower()

# Remove unwanted characters
text = re.sub(r'[^a-z .,!?]', '', text)

# Use only the first 30,000 characters
text = text[:30000]

# Map each character to a unique numeric index
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

seq_length = 40   # we feed 40 characters and try to predict the 41st
step = 3          # take a new sample every 3 characters
X = []
y = []

for i in range(0, len(text) - seq_length, step):
    input_seq = text[i: i + seq_length]
    target_char = text[i + seq_length]
    X.append([char_to_idx[char] for char in input_seq])
    y.append(char_to_idx[target_char])

X = np.array(X)
y = np.array(y)

# np.eye() creates an identity matrix of size n x n
X_encoded = np.eye(len(chars))[X]   # (num_samples, 40, num_characters)
y_encoded = np.eye(len(chars))[y]   # (num_samples, num_characters)

# define a simple character-level RNN model

model = Sequential()
model.add(SimpleRNN(128, input_shape=(seq_length, len(chars))))   # recurrent layer with 128 units
model.add(Dense(len(chars), activation='softmax'))                # output layer over all characters

# compile with categorical cross-entropy since we are predicting a single character class
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

# train the model
model.fit(X_encoded, y_encoded, batch_size=128, epochs=100)

# Prediction Function
def sample(preds, temperature=1.0):
    # sampling function that adjusts the probability distribution
    # using the temperature parameter for more or less randomness
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_sentence(model, seed_text, char_to_idx, idx_to_char,
                      seq_length=40, max_len=50, temperature=0.7):
    """
    Generate text given a seed string using the trained RNN model.
    Stops early if an end-of-sentence punctuation mark is generated.
    """
    sentence = seed_text

    for _ in range(max_len):
        # encode the current sequence
        input_seq = [char_to_idx.get(c, 0) for c in sentence[-seq_length:]]

        # pad if shorter than required length
        if len(input_seq) < seq_length:
            input_seq = [0] * (seq_length - len(input_seq)) + input_seq

        input_encoded = np.eye(len(char_to_idx))[input_seq]
        input_encoded = input_encoded.reshape(1, seq_length, len(char_to_idx))

        # predict the next character
        prediction = model.predict(input_encoded, verbose=0)
        next_index = sample(prediction[0], temperature=temperature)
        next_char = idx_to_char[next_index]

        sentence += next_char

        # optional early stopping
        if next_char in ['.', '!', '?']:
            break

    return sentence

# Test text generation
start = "alice was beginning to get very tired "
print("Generated sentence:\n")

# lower temperature = more deterministic predictions
print(generate_sentence(model, start, char_to_idx, idx_to_char, temperature=0.2))

start = "alice was sitting"

# try different temperature values to observe diversity in generation
for temp in [0.2, 0.5, 0.8, 1.0]:
    print(f"\nTemp {temp}:")
    print(generate_sentence(model, start, char_to_idx, idx_to_char, temperature=temp)
