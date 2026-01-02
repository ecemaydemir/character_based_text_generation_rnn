import numpy as np                        # numerical operations and array handling
from tensorflow.keras.models import Sequential    # sequential model architecture
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Dropout   # RNN layers, Dropout and output layer
from tensorflow.keras.utils import to_categorical  # for one-hot encoding

import warnings
warnings.filterwarnings("ignore")

import os
import re
import random

# --- 1. DATA PREPARATION ---

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
file_path = os.path.join(desktop_path, "alice.txt")

# Read the text file
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().lower()

# Remove unwanted characters
text = re.sub(r'[^a-z .,!?]', '', text)

# Use only the first 30,000 characters (as defined in your project)
text = text[:30000]

# Map each character to a unique numeric index
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# UPGRADE: Increased sequence length to 100 to capture better context
seq_length = 100   
step = 3           
X = []
y = []

for i in range(0, len(text) - seq_length, step):
    input_seq = text[i: i + seq_length]
    target_char = text[i + seq_length]
    X.append([char_to_idx[char] for char in input_seq])
    y.append(char_to_idx[target_char])

X = np.array(X)
y = np.array(y)

# One-hot encoding
X_encoded = np.eye(len(chars))[X]   # (num_samples, 100, num_characters)
y_encoded = np.eye(len(chars))[y]   # (num_samples, num_characters)

# --- 2. MODEL ARCHITECTURE (DEEP RNN) ---

model = Sequential()

# 1st RNN Layer: 256 units, return_sequences=True (to feed the next RNN layer)
model.add(SimpleRNN(256, return_sequences=True, input_shape=(seq_length, len(chars))))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# 2nd RNN Layer: 128 units (compressed representation)
model.add(SimpleRNN(128))
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(len(chars), activation='softmax'))

# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

# --- 3. TRAINING ---
# Train the model (batch_size=128, epochs=100)
model.fit(X_encoded, y_encoded, batch_size=128, epochs=100)

# --- 4. PREDICTION FUNCTIONS ---

def sample(preds, temperature=1.0):
    # Helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_sentence(model, seed_text, char_to_idx, idx_to_char, 
                      seq_length=100, max_len=50, temperature=0.7):
    """
    Generate text given a seed string using the trained RNN model.
    Stops early if an end-of-sentence punctuation mark is generated.
    """
    sentence = seed_text

    for _ in range(max_len):
        # Encode the current sequence (taking the last 'seq_length' characters)
        input_seq = [char_to_idx.get(c, 0) for c in sentence[-seq_length:]]

        # Padding (only needed if seed is shorter than 100 chars)
        if len(input_seq) < seq_length:
            input_seq = [0] * (seq_length - len(input_seq)) + input_seq

        input_encoded = np.eye(len(char_to_idx))[input_seq]
        input_encoded = input_encoded.reshape(1, seq_length, len(char_to_idx))

        # Predict the next character
        prediction = model.predict(input_encoded, verbose=0)
        next_index = sample(prediction[0], temperature=temperature)
        next_char = idx_to_char[next_index]

        sentence += next_char

        # Optional early stopping
        if next_char in ['.', '!', '?']:
            break

    return sentence

# --- 5. TEST GENERATION (WARM START) ---

print("--- Text Generation Test ---")

# Select a random starting point from the book to give full context
start_index = random.randint(0, len(text) - seq_length - 1)
long_seed_text = text[start_index : start_index + seq_length]

print(f"Seed Text (from book):\n'{long_seed_text}'\n")

# Try different temperature values to observe diversity
for temp in [0.2, 0.5, 0.8, 1.0]:
    print(f"\n--- Temp {temp} ---")
    generated = generate_sentence(model, long_seed_text, char_to_idx, idx_to_char, 
                                  seq_length=100, temperature=temp)
    
    # Print only the NEWLY generated part
    print(generated[len(long_seed_text):])

import matplotlib.pyplot as plt

# --- 6. VISUALIZATION: CHARACTER DIVERSITY ---

print("\n--- Generating Graph: Character Diversity vs Temperature ---")

temps = [0.2, 0.5, 0.8, 1.0]
diversity = []

# We use the 'long_seed_text' (100 chars) from the previous step as our fixed context
start_seed = long_seed_text 

# Loop through each temperature to measure diversity
for t in temps:
    # Generate text with the deep model and 100-char context
    gen = generate_sentence(model, start_seed, char_to_idx, idx_to_char, 
                            seq_length=100, temperature=t)
    
    # Analyze ONLY the newly generated characters (ignore the seed text)
    generated_part = gen[len(start_seed):]
    
    # Count unique characters in the generated part
    unique_count = len(set(generated_part))
    diversity.append(unique_count)
    
    print(f"Temp {t} -> Unique Chars: {unique_count}")

# Plotting the Bar Chart
positions = range(len(temps)) 

plt.figure(figsize=(6, 4))
# Create bars with a nice blue color and black edges
plt.bar(positions, diversity, width=0.45, edgecolor="black", color="#4c72b0")

plt.xticks(positions, temps)
plt.xlabel("Temperature")
plt.ylabel("Number of Unique Characters")
plt.title("Character Diversity vs Temperature")

# Adjust y-axis limit for better visibility
if len(diversity) > 0:
    plt.ylim(0, max(diversity) + 5)

plt.grid(axis="y", linestyle="--", alpha=0.4)

# Add numeric labels on top of bars
for x, val in zip(positions, diversity):
    plt.text(x, val + 0.1, str(val), ha="center", va="bottom", fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
