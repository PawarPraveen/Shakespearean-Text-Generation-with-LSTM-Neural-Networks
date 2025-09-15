# Shakespearean Text Generation with LSTM Neural Networks

A deep learning project that generates Shakespeare-style text using character-level LSTM networks trained on Hamlet.

## Project Overview

This project implements a character-level text generation system using Long Short-Term Memory (LSTM) neural networks. The model is trained on Shakespeare's Hamlet to learn linguistic patterns, grammatical structures, and thematic elements characteristic of Elizabethan English.

## Approach & Methodology

### Step 1: Data Acquisition & Preprocessing
- **Source**: NLTK's Gutenberg corpus containing Shakespeare's Hamlet
- **Cleaning**: Removed non-ASCII characters, normalized whitespace, preserved essential punctuation
- **Normalization**: Converted to lowercase while maintaining sentence structure cues

### Step 2: Character Encoding
- **Vocabulary Creation**: Extracted unique characters (letters, punctuation, spaces)
- **Mapping**: Created bidirectional dictionaries (char↔integer) for neural network processing
- **Sequence Length**: Used 25-character context window for optimal learning

### Step 3: Training Data Preparation
- **Sliding Window**: Generated sequences of 25 characters as input
- **Next Character Prediction**: Used the 26th character as target label
- **Dataset Size**: Created ~200,000 training samples from the text

### Step 4: Model Architecture
```python
Embedding(64) → LSTM(128) → Dropout(0.3) → LSTM(64) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Output
