# FemtoFlow: Deep Learning from Scratch with NumPy

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

FemtoFlow is a lightweight deep learning library built entirely from scratch using Python and NumPy. It aims to provide a clear and understandable implementation of core neural network components, suitable for educational purposes and experimentation.

The library focuses on implementing essential building blocks like layers, optimizers, loss functions, and metrics, allowing users to construct and train various neural network architectures. It also includes implementations of more advanced concepts crucial for modern deep learning, particularly in Natural Language Processing.

## Key Features

*   **Pure NumPy Implementation:** Built entirely using NumPy for numerical operations, making the underlying mechanics transparent.
*   **Modular Design:** Easily extensible architecture based on `Layer` and `Network` classes.
*   **Core Components:**
    *   **Layers:** Base `Layer`, `MetaLayer` (sequential container), `Activation`, `Dense1D`, `Dense2D`, `Embedding`, `PositionalEmbedding`, `LayerNormalisation`, `InvertedDropout`.
    *   **Optimizers:** Base optimizer interface integrated into layers (ADAM implementation mentioned, likely present in `femto_flow.optimizers`).
    *   **Loss Functions:** Interface for loss functions and their derivatives (CCE shown in example).
    *   **Metrics:** Interface for evaluation metrics (Categorical Accuracy shown in example).
*   **Advanced Features:**
    *   **Multi-Head Self-Attention:** Implementation (`MultiHeadSelfAttention`) crucial for Transformer architectures.
    *   **Byte-Pair Encoding (BPE):** A `BytePairTokenizer` class for subword tokenization.
    *   **Vectorizer:** Simple vocabulary mapping and sequence vectorization.
    *   **Positional Embeddings:** Standard sinusoidal positional encodings combined with token embeddings.
    *   **Layer Normalization:** Standard layer normalization implementation.
    *   **Weight Initialization:** Xavier/Glorot uniform initialization used in Dense/Embedding layers.
    *   **Learning Rate Schedules:** Support for dynamic learning rates (Example shows `ExponentialDecaySchedule` and basic `LearningRateSchedule`).
*   **Training Loop:** A flexible `fit` method in the `Network` class handling batching, epochs, validation, callbacks, and gradient clipping.

## Code Overview

The project is structured around several key modules (based on imports and typical structure):

*   `femto_flow/layers.py`: Contains definitions for all neural network layers. Each layer implements `forward` and `backward` methods.
*   `femto_flow/network.py`: Defines the `Network` class, responsible for assembling layers, managing the training process (`fit`), and making predictions (`predict`).
*   `femto_flow/tokenizers.py` Contains the `BytePairTokenizer` and `Vectorizer` classes for text processing.
*   `femto_flow/optimizers.py`: Contains optimizer implementations (like ADAM) and learning rate schedules.
*   `femto_flow/losses.py`: Defines loss functions (like CCE) and their derivatives.
*   `femto_flow/activations.py`: Defines activation functions (like Softmax, Swish) and their derivatives.
*   `femto_flow/metrics.py`: Defines evaluation metrics (like Categorical Accuracy).
*   `femto_flow/callbacks.py`: Contains callback classes for use during training (e.g., `PrintLRCallback`, `SaveOnProgressCallback`).
*   `demos/`: Contains example scripts showcasing how to use the library (like the provided generative C model).

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/TQCB/femto-flow
    cd femto_flow
    ```
2.  **Install dependencies:**
    ```bash
    pip install numpy regex
    ```

## Usage Example

Here's a simplified example based on the provided `main.py`, showing how to define and train a Transformer-like model:

```python
import femto_flow as ff
import femto_flow.layers as l
import numpy as np

# --- Configuration ---
output_size = 64
embed_size = 128
seq_len = 8
n_heads = 4
n_transformers = 3
dropout_rate = 0.1 # Using dropout now requires InvertedDropout

# --- Load Data (Example) ---
# x_train, y_train = load_your_batched_data(...)
# x_val, y_val = load_your_validation_data(...)
# Ensure data is NumPy arrays with shape (num_batches, batch_size, seq_len)

# --- Model Definition ---
model = ff.network.Network()

def create_transformer():
    # Example Transformer Block structure
    return l.MetaLayer([
        l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads),
        l.LayerNormalisation(embed_size),
        # FeedForward part
        l.Dense2D(input_dim=embed_size, output_dim=embed_size * 2), # Example expansion
        l.Activation(ff.activations.Swish), # Or ReLU, GeLU etc.
        l.InvertedDropout(dropout_rate),
        l.Dense2D(input_dim=embed_size * 2, output_dim=embed_size),
        l.InvertedDropout(dropout_rate),
        l.LayerNormalisation(embed_size), # Often applied after residual connection
    ])

# Input Embedding + Positional Encoding
model.add(l.PositionalEmbedding(seq_len=seq_len, output_dim=embed_size, vocab_size=output_size))
model.add(l.InvertedDropout(dropout_rate)) # Dropout after embedding

# Transformer Blocks
for _ in range(n_transformers):
    model.add(create_transformer())

# Final Layers for Classification/Generation
model.add(l.MultiHeadSelfAttention(input_dim=embed_size, n_dim=embed_size, n_heads=n_heads, return_sequences=False)) # Use last token output
model.add(l.LayerNormalisation(embed_size))
model.add(l.Dense1D(input_dim=embed_size, output_dim=output_size))
model.add(l.Activation(ff.activations.Softmax)) # Output probabilities

# --- Build & Train ---
lr_schedule = ff.optimizers.LearningRateSchedule(1e-3) # Or use a decay schedule
optimizer = ff.optimizers.AdamOptimizer # Or another optimizer

model.build(loss=ff.losses.cce,
            d_loss=ff.losses.d_cce,
            metric=ff.metrics.categorical_accuracy,
            optimizer=optimizer,
            learning_rate_schedule=lr_schedule)

print(f"Parameter count: {model.param_count:,.0f}")

# Define callbacks if needed
# save_cb = ff.callbacks.SaveOnProgressCallback('checkpoints')

# model.fit(x_train, y_train,
#           epochs=50,
#           x_val=x_val, y_val=y_val,
#           validation=True,
#           callbacks=[save_cb],
#           batch_print_steps=10)
```

## TODO / Future Work
*   Implement dropout layer: the standard Dropout layer currently raises NotImplementedError.
*   Refine BPE tokenizer: transform method in BytePairTokenizer using Trie needs thorough testing and potentially refinement for edge cases and efficiency. Add encoding/decoding pipeline methods.
*   Add more optimizers:
    *   SGD (with momentum)
    *   RMSprop
    *   AdaGrad
*   Expand loss functions
*   Add more activation functions
    *   GELU
*   Implement more layer types:
    *    Convolutional layers (Conv1D, Conv2D)
    *    Pooling layers (MaxPooling, AveragePooling)
    *    Recurrent layers (RNN, LSTM, GRU) - might be challenging with just NumPy but possible.
*   Model serialization: add functionality to save and load trained model weights and architecture.
*   Unit testing: develop a comprehensive test suite to ensure correctness of layers, optimizers, and training process.
*   Documentation: improve docstrings and potentially add Sphinx documentation.
*   Input validation: Add more robust checks for input shapes and types in layers.
*   Regularization: implement L1/L2 weight regularization options.
*   Explore multi-latent Attention

## Contributing

Not currently in need of contributions but they are always welcome.