# metal_ai

A macOS Metal GPU framework for training and inference of Transformer language models — powered by tinygrad, custom Metal shaders, and a self-managed GPU buffer pool.

---

## Installation

```bash
pip install git+https://github.com/bobbyshop-vui/bybyai-framework
```

**Requirements:** macOS, Python ≥ 3.10, Apple Silicon (M1/M2/M3/M4) or any Mac with Metal support.

---

## Quick Start

### 1. Import

```python
from metal_ai import MetalCharLM, CharTokenizer, build_dataset
```

### 2. Create a tokenizer

```python
texts = ["hello world", "training data here", "more text..."]
tok = CharTokenizer(texts)
print(tok.vocab_size)   # number of unique characters

# Save vocab to JSON
tok.save("tokenizer.json")
```

### 3. Build a dataset

```python
X, Y = build_dataset(texts, tok, seq_len=32)

# Limit number of samples
X, Y = build_dataset(texts, tok, seq_len=32, max_samples=10000)
```

### 4. Create a model

```python
model = MetalCharLM(
    vocab      = tok.vocab_size,
    embed_dim  = 256,
    hidden_dim = 256,
    num_layers = 8,
    lr         = 1e-3,
)
# Prints: >>> Model initialized: vocab=..., hidden_dim=256, layers=8
```

### 5. Train

Checkpoints are saved **automatically** every step as `checkpoint_step{N}.bybyai` in the current directory. You do not choose the filename — it is always `checkpoint_step1.bybyai`, `checkpoint_step2.bybyai`, etc.

```python
trainer = MetalCharLM(vocab=tok.vocab_size)

trained_model = trainer.train(
    model      = model,
    X          = X,
    Y          = Y,
    steps      = 100,
    BATCH_SIZE = 32,
    lr_base    = 1e-4,
)
# Auto-saves: checkpoint_step1.bybyai, checkpoint_step2.bybyai, ...
```

Resume from the latest checkpoint automatically:

```python
# trainer.train() auto-detects the latest checkpoint_stepN.bybyai
# in the current directory and resumes from it
trained_model = trainer.train(
    model      = model,
    X          = X,
    Y          = Y,
    steps      = 200,
    BATCH_SIZE = 32,
    lr_base    = 1e-4,
)
```

Or resume from a specific checkpoint:

```python
trained_model = trainer.train(
    model                = model,
    X                    = X,
    Y                    = Y,
    steps                = 200,
    BATCH_SIZE           = 32,
    checkpoint_to_resume = "checkpoint_step50.bybyai",
    lr_base              = 1e-4,
)
```

### 6. Streaming training (large datasets that don't fit in RAM)

Same auto-checkpoint behaviour — saves `checkpoint_step{N}.bybyai` every step.

```python
def my_generator():
    for chunk in load_chunks():          # your own chunk loader
        X, Y = build_dataset(chunk, tok, seq_len=32)
        yield X, Y

trainer.train_streaming(
    model           = model,
    batch_generator = my_generator(),
    steps           = 1000,
    BATCH_SIZE      = 32,
    lr_base         = 1e-4,
)
```

### 7. Fine-tune

```python
trainer.finetune(
    model      = model,
    tokenizer  = tok,
    texts      = new_texts,
    X          = False,   # False = auto-build from texts
    Y          = False,
    seq_len    = 32,
    steps      = 50,
    batch_size = 16,
    lr_base    = 5e-5,
)
```

### 8. Generate text

```python
output = model.generate(
    tokenizer   = tok,
    prompt      = "hello",
    max_new     = 200,
    ctx         = 512,
    temperature = 0.8,
    top_p       = 0.9,
)
print(output)
```

### 9. Save and load model

```python
# Binary format (fast, compact)
model.save("my_model.bybyai")
model.load("my_model.bybyai")

# SafeTensors format (HuggingFace compatible)
model.save("my_model.safetensors")
```

### 10. Inspect a checkpoint without loading weights

```python
info = MetalCharLM.peek_checkpoint("checkpoint_step50.bybyai")
print(info)
# {'step': 50, 'vocab': 128, 'embed_dim': 256, 'hidden_dim': 256}
```

### 11. Merge / average all checkpoints

```python
from metal_ai import merge_all_checkpoints

# Finds all checkpoint_stepN.bybyai in current directory,
# averages their weights, and returns the result dict
averaged = merge_all_checkpoints()
```

---

## API Reference

| Name | Description |
|------|-------------|
| `MetalCharLM` | Main model — transformer LM on Metal GPU |
| `CharTokenizer` | Character-level tokenizer |
| `build_dataset(texts, tok, seq_len, max_samples)` | Build `(X, Y)` from a list of strings |
| `merge_all_checkpoints()` | Average weights across all `checkpoint_stepN.bybyai` files |
| `MetalTensor` | Low-level Metal GPU tensor |
| `MetalLinear` | Linear layer on Metal |
| `MetalLayerNorm` | Layer normalization on Metal |
| `MetalAdam` | Pure-NumPy Adam optimizer |
| `MetalSequential` | Sequential layer container |
| `sample_top_p(logits, p, temperature)` | Top-p nucleus sampling |
| `gpu_cleanup()` | Trim GPU buffer pool |
| `cleanup_tensors()` | Clear tinygrad tensor cache |

---

## Checkpoint behaviour

Checkpoints are **always** named `checkpoint_step{N}.bybyai` and saved in the **current working directory**. There is no option to change the filename or location. On resume, the trainer scans the current directory for the latest `checkpoint_stepN.bybyai` automatically.

| Format | Extension | Description |
|--------|-----------|-------------|
| Training state | `checkpoint_step{N}.bybyai` | Auto-saved every step; contains weights + step number |
| Full model binary | `.bybyai` (via `model.save()`) | Compact binary; load with `model.load()` |
| SafeTensors | `.safetensors` | HuggingFace-compatible export |

---

## Metal Math Operations

Low-level GPU operations available as standalone functions. All inputs can be either `MetalTensor` or `numpy.ndarray` — they are converted automatically.

```python
from metal_ai import (
    metal_add, metal_mul, metal_matmul,
    metal_relu, metal_sigmoid, metal_tanh,
    metal_softmax, metal_layernorm,
    metal_mean, metal_sum, metal_reshape,
    metal_embedding_lookup,
)
```

### Arithmetic

```python
# Element-wise addition
c = metal_add(a, b)        # a + b

# Element-wise multiplication
c = metal_mul(a, b)        # a * b

# Matrix multiplication (2D only)
c = metal_matmul(a, b)     # (M, K) @ (K, N) → (M, N)
```

### Activations

```python
y = metal_relu(x)          # max(0, x)
y = metal_sigmoid(x)       # 1 / (1 + exp(-x))
y = metal_tanh(x)          # tanh(x)
y = metal_softmax(x, axis=-1)   # softmax along last axis only
```

### Normalization

```python
# Layer normalization
# weight and bias must match the last dim of x
y = metal_layernorm(x, weight, bias, eps=1e-5)
```

### Reductions

```python
mean_all = metal_mean(x)              # scalar mean over all elements
mean_ax  = metal_mean(x, axis=1)      # mean along axis

sum_all  = metal_sum(x)               # scalar sum
sum_ax   = metal_sum(x, axis=0)       # sum along axis
```

### Shape & Indexing

```python
# Reshape (metadata-only, no GPU copy)
y = metal_reshape(x, (new_shape,))

# Embedding lookup — indices shape (B, T) → output (B, T, embed_dim)
out = metal_embedding_lookup(embed_table, indices)
```

### Notes

- All operations run on the Metal GPU and return a `MetalTensor`.
- Call `.numpy()` on the result to get a `numpy.ndarray` back on CPU.
- `metal_softmax` only supports the last axis (`axis=-1`).
- `metal_matmul` requires exactly 2D inputs `(M, K)` and `(K, N)`.

```python
# Example
import numpy as np
from metal_ai import MetalTensor, metal_relu, metal_matmul

a = MetalTensor(np.random.randn(4, 8).astype("float32"))
b = MetalTensor(np.random.randn(8, 16).astype("float32"))

c = metal_matmul(a, b)     # MetalTensor shape (4, 16)
c = metal_relu(c)           # MetalTensor shape (4, 16)
print(c.numpy())            # numpy array back on CPU
```

- **macOS only** — requires Metal runtime and PyObjC.
- `attention.metal` and `metal_kernels.metal` must be in the **same directory** as your script at runtime.
- GPU RAM is capped at 70 GB by default (`GPU_RAM_LIMIT`). Edit `metal_ai.py` to change it.
