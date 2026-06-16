# metal_ai

A macOS Metal GPU framework for training and inference of Transformer language models — powered by tinygrad, custom Metal shaders, and a self-managed GPU buffer pool.

---

## Installation

```bash
# Install from source (setup.py + metal_ai.py in the same folder)
pip install -e .

# Or from a built wheel
pip install metal_ai-0.1.0-py3-none-any.whl
```

**Requirements:** macOS, Python ≥ 3.10, Metal-capable GPU (Apple Silicon or AMD).

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
```

### 5. Train

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
```

Resume from a checkpoint:

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

### 6. Generate text

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

### 7. Save and load

```python
# Save as binary .bybyai (fast, compact)
model.save("my_model.bybyai")
model.load("my_model.bybyai")

# Save as safetensors (HuggingFace compatible)
model.save("my_model.safetensors")
```

### 8. Inspect a checkpoint without loading weights

```python
info = MetalCharLM.peek_checkpoint("checkpoint_step50.bybyai")
print(info)
# {'step': 50, 'vocab': 128, 'embed_dim': 256, 'hidden_dim': 256}
```

### 9. Merge checkpoints (model averaging)

```python
from metal_ai import merge_all_checkpoints

# Averages all checkpoint_stepN.bybyai files in the current directory
averaged_weights = merge_all_checkpoints()
```

### 10. Streaming training (large datasets, low RAM)

```python
def my_generator():
    for chunk in load_chunks():
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

### 11. Fine-tune

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

---

## API Reference

| Name | Description |
|------|-------------|
| `MetalCharLM` | Main model — transformer LM on Metal GPU |
| `CharTokenizer` | Character-level tokenizer |
| `build_dataset` | Build `(X, Y)` from a list of strings |
| `merge_all_checkpoints` | Average weights across multiple checkpoints |
| `MetalTensor` | Low-level Metal GPU tensor |
| `MetalLinear` | Linear layer on Metal |
| `MetalLayerNorm` | Layer normalization on Metal |
| `MetalAdam` | Pure-NumPy Adam optimizer |
| `MetalSequential` | Sequential layer container |
| `sample_top_p` | Top-p sampling for generation |
| `gpu_cleanup` | Clear GPU buffer pool |
| `cleanup_tensors` | Clear tinygrad tensor cache |

---

## Checkpoint formats

| Format | Extension | When to use |
|--------|-----------|-------------|
| Custom binary | `.bybyai` | Default — fastest, smallest |
| SafeTensors | `.safetensors` | Sharing with HuggingFace ecosystem |
| Pickle state dict | `.bybyai` (auto) | Auto-saved every step during training |

---

## Notes

- **macOS only** — requires Metal runtime and PyObjC.
- `attention.metal` and `metal_kernels.metal` must be in the **same directory** as your script at runtime.
- GPU RAM is capped at 70 GB by default (`GPU_RAM_LIMIT` in `metal_ai.py`). Adjust as needed.
