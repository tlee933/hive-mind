# tiktoken for Python 3.14

Pre-built tiktoken wheel and custom encodings for Hive-Mind.

## Pre-built Wheel

```
wheels/tiktoken-0.12.0-cp314-cp314-linux_x86_64.whl
```

Install directly:
```bash
pip install wheels/tiktoken-0.12.0-cp314-cp314-linux_x86_64.whl
```

## Custom Encodings

Custom tiktoken encodings for HiveCoder and local models.

```python
import tiktoken
enc = tiktoken.get_encoding("hivecoder")  # Custom encoding
tokens = enc.encode("Hello world")
```

Install:
```bash
pip install -e my_encodings/
```

## Rebuild from Source

If you need to rebuild (e.g., new tiktoken version):

```bash
./build.sh
```

Requires:
- Rust 1.70+
- Python 3.14
- setuptools-rust

## Integration with hivemind_client

```python
from hivemind_client import tokenizer

# Fast token counting
count = tokenizer.count_tokens("Your text here")

# Chunking for embeddings
chunks = tokenizer.chunk_text(long_text, chunk_size=512, overlap=50)
```

## Why?

Python 3.14 is bleeding edge - tiktoken doesn't have official wheels yet.
This provides a pre-built wheel + custom encodings for the Hive-Mind stack.
