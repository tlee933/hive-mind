# My tiktoken Encodings

Custom tiktoken encodings for HiveCoder and local models.

## Install

```bash
pip install -e .
```

## Usage

```python
import tiktoken

# Use custom HiveCoder encoding
enc = tiktoken.get_encoding("hivecoder")
tokens = enc.encode("Hello, world!")
text = enc.decode(tokens)

# Count tokens
num_tokens = len(enc.encode("Your text here"))
```

## Available Encodings

| Name | Description |
|------|-------------|
| `hivecoder` | HiveCoder-7B encoding (cl100k_base + custom special tokens) |
| `qwen_base` | Qwen model placeholder (uses cl100k_base) |

## Customizing

Edit `tiktoken_ext/my_encodings.py` to:
- Add new encodings
- Modify special tokens
- Use different base vocabularies

## Integration with Hive-Mind

```python
import tiktoken
from hivemind_client import HiveMindClient

enc = tiktoken.get_encoding("hivecoder")
hive = HiveMindClient()

# Fast token counting before LLM calls
text = "Your prompt here"
token_count = len(enc.encode(text))
print(f"Prompt tokens: {token_count}")
```
