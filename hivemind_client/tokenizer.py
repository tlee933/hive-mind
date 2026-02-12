"""
Fast tokenizer utilities using tiktoken.

tiktoken is Rust-based and significantly faster than Python tokenizers.
Use for token counting, context management, and text chunking.
"""

import tiktoken
from functools import lru_cache
from typing import List, Optional, Tuple


@lru_cache(maxsize=4)
def get_encoder(encoding_name: str = "hivecoder") -> tiktoken.Encoding:
    """Get a cached tiktoken encoder."""
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding: str = "hivecoder") -> int:
    """
    Fast token count using tiktoken.

    Args:
        text: Text to tokenize
        encoding: Encoding name (hivecoder, cl100k_base, etc.)

    Returns:
        Number of tokens
    """
    enc = get_encoder(encoding)
    return len(enc.encode(text))


def encode(text: str, encoding: str = "hivecoder") -> List[int]:
    """Encode text to token IDs."""
    enc = get_encoder(encoding)
    return enc.encode(text)


def decode(tokens: List[int], encoding: str = "hivecoder") -> str:
    """Decode token IDs to text."""
    enc = get_encoder(encoding)
    return enc.decode(tokens)


def truncate_to_tokens(
    text: str,
    max_tokens: int,
    encoding: str = "hivecoder",
    suffix: str = "..."
) -> str:
    """
    Truncate text to fit within max_tokens.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        encoding: Encoding name
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    enc = get_encoder(encoding)
    tokens = enc.encode(text)

    if len(tokens) <= max_tokens:
        return text

    # Reserve space for suffix
    suffix_tokens = enc.encode(suffix)
    truncated = tokens[:max_tokens - len(suffix_tokens)]

    return enc.decode(truncated) + suffix


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding: str = "hivecoder"
) -> List[str]:
    """
    Split text into chunks by token count with overlap.

    Useful for embedding long documents.

    Args:
        text: Text to chunk
        chunk_size: Tokens per chunk
        overlap: Overlapping tokens between chunks
        encoding: Encoding name

    Returns:
        List of text chunks
    """
    enc = get_encoder(encoding)
    tokens = enc.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))

        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


def estimate_context_usage(
    system_prompt: str,
    user_message: str,
    max_context: int = 8192,
    encoding: str = "hivecoder"
) -> Tuple[int, int, int]:
    """
    Estimate context window usage for a chat completion.

    Args:
        system_prompt: System prompt
        user_message: User message
        max_context: Maximum context window
        encoding: Encoding name

    Returns:
        Tuple of (total_tokens, remaining_tokens, percentage_used)
    """
    system_tokens = count_tokens(system_prompt, encoding)
    user_tokens = count_tokens(user_message, encoding)

    # Add ~10 tokens for chat formatting overhead
    total = system_tokens + user_tokens + 10
    remaining = max_context - total
    percentage = int((total / max_context) * 100)

    return total, remaining, percentage


# Convenience aliases
count = count_tokens
enc = encode
dec = decode
