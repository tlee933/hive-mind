"""
Custom tiktoken encodings for HiveCoder and local models.

Usage:
    import tiktoken
    enc = tiktoken.get_encoding("hivecoder")
    tokens = enc.encode("Hello, world!")
"""

from tiktoken.load import load_tiktoken_bpe

# Special tokens for HiveCoder / local models
ENDOFTEXT = "<|endoftext|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
PAD = "<|pad|>"


def hivecoder():
    """
    HiveCoder encoding - based on cl100k_base (GPT-4 tokenizer).

    Customize this for your local model's vocabulary if needed.
    For now, uses cl100k_base as the base with custom special tokens.
    """
    # Load cl100k_base as the base (same as GPT-4)
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    )

    # Pattern for tokenization (same as cl100k)
    pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    # Special tokens - can customize for your model
    special_tokens = {
        ENDOFTEXT: 100257,
        SYSTEM: 100258,
        USER: 100259,
        ASSISTANT: 100260,
        PAD: 100261,
    }

    return {
        "name": "hivecoder",
        "pat_str": pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


def qwen_base():
    """
    Qwen model encoding - placeholder.

    Qwen uses a different tokenizer (SentencePiece-based).
    This is a placeholder that uses cl100k_base.
    For true Qwen tokenization, use transformers AutoTokenizer.
    """
    mergeable_ranks = load_tiktoken_bpe(
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    )

    pat_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    return {
        "name": "qwen_base",
        "pat_str": pat_str,
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": {ENDOFTEXT: 100257},
    }


# Registry of encoding constructors - tiktoken discovers these via entry_points
ENCODING_CONSTRUCTORS = {
    "hivecoder": hivecoder,
    "qwen_base": qwen_base,
}
