"""
Hive-Mind Python Client

Fast distributed memory and tokenization utilities.
"""

from .client import HiveMindClient
from . import tokenizer

# Re-export for backwards compatibility
__all__ = ['HiveMindClient', 'tokenizer']
