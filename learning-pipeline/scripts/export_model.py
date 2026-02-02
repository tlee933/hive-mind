#!/usr/bin/env python3
"""
Model Export Script
Merges LoRA adapters with base model and exports for deployment
"""

import os
import torch
import argparse
import logging
from pathlib import Path
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelExporter:
    """Handles model merging and export"""

    def __init__(
        self,
        lora_path: str,
        base_model: str,
        output_path: str,
        device_map: str = "auto"
    ):
        """
        Initialize model exporter

        Args:
            lora_path: Path to LoRA adapter weights
            base_model: Base model name or path
            output_path: Output directory for merged model
            device_map: Device mapping strategy
        """
        self.lora_path = Path(lora_path)
        self.base_model = base_model
        self.output_path = Path(output_path)
        self.device_map = device_map

        if not self.lora_path.exists():
            raise ValueError(f"LoRA path does not exist: {lora_path}")

        self.output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized exporter:")
        logger.info(f"  LoRA: {self.lora_path}")
        logger.info(f"  Base: {self.base_model}")
        logger.info(f"  Output: {self.output_path}")

    def load_and_merge(self):
        """Load base model and merge with LoRA adapters"""
        logger.info("Loading base model...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )

        # Load base model
        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            trust_remote_code=True
        )

        logger.info("Loading LoRA adapters...")

        # Load and merge LoRA
        model = PeftModel.from_pretrained(
            base,
            str(self.lora_path),
            torch_dtype=torch.bfloat16
        )

        logger.info("Merging weights...")
        merged = model.merge_and_unload()

        self.model = merged
        logger.info("✅ Model merged successfully")

        return self.model

    def save_hf_format(self):
        """Save in HuggingFace format"""
        logger.info("Saving model in HuggingFace format...")

        hf_output = self.output_path / "hf"
        hf_output.mkdir(exist_ok=True)

        # Save model
        self.model.save_pretrained(
            str(hf_output),
            safe_serialization=True
        )

        # Save tokenizer
        self.tokenizer.save_pretrained(str(hf_output))

        logger.info(f"✅ Saved to {hf_output}")
        return hf_output

    def save_safetensors(self):
        """Save in safetensors format"""
        logger.info("Saving model in safetensors format...")

        st_output = self.output_path / "safetensors"
        st_output.mkdir(exist_ok=True)

        self.model.save_pretrained(
            str(st_output),
            safe_serialization=True,
            max_shard_size="5GB"
        )

        logger.info(f"✅ Saved to {st_output}")
        return st_output

    def export(self, formats: list = None):
        """
        Export model in specified formats

        Args:
            formats: List of formats ['hf', 'safetensors', 'gguf']
        """
        formats = formats or ['hf']

        # Load and merge
        self.load_and_merge()

        # Export to requested formats
        outputs = {}

        if 'hf' in formats:
            outputs['hf'] = self.save_hf_format()

        if 'safetensors' in formats:
            outputs['safetensors'] = self.save_safetensors()

        if 'gguf' in formats:
            logger.warning("GGUF export requires llama.cpp conversion (not implemented)")
            logger.info("To convert manually:")
            logger.info(f"  1. python llama.cpp/convert_hf_to_gguf.py {outputs.get('hf', self.output_path)}")
            logger.info(f"  2. ./llama.cpp/llama-quantize model.gguf model-q4_k_m.gguf Q4_K_M")

        logger.info("Export complete!")
        return outputs


def main():
    parser = argparse.ArgumentParser(description='Export merged model')
    parser.add_argument('--lora-path', type=str, required=True, help='Path to LoRA adapters')
    parser.add_argument('--base-model', type=str, required=True, help='Base model name or path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--formats', type=str, nargs='+', default=['hf'],
                        choices=['hf', 'safetensors', 'gguf'],
                        help='Export formats')
    parser.add_argument('--device-map', type=str, default='auto', help='Device mapping')

    args = parser.parse_args()

    try:
        exporter = ModelExporter(
            lora_path=args.lora_path,
            base_model=args.base_model,
            output_path=args.output,
            device_map=args.device_map
        )

        outputs = exporter.export(formats=args.formats)

        logger.info("✅ All exports complete!")
        for format_name, path in outputs.items():
            logger.info(f"  {format_name}: {path}")

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
