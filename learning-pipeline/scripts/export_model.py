#!/usr/bin/env python3
"""
Model Export Script
Merges LoRA adapters with base model and exports for deployment

Supports:
- HuggingFace format (safetensors)
- GGUF format (for llama.cpp)
- Quantization (Q4_K_M, Q5_K_M, Q8_0, etc.)
"""

import os
import subprocess
import shutil
import torch
import argparse
import logging
from pathlib import Path
from typing import Optional, List

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# GGUF conversion tools
CONVERT_HF_TO_GGUF = "/home/linuxbrew/.linuxbrew/bin/convert_hf_to_gguf.py"
LLAMA_QUANTIZE = "/usr/local/bin/llama-quantize"

# Available quantization types
QUANT_TYPES = [
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0", "F16", "F32"
]

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

    def convert_to_gguf(self, hf_path: Path, outtype: str = "f16") -> Path:
        """
        Convert HuggingFace model to GGUF format

        Args:
            hf_path: Path to HuggingFace model directory
            outtype: Output type (f16, f32, bf16, q8_0, auto)

        Returns:
            Path to GGUF file
        """
        if not Path(CONVERT_HF_TO_GGUF).exists():
            raise FileNotFoundError(f"GGUF converter not found: {CONVERT_HF_TO_GGUF}")

        gguf_output = self.output_path / f"model-{outtype}.gguf"

        logger.info(f"Converting to GGUF ({outtype})...")
        logger.info(f"  Input: {hf_path}")
        logger.info(f"  Output: {gguf_output}")

        cmd = [
            "python3", CONVERT_HF_TO_GGUF,
            str(hf_path),
            "--outfile", str(gguf_output),
            "--outtype", outtype
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"GGUF conversion failed: {result.stderr}")
            raise RuntimeError(f"GGUF conversion failed: {result.stderr}")

        logger.info(f"✅ GGUF created: {gguf_output}")
        logger.info(f"   Size: {gguf_output.stat().st_size / 1e9:.2f} GB")

        return gguf_output

    def quantize_gguf(self, gguf_path: Path, quant_type: str = "Q4_K_M") -> Path:
        """
        Quantize GGUF model

        Args:
            gguf_path: Path to input GGUF file
            quant_type: Quantization type (Q4_K_M, Q5_K_M, Q8_0, etc.)

        Returns:
            Path to quantized GGUF file
        """
        if not Path(LLAMA_QUANTIZE).exists():
            raise FileNotFoundError(f"llama-quantize not found: {LLAMA_QUANTIZE}")

        if quant_type not in QUANT_TYPES:
            raise ValueError(f"Invalid quant_type: {quant_type}. Must be one of: {QUANT_TYPES}")

        quant_output = self.output_path / f"model-{quant_type.lower()}.gguf"

        logger.info(f"Quantizing to {quant_type}...")
        logger.info(f"  Input: {gguf_path}")
        logger.info(f"  Output: {quant_output}")

        cmd = [LLAMA_QUANTIZE, str(gguf_path), str(quant_output), quant_type]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Quantization failed: {result.stderr}")
            raise RuntimeError(f"Quantization failed: {result.stderr}")

        original_size = gguf_path.stat().st_size
        quant_size = quant_output.stat().st_size
        compression = (1 - quant_size / original_size) * 100

        logger.info(f"✅ Quantized: {quant_output}")
        logger.info(f"   Size: {quant_size / 1e9:.2f} GB ({compression:.1f}% smaller)")

        return quant_output

    def export(self, formats: list = None, quant_types: list = None):
        """
        Export model in specified formats

        Args:
            formats: List of formats ['hf', 'safetensors', 'gguf']
            quant_types: List of quantization types for GGUF ['Q4_K_M', 'Q5_K_M', etc.]
        """
        formats = formats or ['hf']
        quant_types = quant_types or []

        # Load and merge
        self.load_and_merge()

        # Export to requested formats
        outputs = {}

        if 'hf' in formats or 'gguf' in formats:
            # HF format is required for GGUF conversion
            outputs['hf'] = self.save_hf_format()

        if 'safetensors' in formats:
            outputs['safetensors'] = self.save_safetensors()

        if 'gguf' in formats:
            # Convert to GGUF (F16 base)
            hf_path = outputs.get('hf', self.output_path / 'hf')
            gguf_path = self.convert_to_gguf(hf_path, outtype="f16")
            outputs['gguf'] = gguf_path

            # Apply quantizations if requested
            if quant_types:
                outputs['quantized'] = {}
                for qtype in quant_types:
                    try:
                        quant_path = self.quantize_gguf(gguf_path, qtype)
                        outputs['quantized'][qtype] = quant_path
                    except Exception as e:
                        logger.error(f"Failed to quantize to {qtype}: {e}")

        logger.info("Export complete!")
        return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Export merged model (LoRA + Base → HF/GGUF)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to HuggingFace format only
  python export_model.py --lora-path models/lora --base-model Qwen/Qwen2.5-0.5B --output models/merged

  # Export to GGUF with Q4_K_M quantization
  python export_model.py --lora-path models/lora --base-model Qwen/Qwen2.5-0.5B --output models/merged \\
      --formats hf gguf --quant Q4_K_M

  # Export with multiple quantizations
  python export_model.py --lora-path models/lora --base-model Qwen/Qwen2.5-0.5B --output models/merged \\
      --formats gguf --quant Q4_K_M Q5_K_M Q8_0
        """
    )
    parser.add_argument('--lora-path', type=str, required=True, help='Path to LoRA adapters')
    parser.add_argument('--base-model', type=str, required=True, help='Base model name or path')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--formats', type=str, nargs='+', default=['hf'],
                        choices=['hf', 'safetensors', 'gguf'],
                        help='Export formats (default: hf)')
    parser.add_argument('--quant', type=str, nargs='+', default=[],
                        choices=QUANT_TYPES,
                        help=f'Quantization types for GGUF. Options: {", ".join(QUANT_TYPES)}')
    parser.add_argument('--device-map', type=str, default='auto', help='Device mapping')

    args = parser.parse_args()

    # Validate: quant requires gguf format
    if args.quant and 'gguf' not in args.formats:
        logger.warning("--quant specified but 'gguf' not in formats. Adding 'gguf' to formats.")
        args.formats.append('gguf')

    try:
        exporter = ModelExporter(
            lora_path=args.lora_path,
            base_model=args.base_model,
            output_path=args.output,
            device_map=args.device_map
        )

        outputs = exporter.export(formats=args.formats, quant_types=args.quant)

        logger.info("=" * 60)
        logger.info("✅ All exports complete!")
        logger.info("=" * 60)
        for format_name, path in outputs.items():
            if format_name == 'quantized':
                for qtype, qpath in path.items():
                    logger.info(f"  {qtype}: {qpath}")
            else:
                logger.info(f"  {format_name}: {path}")

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
