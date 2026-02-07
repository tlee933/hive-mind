#!/usr/bin/env python3
"""
Auto-Optimizer for Hive-Mind Learning Pipeline
Intelligently selects best configuration based on model, data, and hardware.
"""

import os
import torch
import argparse
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class QualityMode(Enum):
    FAST = "fast"           # Prioritize speed, accept quality trade-offs
    BALANCED = "balanced"   # Balance between speed and quality
    BEST = "best"           # Prioritize quality, accept slower speed


class TaskType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    EXPORT = "export"


@dataclass
class HardwareProfile:
    """Detected hardware capabilities"""
    vram_total_gb: float
    vram_free_gb: float
    gpu_name: str
    gpu_arch: str
    supports_bf16: bool
    supports_flash_attn: bool

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Auto-detect hardware capabilities"""
        if not torch.cuda.is_available():
            return cls(
                vram_total_gb=0, vram_free_gb=0,
                gpu_name="CPU", gpu_arch="cpu",
                supports_bf16=False, supports_flash_attn=False
            )

        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1e9
        free = (props.total_memory - torch.cuda.memory_allocated()) / 1e9

        # Detect architecture
        gpu_arch = os.environ.get("PYTORCH_ROCM_ARCH", "")
        if not gpu_arch:
            gpu_arch = os.environ.get("HSA_OVERRIDE_GFX_VERSION", "")
            if gpu_arch:
                gpu_arch = f"gfx{gpu_arch.replace('.', '')}"
        if not gpu_arch:
            # Try to detect from GPU name
            if "R9700" in props.name or "9700" in props.name:
                gpu_arch = "gfx1201"
            elif "7900" in props.name:
                gpu_arch = "gfx1100"
            else:
                gpu_arch = "unknown"

        # gfx1201 (RDNA4) capabilities
        is_rdna4 = "gfx1201" in gpu_arch or "RDNA4" in props.name.upper() or "R9700" in props.name
        is_rdna3 = "gfx1100" in gpu_arch or "gfx1101" in gpu_arch or "7900" in props.name

        return cls(
            vram_total_gb=total,
            vram_free_gb=free,
            gpu_name=props.name,
            gpu_arch=gpu_arch,
            supports_bf16=is_rdna4 or is_rdna3 or props.major >= 8,
            supports_flash_attn=is_rdna4  # Flash Attention works on gfx1201
        )


@dataclass
class ModelProfile:
    """Model characteristics"""
    name: str
    param_count_b: float  # Billions
    is_instruct: bool

    @classmethod
    def from_name(cls, model_name: str) -> "ModelProfile":
        """Parse model profile from name"""
        name_lower = model_name.lower()

        # Estimate param count from name
        param_count = 0.5  # default
        for size in ["0.5b", "1b", "3b", "7b", "8b", "13b", "14b", "32b", "70b", "72b"]:
            if size in name_lower:
                param_count = float(size.replace("b", ""))
                break

        is_instruct = "instruct" in name_lower or "chat" in name_lower

        return cls(name=model_name, param_count_b=param_count, is_instruct=is_instruct)


@dataclass
class DataProfile:
    """Dataset characteristics"""
    sample_count: int
    avg_length: int  # tokens

    @classmethod
    def from_file(cls, path: Path) -> "DataProfile":
        """Analyze dataset file"""
        if not path.exists():
            return cls(sample_count=0, avg_length=256)

        with open(path) as f:
            lines = f.readlines()

        sample_count = len(lines)
        # Rough estimate: 4 chars per token
        avg_length = sum(len(l) for l in lines[:100]) // (4 * min(100, len(lines)))

        return cls(sample_count=sample_count, avg_length=avg_length)


@dataclass
class OptimizationConfig:
    """Recommended configuration"""
    # Training
    lora_r: int
    lora_alpha: int
    batch_size: int
    grad_accum: int
    precision: str  # "bf16", "fp16", "fp32"
    use_flash_attn: bool
    use_gradient_checkpointing: bool

    # Inference/Export
    quant_type: str  # "none", "int8", "int4", "Q4_K_M", etc.

    # Reasoning
    reasoning: Dict[str, str]


class AutoOptimizer:
    """Smart optimizer that selects best configuration"""

    def __init__(self, quality_mode: QualityMode = QualityMode.BALANCED):
        self.quality_mode = quality_mode
        self.hardware = HardwareProfile.detect()

    def optimize_training(
        self,
        model_name: str,
        dataset_path: Optional[Path] = None
    ) -> OptimizationConfig:
        """Recommend optimal training configuration"""

        model = ModelProfile.from_name(model_name)
        data = DataProfile.from_file(dataset_path) if dataset_path else DataProfile(1000, 256)
        hw = self.hardware

        reasoning = {}

        # LoRA rank selection
        if model.param_count_b <= 1:
            lora_r, lora_alpha = 8, 16
            reasoning["lora"] = f"Small model ({model.param_count_b}B) → r=8"
        elif model.param_count_b <= 8:
            lora_r, lora_alpha = 16, 32
            reasoning["lora"] = f"Medium model ({model.param_count_b}B) → r=16"
        else:
            lora_r, lora_alpha = 32, 64
            reasoning["lora"] = f"Large model ({model.param_count_b}B) → r=32"

        # Batch size based on VRAM and model size
        vram_per_sample_gb = model.param_count_b * 0.5  # Rough estimate
        max_batch = int(hw.vram_free_gb * 0.7 / vram_per_sample_gb)
        max_batch = max(1, min(max_batch, 32))

        if self.quality_mode == QualityMode.FAST:
            batch_size = max_batch
            grad_accum = 2
        elif self.quality_mode == QualityMode.BEST:
            batch_size = max(1, max_batch // 2)
            grad_accum = 8
        else:  # BALANCED
            batch_size = max(1, max_batch // 2)
            grad_accum = 4

        reasoning["batch"] = f"VRAM {hw.vram_free_gb:.1f}GB → batch={batch_size}, accum={grad_accum}"

        # Precision
        if hw.supports_bf16:
            precision = "bf16"
            reasoning["precision"] = "GPU supports BF16 → using BF16"
        else:
            precision = "fp16"
            reasoning["precision"] = "Fallback to FP16"

        # Flash Attention
        use_flash = hw.supports_flash_attn and model.param_count_b >= 3
        reasoning["flash_attn"] = f"Flash Attention: {'enabled' if use_flash else 'disabled'}"

        # Gradient checkpointing for larger models
        use_gc = model.param_count_b >= 7
        reasoning["grad_ckpt"] = f"Gradient checkpointing: {'enabled' if use_gc else 'disabled'}"

        return OptimizationConfig(
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            batch_size=batch_size,
            grad_accum=grad_accum,
            precision=precision,
            use_flash_attn=use_flash,
            use_gradient_checkpointing=use_gc,
            quant_type="none",
            reasoning=reasoning
        )

    def optimize_inference(self, model_name: str) -> OptimizationConfig:
        """Recommend optimal inference configuration"""

        model = ModelProfile.from_name(model_name)
        hw = self.hardware

        reasoning = {}

        # Quantization selection based on model size and VRAM
        model_size_gb = model.param_count_b * 2  # BF16 estimate

        if model_size_gb <= hw.vram_free_gb * 0.8:
            # Model fits comfortably
            if self.quality_mode == QualityMode.BEST:
                quant = "none"
                reasoning["quant"] = f"Model fits ({model_size_gb:.1f}GB), best quality → no quant"
            else:
                quant = "int8"
                reasoning["quant"] = f"Model fits, balanced → int8 for speed"
        elif model_size_gb * 0.5 <= hw.vram_free_gb:
            # Needs int8
            quant = "int8"
            reasoning["quant"] = f"Tight fit → int8 quantization"
        elif model_size_gb * 0.25 <= hw.vram_free_gb:
            # Needs int4
            quant = "int4"
            reasoning["quant"] = f"Limited VRAM → int4 quantization"
        else:
            quant = "int4"
            reasoning["quant"] = f"Model too large, int4 required"

        return OptimizationConfig(
            lora_r=0, lora_alpha=0,
            batch_size=1, grad_accum=1,
            precision="bf16" if hw.supports_bf16 else "fp16",
            use_flash_attn=hw.supports_flash_attn,
            use_gradient_checkpointing=False,
            quant_type=quant,
            reasoning=reasoning
        )

    def optimize_export(self, model_name: str, target: str = "local") -> OptimizationConfig:
        """Recommend optimal export configuration"""

        model = ModelProfile.from_name(model_name)

        reasoning = {}

        # GGUF quant selection
        if self.quality_mode == QualityMode.BEST:
            quant = "Q8_0"
            reasoning["gguf"] = "Best quality → Q8_0"
        elif self.quality_mode == QualityMode.FAST:
            quant = "Q4_K_M"
            reasoning["gguf"] = "Fast/small → Q4_K_M"
        else:
            quant = "Q5_K_M"
            reasoning["gguf"] = "Balanced → Q5_K_M"

        # Adjust for very large models
        if model.param_count_b >= 30:
            quant = "Q4_K_M"
            reasoning["gguf"] = f"Large model ({model.param_count_b}B) → Q4_K_M for size"

        return OptimizationConfig(
            lora_r=0, lora_alpha=0,
            batch_size=1, grad_accum=1,
            precision="bf16",
            use_flash_attn=False,
            use_gradient_checkpointing=False,
            quant_type=quant,
            reasoning=reasoning
        )


def main():
    parser = argparse.ArgumentParser(description="Auto-optimize configuration")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--task", choices=["training", "inference", "export"], default="training")
    parser.add_argument("--quality", choices=["fast", "balanced", "best"], default="balanced")
    parser.add_argument("--dataset", type=Path, help="Dataset path (for training)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    optimizer = AutoOptimizer(quality_mode=QualityMode(args.quality))

    # Detect hardware
    logger.info("=" * 60)
    logger.info("🧠 Hive-Mind Auto-Optimizer")
    logger.info("=" * 60)
    logger.info(f"\n📊 Hardware Detected:")
    logger.info(f"   GPU: {optimizer.hardware.gpu_name}")
    logger.info(f"   VRAM: {optimizer.hardware.vram_total_gb:.1f} GB ({optimizer.hardware.vram_free_gb:.1f} GB free)")
    logger.info(f"   Arch: {optimizer.hardware.gpu_arch}")
    logger.info(f"   BF16: {'✅' if optimizer.hardware.supports_bf16 else '❌'}")
    logger.info(f"   Flash Attn: {'✅' if optimizer.hardware.supports_flash_attn else '❌'}")

    # Get recommendation
    if args.task == "training":
        config = optimizer.optimize_training(args.model, args.dataset)
    elif args.task == "inference":
        config = optimizer.optimize_inference(args.model)
    else:
        config = optimizer.optimize_export(args.model)

    logger.info(f"\n🎯 Recommended Config for {args.task.upper()} ({args.quality}):")
    logger.info("-" * 40)

    if args.task == "training":
        logger.info(f"   LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
        logger.info(f"   Batch: {config.batch_size} × {config.grad_accum} = {config.batch_size * config.grad_accum} effective")
        logger.info(f"   Precision: {config.precision}")
        logger.info(f"   Flash Attention: {'✅' if config.use_flash_attn else '❌'}")
        logger.info(f"   Grad Checkpointing: {'✅' if config.use_gradient_checkpointing else '❌'}")
    else:
        logger.info(f"   Quantization: {config.quant_type}")
        logger.info(f"   Precision: {config.precision}")

    logger.info(f"\n💡 Reasoning:")
    for key, reason in config.reasoning.items():
        logger.info(f"   • {reason}")

    logger.info("=" * 60)

    if args.json:
        import json
        print(json.dumps({
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "batch_size": config.batch_size,
            "grad_accum": config.grad_accum,
            "precision": config.precision,
            "quant_type": config.quant_type,
            "use_flash_attn": config.use_flash_attn,
            "use_gradient_checkpointing": config.use_gradient_checkpointing,
        }, indent=2))


if __name__ == "__main__":
    main()
