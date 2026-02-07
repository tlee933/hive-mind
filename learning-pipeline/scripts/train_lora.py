#!/usr/bin/env python3
"""
LoRA Fine-tuning Script
Fine-tunes LLM using LoRA on collected interaction data
"""

# Suppress bitsandbytes ROCm version warnings (we don't use quantization)
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='bitsandbytes')

import os
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_optimal_batch_size(overhead_percent: float = 0.20, min_batch: int = 1, max_batch: int = 32) -> int:
    """
    Calculate optimal batch size based on available VRAM.

    Args:
        overhead_percent: Reserve this much VRAM as overhead (default 20%)
        min_batch: Minimum batch size to return
        max_batch: Maximum batch size to return

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using minimum batch size")
        return min_batch

    try:
        # Get GPU memory info
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        free_memory = total_memory - allocated_memory

        # Calculate usable memory (free minus overhead)
        usable_memory = free_memory * (1 - overhead_percent)

        # Estimate memory per sample (empirical: ~40MB per sample for Qwen-0.5B with BF16)
        # This is a conservative estimate based on model size and precision
        memory_per_sample = 40 * 1024 * 1024  # 40 MB in bytes

        # Calculate batch size
        estimated_batch = int(usable_memory / memory_per_sample)

        # Clamp to min/max range
        optimal_batch = max(min_batch, min(estimated_batch, max_batch))

        # Log memory stats
        logger.info(f"GPU Memory: Total={total_memory / 1e9:.2f}GB, "
                   f"Allocated={allocated_memory / 1e9:.2f}GB, "
                   f"Free={free_memory / 1e9:.2f}GB")
        logger.info(f"Usable (with {overhead_percent*100}% overhead): {usable_memory / 1e9:.2f}GB")
        logger.info(f"Calculated optimal batch size: {optimal_batch}")

        return optimal_batch

    except Exception as e:
        logger.warning(f"Error calculating optimal batch size: {e}. Using minimum: {min_batch}")
        return min_batch


class LoRATrainer:
    """Manages LoRA fine-tuning"""

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
    ):
        """Initialize LoRA trainer"""
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LoRA config
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        # Training config
        self.training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=False,  # Use bf16 for ROCm
            bf16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            optim="adamw_torch",
            report_to="tensorboard",
            logging_dir=str(self.output_dir / "logs"),
            remove_unused_columns=False,  # Keep 'text' column for on-the-fly tokenization
        )

        logger.info(f"Initialized trainer for model: {model_name}")
        logger.info(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        logger.info(f"Training: lr={learning_rate}, epochs={num_epochs}, batch={batch_size}")

    def load_model_and_tokenizer(self):
        """Load base model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare for LoRA (skip kbit prep since we're using BF16, not quantized)
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        self.model = get_peft_model(self.model, self.lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        logger.info("Model loaded successfully")

    def load_dataset_file(self):
        """Load and prepare dataset"""
        logger.info(f"Loading dataset from: {self.dataset_path}")

        # Load JSONL dataset
        dataset = load_dataset('json', data_files=str(self.dataset_path), split='train')

        logger.info(f"Loaded {len(dataset)} examples")

        # Format examples
        def format_instruction(example):
            """Format example as instruction-following for system interactions"""
            user_request = example['user_request']
            command = example.get('command', '')
            output = example.get('output', '')
            tool = example.get('tool', 'bash')

            # Format as a system interaction
            prompt = f"### User Request:\n{user_request}\n\n### Tool: {tool}\n### Command:\n{command}\n\n### Output:\n"
            full_text = prompt + output

            return {'text': full_text}

        # Just format text, don't tokenize yet (will tokenize on-the-fly)
        self.train_dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

        logger.info(f"Dataset prepared: {len(self.train_dataset)} formatted examples (tokenization on-the-fly)")

    def train(self):
        """Run training"""
        logger.info("Starting training...")

        # Custom collator that tokenizes on-the-fly
        def collate_fn(examples):
            texts = [ex['text'] for ex in examples]
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors='pt'
            )
            tokenized['labels'] = tokenized['input_ids'].clone()
            return tokenized

        data_collator = collate_fn

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
        )

        # Train
        train_result = trainer.train()

        # Save model
        logger.info("Saving model...")
        trainer.save_model()

        # Save metrics
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)

        logger.info(f"✅ Training complete! Model saved to {self.output_dir}")

        return train_result.metrics

    def run(self):
        """Run full training pipeline"""
        start_time = datetime.now()

        try:
            self.load_model_and_tokenizer()
            self.load_dataset_file()
            metrics = self.train()

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Training completed in {elapsed:.1f}s")

            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning')
    parser.add_argument('--model', type=str, required=True, help='Base model name or path')
    parser.add_argument('--dataset', type=str, required=True, help='Training dataset (JSONL)')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--lora-r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--batch-size', type=str, default='auto',
                       help='Batch size (integer or "auto" to calculate based on free VRAM)')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--vram-overhead', type=float, default=0.20,
                       help='VRAM overhead percentage to reserve (default 0.20 = 20%%)')

    args = parser.parse_args()

    # Calculate batch size if auto
    if args.batch_size == 'auto':
        batch_size = get_optimal_batch_size(overhead_percent=args.vram_overhead)
        logger.info(f"Auto-calculated batch size: {batch_size}")
    else:
        try:
            batch_size = int(args.batch_size)
        except ValueError:
            logger.error(f"Invalid batch-size: {args.batch_size}. Use integer or 'auto'")
            return

    trainer = LoRATrainer(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )

    metrics = trainer.run()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
