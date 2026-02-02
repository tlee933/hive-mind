#!/usr/bin/env python3
"""
LoRA Fine-tuning Script
Fine-tunes LLM using LoRA on collected interaction data
"""

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
    prepare_model_for_kbit_training,
    TaskType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

        # Prepare for LoRA
        self.model = prepare_model_for_kbit_training(self.model)
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
            """Format example as instruction-following"""
            instruction = example['instruction']
            input_text = example.get('input', '')
            output = example['output']

            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

            full_text = prompt + output

            return {'text': full_text}

        dataset = dataset.map(format_instruction)

        # Tokenize
        def tokenize(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=512,
                padding='max_length',
            )

        self.train_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )

        logger.info(f"Dataset prepared: {len(self.train_dataset)} tokenized examples")

    def train(self):
        """Run training"""
        logger.info("Starting training...")

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

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
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--grad-accum', type=int, default=4, help='Gradient accumulation steps')

    args = parser.parse_args()

    trainer = LoRATrainer(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
    )

    metrics = trainer.run()
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
