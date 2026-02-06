#!/usr/bin/env python3
"""
Benchmark script for LoRA training on Hive-Mind
Tracks performance metrics and compares with baselines
"""

import argparse
import json
import time
import torch
from pathlib import Path
from datetime import datetime
import subprocess

def get_gpu_memory():
    """Get current GPU memory usage"""
    try:
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                "allocated_gb": round(mem_allocated, 2),
                "reserved_gb": round(mem_reserved, 2)
            }
    except:
        pass
    return None

def get_system_info():
    """Collect system information"""
    import sys
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_count"] = torch.cuda.device_count()

        # Try to get VRAM info from rocm-smi
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "VRAM Total Memory" in line:
                        # Extract memory in GB
                        mem_bytes = int(line.split(':')[-1].strip())
                        info["vram_total_gb"] = round(mem_bytes / 1024**3, 2)
        except:
            pass

    return info

def run_benchmark(model_name, dataset_path, output_dir, **train_args):
    """Run training benchmark"""

    print("=" * 80)
    print("🔥 Hive-Mind LoRA Training Benchmark")
    print("=" * 80)

    # System info
    sys_info = get_system_info()
    print(f"\n📊 System Info:")
    print(f"  Python: {sys_info['python_version']}")
    print(f"  PyTorch: {sys_info['pytorch_version']}")
    print(f"  ROCm: {sys_info['rocm_version']}")
    print(f"  Device: {sys_info.get('device_name', 'N/A')}")
    print(f"  VRAM: {sys_info.get('vram_total_gb', 'N/A')} GB")

    # Training config
    print(f"\n⚙️  Training Config:")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_path}")
    print(f"  LoRA r: {train_args.get('lora_r', 8)}")
    print(f"  LoRA alpha: {train_args.get('lora_alpha', 16)}")
    print(f"  Batch size: {train_args.get('batch_size', 2)}")
    print(f"  Grad accum: {train_args.get('grad_accum', 4)}")
    print(f"  Epochs: {train_args.get('epochs', 3)}")
    print(f"  Learning rate: {train_args.get('lr', 2e-4)}")

    # Initial memory
    initial_mem = get_gpu_memory()
    if initial_mem:
        print(f"\n💾 Initial GPU Memory:")
        print(f"  Allocated: {initial_mem['allocated_gb']} GB")
        print(f"  Reserved: {initial_mem['reserved_gb']} GB")

    # Build command
    cmd = [
        "python3", "scripts/train_lora.py",
        "--model", model_name,
        "--dataset", dataset_path,
        "--output", output_dir,
        "--lora-r", str(train_args.get('lora_r', 8)),
        "--lora-alpha", str(train_args.get('lora_alpha', 16)),
        "--batch-size", str(train_args.get('batch_size', 2)),
        "--grad-accum", str(train_args.get('grad_accum', 4)),
        "--epochs", str(train_args.get('epochs', 3)),
        "--lr", str(train_args.get('lr', 2e-4)),
    ]

    print(f"\n🚀 Starting training...")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 80)

    # Run training with timing
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()

    training_time = end_time - start_time

    # Parse output for metrics
    output_lines = result.stdout.split('\n')
    metrics = {
        "training_time_seconds": round(training_time, 2),
        "training_time_minutes": round(training_time / 60, 2),
        "success": result.returncode == 0,
        "final_loss": None,
        "samples_per_second": None,
        "steps_per_second": None,
    }

    # Extract metrics from output
    for line in output_lines:
        if "train_samples_per_second" in line:
            try:
                metrics["samples_per_second"] = float(line.split('=')[-1].strip())
            except:
                pass
        if "train_steps_per_second" in line:
            try:
                metrics["steps_per_second"] = float(line.split('=')[-1].strip())
            except:
                pass
        if "'loss':" in line:
            try:
                metrics["final_loss"] = float(line.split("'loss':")[1].split(',')[0].strip())
            except:
                pass

    # Final memory
    final_mem = get_gpu_memory()
    if final_mem:
        metrics["peak_memory_gb"] = final_mem['allocated_gb']

    # Print results
    print("\n" + "=" * 80)
    print("✅ Benchmark Results")
    print("=" * 80)
    print(f"  Status: {'SUCCESS' if metrics['success'] else 'FAILED'}")
    print(f"  Training time: {metrics['training_time_minutes']} minutes ({metrics['training_time_seconds']}s)")
    if metrics['samples_per_second']:
        print(f"  Samples/sec: {metrics['samples_per_second']:.3f}")
    if metrics['steps_per_second']:
        print(f"  Steps/sec: {metrics['steps_per_second']:.3f}")
    if metrics['final_loss']:
        print(f"  Final loss: {metrics['final_loss']:.4f}")
    if metrics.get('peak_memory_gb'):
        print(f"  Peak memory: {metrics['peak_memory_gb']:.2f} GB")
    print("=" * 80)

    # Save results
    benchmark_data = {
        "system_info": sys_info,
        "training_config": {
            "model": model_name,
            "dataset": str(dataset_path),
            **train_args
        },
        "metrics": metrics,
        "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
        "stderr": result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr,
    }

    results_file = Path("benchmarks") / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_file.parent.mkdir(exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n📁 Results saved to: {results_file}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LoRA training")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Model to train")
    parser.add_argument("--dataset", default="data/training_data_small.jsonl", help="Training dataset")
    parser.add_argument("--output", default="models/benchmark-run", help="Output directory")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        epochs=args.epochs,
        lr=args.lr,
    )
