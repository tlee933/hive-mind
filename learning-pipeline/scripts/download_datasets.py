#!/usr/bin/env python3
"""
Download and prepare external datasets for Hive-Mind training
"""

import argparse
import json
import logging
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and format datasets for Hive-Mind"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_code_alpaca(self, max_examples: int = 5000):
        """Download Code Alpaca dataset - perfect for code tasks"""
        logger.info("Downloading Code Alpaca dataset...")

        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

        output_file = self.output_dir / "code_alpaca.jsonl"
        count = 0

        with open(output_file, 'w') as f:
            for example in tqdm(dataset.select(range(min(max_examples, len(dataset)))),
                              desc="Processing Code Alpaca"):
                # Format for our training pipeline
                formatted = {
                    "user_request": example['instruction'],
                    "tool": "code_generation",
                    "command": example.get('input', ''),
                    "output": example['output'],
                    "timestamp": "2026-02-06T00:00:00Z",
                    "metadata": {"source": "code_alpaca"}
                }
                f.write(json.dumps(formatted) + '\n')
                count += 1

        logger.info(f"✅ Downloaded {count} Code Alpaca examples to {output_file}")
        return count

    def download_bash_commands(self, max_examples: int = 5000):
        """Download NL2Bash dataset - natural language to bash"""
        logger.info("Downloading NL2Bash dataset...")

        try:
            dataset = load_dataset("neulab/docprompting-conala", split="train")
        except:
            # Fallback: use a similar dataset
            logger.warning("NL2Bash not available, using alternative...")
            dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")

        output_file = self.output_dir / "bash_commands.jsonl"
        count = 0

        with open(output_file, 'w') as f:
            for i, example in enumerate(tqdm(dataset, desc="Processing Bash")):
                if i >= max_examples:
                    break

                # Try to extract bash-related examples
                instruction = example.get('instruction', example.get('nl', ''))
                output = example.get('output', example.get('cmd', ''))

                if any(keyword in instruction.lower() for keyword in
                      ['bash', 'command', 'shell', 'linux', 'file', 'directory']):
                    formatted = {
                        "user_request": instruction,
                        "tool": "bash",
                        "command": instruction,
                        "output": output,
                        "timestamp": "2026-02-06T00:00:00Z",
                        "metadata": {"source": "bash_dataset"}
                    }
                    f.write(json.dumps(formatted) + '\n')
                    count += 1

        logger.info(f"✅ Downloaded {count} Bash examples to {output_file}")
        return count

    def download_glaive_tools(self, max_examples: int = 5000):
        """Download Glaive tool-calling dataset"""
        logger.info("Downloading Glaive tool-calling dataset...")

        dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")

        output_file = self.output_dir / "glaive_tools.jsonl"
        count = 0

        with open(output_file, 'w') as f:
            for example in tqdm(dataset.select(range(min(max_examples, len(dataset)))),
                              desc="Processing Glaive"):
                # Parse the conversation format
                system = example.get('system', '')
                chat = example.get('chat', '')

                # Extract user message and assistant response from text format
                # Format: "USER: <msg>\n\n\nA: <response>"
                try:
                    if not isinstance(chat, str) or not chat:
                        continue

                    # Split on "USER:" and "A:" or "ASSISTANT:"
                    if 'USER:' in chat:
                        parts = chat.split('USER:', 1)
                        if len(parts) > 1:
                            rest = parts[1]
                            # Find assistant response
                            for separator in ['\n\n\nA:', '\nA:', 'ASSISTANT:', '\n\nA:']:
                                if separator in rest:
                                    user_msg, asst_msg = rest.split(separator, 1)
                                    user_msg = user_msg.strip()
                                    asst_msg = asst_msg.strip()

                                    # Remove <|endoftext|> markers
                                    asst_msg = asst_msg.replace('<|endoftext|>', '').strip()

                                    if user_msg and asst_msg:
                                        formatted = {
                                            "user_request": user_msg,
                                            "tool": "function_calling",
                                            "command": user_msg,
                                            "output": asst_msg,
                                            "timestamp": "2026-02-06T00:00:00Z",
                                            "metadata": {"source": "glaive_tools", "system": system[:200]}
                                        }
                                        f.write(json.dumps(formatted) + '\n')
                                        count += 1
                                        break
                except Exception as e:
                    continue

        logger.info(f"✅ Downloaded {count} Glaive examples to {output_file}")
        return count

    def create_merged_dataset(self):
        """Merge all downloaded datasets into one file"""
        logger.info("Merging all datasets...")

        merged_file = self.output_dir / "merged_dataset.jsonl"
        total = 0

        with open(merged_file, 'w') as outf:
            for dataset_file in self.output_dir.glob("*.jsonl"):
                if dataset_file.name == "merged_dataset.jsonl":
                    continue

                logger.info(f"Adding {dataset_file.name}...")
                with open(dataset_file) as inf:
                    for line in inf:
                        outf.write(line)
                        total += 1

        logger.info(f"✅ Merged dataset created: {merged_file} ({total} examples)")
        return merged_file, total


def main():
    parser = argparse.ArgumentParser(description='Download datasets for Hive-Mind')
    parser.add_argument('--output', type=str, default='data/external',
                       help='Output directory for datasets')
    parser.add_argument('--max-per-dataset', type=int, default=5000,
                       help='Max examples per dataset')
    parser.add_argument('--datasets', nargs='+',
                       choices=['code_alpaca', 'bash', 'glaive', 'all'],
                       default=['all'],
                       help='Which datasets to download')

    args = parser.parse_args()

    downloader = DatasetDownloader(args.output)

    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['code_alpaca', 'bash', 'glaive']

    total_downloaded = 0

    if 'code_alpaca' in datasets_to_download:
        total_downloaded += downloader.download_code_alpaca(args.max_per_dataset)

    if 'bash' in datasets_to_download:
        total_downloaded += downloader.download_bash_commands(args.max_per_dataset)

    if 'glaive' in datasets_to_download:
        total_downloaded += downloader.download_glaive_tools(args.max_per_dataset)

    # Create merged dataset
    merged_file, merged_total = downloader.create_merged_dataset()

    print("\n" + "="*80)
    print("✅ DATASET DOWNLOAD COMPLETE")
    print("="*80)
    print(f"Total examples: {total_downloaded}")
    print(f"Merged file: {merged_file}")
    print(f"Merged total: {merged_total}")
    print("\nNext steps:")
    print(f"1. Review: head -5 {merged_file}")
    print(f"2. Train: python3 scripts/train_lora.py --dataset {merged_file}")
    print("="*80)


if __name__ == '__main__':
    main()
