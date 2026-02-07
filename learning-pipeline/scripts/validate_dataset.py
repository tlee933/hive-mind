#!/usr/bin/env python3
"""
Dataset Validation Script
Identifies and fixes problematic examples that may cause training crashes
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate and clean training datasets"""

    def __init__(self, input_file: str, max_seq_length: int = 2048):
        self.input_file = Path(input_file)
        self.max_seq_length = max_seq_length
        self.issues = defaultdict(list)
        self.stats = {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'truncated': 0,
            'cleaned': 0
        }

    def validate_json(self, line: str, line_num: int) -> Tuple[bool, Dict]:
        """Validate JSON parsing"""
        try:
            data = json.loads(line.strip())
            return True, data
        except json.JSONDecodeError as e:
            self.issues['json_errors'].append({
                'line': line_num,
                'error': str(e),
                'content': line[:100]
            })
            return False, {}

    def validate_schema(self, data: Dict, line_num: int) -> bool:
        """Validate required fields exist"""
        required_fields = ['user_request', 'tool', 'command', 'output', 'timestamp']

        for field in required_fields:
            if field not in data:
                self.issues['missing_fields'].append({
                    'line': line_num,
                    'missing': field,
                    'available': list(data.keys())
                })
                return False

        return True

    def check_sequence_length(self, data: Dict, line_num: int) -> Tuple[bool, Dict]:
        """Check if text is too long and truncate if needed"""
        # Rough estimate: 1 token ≈ 4 characters
        chars_per_token = 4
        max_chars = self.max_seq_length * chars_per_token

        truncated = False

        # Check and truncate each text field
        for field in ['user_request', 'command', 'output']:
            if field in data and isinstance(data[field], str):
                if len(data[field]) > max_chars:
                    self.issues['long_sequences'].append({
                        'line': line_num,
                        'field': field,
                        'length': len(data[field]),
                        'max': max_chars
                    })
                    data[field] = data[field][:max_chars]
                    truncated = True

        return truncated, data

    def check_special_characters(self, data: Dict, line_num: int) -> bool:
        """Check for problematic special characters"""
        problematic_chars = ['\x00', '\ufffd', '\ufeff']  # Null, replacement char, BOM

        for field in ['user_request', 'command', 'output']:
            if field in data and isinstance(data[field], str):
                for char in problematic_chars:
                    if char in data[field]:
                        self.issues['special_chars'].append({
                            'line': line_num,
                            'field': field,
                            'char': repr(char)
                        })
                        # Remove problematic characters
                        data[field] = data[field].replace(char, '')

        return True

    def validate_and_clean(self) -> Tuple[List[Dict], Dict]:
        """Validate entire dataset and return cleaned version"""
        logger.info(f"Validating dataset: {self.input_file}")

        valid_examples = []

        with open(self.input_file, 'r', encoding='utf-8', errors='replace') as f:
            for line_num, line in enumerate(f, 1):
                self.stats['total'] += 1

                # Skip empty lines
                if not line.strip():
                    continue

                # Validate JSON
                is_valid_json, data = self.validate_json(line, line_num)
                if not is_valid_json:
                    self.stats['invalid'] += 1
                    continue

                # Validate schema
                if not self.validate_schema(data, line_num):
                    self.stats['invalid'] += 1
                    continue

                # Check sequence lengths
                was_truncated, data = self.check_sequence_length(data, line_num)
                if was_truncated:
                    self.stats['truncated'] += 1

                # Check special characters
                self.check_special_characters(data, line_num)

                # Add to valid examples
                valid_examples.append(data)
                self.stats['valid'] += 1
                self.stats['cleaned'] += 1

        logger.info(f"Validation complete: {self.stats['valid']}/{self.stats['total']} valid")

        return valid_examples, self.stats

    def save_cleaned_dataset(self, examples: List[Dict], output_file: str):
        """Save cleaned dataset"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        logger.info(f"Cleaned dataset saved to: {output_path}")

    def generate_report(self, report_file: str):
        """Generate validation report"""
        report_path = Path(report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("Dataset Validation Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("STATISTICS\n")
            f.write("-" * 80 + "\n")
            for key, value in self.stats.items():
                f.write(f"{key:20s}: {value:,}\n")

            f.write("\n")
            f.write("ISSUES FOUND\n")
            f.write("-" * 80 + "\n")

            if not any(self.issues.values()):
                f.write("No issues found! Dataset is clean.\n")
            else:
                for issue_type, occurrences in self.issues.items():
                    f.write(f"\n{issue_type.upper()} ({len(occurrences)} occurrences):\n")
                    for i, issue in enumerate(occurrences[:10], 1):  # Show first 10
                        f.write(f"  {i}. {issue}\n")
                    if len(occurrences) > 10:
                        f.write(f"  ... and {len(occurrences) - 10} more\n")

            f.write("\n" + "=" * 80 + "\n")

        logger.info(f"Validation report saved to: {report_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate and clean training dataset')
    parser.add_argument('--input', type=str, required=True,
                       help='Input dataset file (JSONL)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output cleaned dataset file (JSONL)')
    parser.add_argument('--report', type=str, default='logs/dataset_validation_report.txt',
                       help='Validation report output file')
    parser.add_argument('--max-seq-length', type=int, default=2048,
                       help='Maximum sequence length in tokens (default: 2048)')

    args = parser.parse_args()

    # Validate
    validator = DatasetValidator(args.input, max_seq_length=args.max_seq_length)
    cleaned_examples, stats = validator.validate_and_clean()

    # Save cleaned dataset
    validator.save_cleaned_dataset(cleaned_examples, args.output)

    # Generate report
    validator.generate_report(args.report)

    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total examples: {stats['total']:,}")
    print(f"Valid examples: {stats['valid']:,}")
    print(f"Invalid examples: {stats['invalid']:,}")
    print(f"Truncated examples: {stats['truncated']:,}")
    print(f"\nCleaned dataset: {args.output}")
    print(f"Validation report: {args.report}")
    print("=" * 80)

    # Exit with error code if too many invalid examples
    invalid_rate = stats['invalid'] / stats['total'] if stats['total'] > 0 else 0
    if invalid_rate > 0.05:  # More than 5% invalid
        logger.error(f"High invalid rate: {invalid_rate:.1%}")
        sys.exit(1)


if __name__ == '__main__':
    main()
