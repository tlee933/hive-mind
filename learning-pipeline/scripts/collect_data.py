#!/usr/bin/env python3
"""
Data Collection Script
Collects interactions from Redis learning queue and prepares training data
"""

import redis
from redis.cluster import RedisCluster, ClusterNode
import json
import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Collects and processes training data from Redis"""

    def __init__(self, config_path: str):
        """Initialize data collector"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Connect to Redis (cluster or standalone)
        redis_config = self.config['redis']

        if redis_config.get('cluster_mode', False):
            # Redis Cluster mode
            startup_nodes = [
                ClusterNode(node['host'], node['port'])
                for node in redis_config['nodes']
            ]
            self.redis = RedisCluster(
                startup_nodes=startup_nodes,
                password=redis_config['password'],
                decode_responses=True,
            )
            logger.info(f"Connected to Redis Cluster ({len(startup_nodes)} nodes)")
        else:
            # Standalone Redis
            self.redis = redis.Redis(
                host=redis_config['nodes'][0]['host'],
                port=redis_config['nodes'][0]['port'],
                password=redis_config['password'],
                decode_responses=True
            )
            logger.info(f"Connected to Redis at {redis_config['nodes'][0]['host']}:{redis_config['nodes'][0]['port']}")

        self.queue_name = self.config['learning']['queue_name']
        self.batch_size = self.config['learning']['batch_size']

    def collect_batch(self, max_items: int = None) -> List[Dict[str, Any]]:
        """Collect a batch of interactions from the queue"""
        max_items = max_items or self.batch_size
        interactions = []

        logger.info(f"Collecting up to {max_items} interactions from queue...")

        # Read from Redis stream
        try:
            queue_length = self.redis.xlen(self.queue_name)
            logger.info(f"Queue length: {queue_length}")

            if queue_length == 0:
                logger.warning("Queue is empty")
                return []

            # Read entries from stream
            entries = self.redis.xread({self.queue_name: '0'}, count=max_items)

            if not entries:
                logger.warning("No entries found")
                return []

            for stream_name, messages in entries:
                for msg_id, msg_data in messages:
                    try:
                        # Parse interaction data (support both old and new field names)
                        interaction = {
                            'id': msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                            'timestamp': msg_data.get('timestamp', ''),
                            'tool': msg_data.get('tool_used') or msg_data.get('tool', ''),
                            'input': msg_data.get('user_query') or msg_data.get('input', ''),
                            'output': msg_data.get('result') or msg_data.get('output', ''),
                            'success': str(msg_data.get('success', 'true')).lower() in ('true', '1', 'yes'),
                            'metadata': {
                                k: v for k, v in msg_data.items()
                                if k not in ('tool_used', 'tool', 'user_query', 'input', 'result', 'output', 'success', 'timestamp')
                            }
                        }
                        interactions.append(interaction)
                    except Exception as e:
                        logger.error(f"Error parsing message {msg_id}: {e}")

            logger.info(f"Collected {len(interactions)} interactions")
            return interactions

        except Exception as e:
            logger.error(f"Error collecting from queue: {e}")
            return []

    def format_for_training(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format interactions into training examples"""
        training_data = []

        for item in interactions:
            if not item['success']:
                continue  # Skip failed interactions

            # Format for system tool interaction training
            example = {
                'user_request': item['input'] if item['input'] else f"Execute {item['tool']}",
                'tool': item['tool'],
                'command': item['input'],  # User query becomes the command
                'output': item['output'],
                'timestamp': item['timestamp'],
                'metadata': item.get('metadata', {})
            }

            training_data.append(example)

        logger.info(f"Formatted {len(training_data)} training examples")
        return training_data

    def save_dataset(self, data: List[Dict[str, str]], output_path: Path):
        """Save training dataset to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for example in data:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Saved {len(data)} examples to {output_path}")

    def collect_and_save(self, output_dir: Path, max_items: int = None):
        """Collect data and save to output directory"""
        # Collect interactions
        interactions = self.collect_batch(max_items)

        if not interactions:
            logger.warning("No interactions collected")
            return 0

        # Format for training
        training_data = self.format_for_training(interactions)

        if not training_data:
            logger.warning("No valid training examples")
            return 0

        # Save dataset
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"training_data_{timestamp}.jsonl"
        self.save_dataset(training_data, output_file)

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'num_examples': len(training_data),
            'num_interactions': len(interactions),
            'success_rate': len(training_data) / len(interactions),
            'queue_name': self.queue_name
        }

        metadata_file = output_dir / f"metadata_{timestamp}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Collection complete: {len(training_data)} examples saved")
        return len(training_data)


def main():
    parser = argparse.ArgumentParser(description='Collect training data from Redis')
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--output', type=str, default='/workspace/data', help='Output directory')
    parser.add_argument('--max-items', type=int, default=None, help='Max items to collect')

    args = parser.parse_args()

    collector = DataCollector(args.config)
    output_dir = Path(args.output)

    num_examples = collector.collect_and_save(output_dir, args.max_items)

    if num_examples > 0:
        logger.info(f"✅ Successfully collected {num_examples} training examples")
    else:
        logger.warning("⚠️  No training data collected")


if __name__ == '__main__':
    main()
