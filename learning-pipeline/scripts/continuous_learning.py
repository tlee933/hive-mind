#!/usr/bin/env python3
"""
Continuous Learning System for HiveCoder
Automatically collects, filters, trains, and deploys model updates
"""

import os
import sys
import json
import yaml
import time
import shutil
import hashlib
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging

from redis.cluster import RedisCluster, ClusterNode
import redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Tracks a model version"""
    version: str
    created_at: str
    base_model: str
    training_samples: int
    final_loss: float
    gguf_path: str
    lora_path: str
    status: str  # 'training', 'ready', 'deployed', 'retired'
    eval_score: Optional[float] = None
    deployed_at: Optional[str] = None


class ModelRegistry:
    """Manages model versions and deployments"""

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_file = registry_path / "registry.json"
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self._load()

    def _load(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                data = json.load(f)
                for v in data.get('versions', []):
                    self.versions[v['version']] = ModelVersion(**v)
            logger.info(f"Loaded {len(self.versions)} model versions from registry")

    def _save(self):
        """Save registry to disk"""
        data = {
            'versions': [asdict(v) for v in self.versions.values()],
            'updated_at': datetime.now().isoformat()
        }
        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_version(self, version: ModelVersion):
        """Add a new model version"""
        self.versions[version.version] = version
        self._save()
        logger.info(f"Registered model version: {version.version}")

    def get_deployed(self) -> Optional[ModelVersion]:
        """Get currently deployed version"""
        for v in self.versions.values():
            if v.status == 'deployed':
                return v
        return None

    def get_latest_ready(self) -> Optional[ModelVersion]:
        """Get latest ready-to-deploy version"""
        ready = [v for v in self.versions.values() if v.status == 'ready']
        if ready:
            return max(ready, key=lambda x: x.created_at)
        return None

    def set_deployed(self, version: str):
        """Mark a version as deployed, retire old one"""
        old_deployed = self.get_deployed()
        if old_deployed:
            old_deployed.status = 'retired'

        if version in self.versions:
            self.versions[version].status = 'deployed'
            self.versions[version].deployed_at = datetime.now().isoformat()
            self._save()
            logger.info(f"Deployed model version: {version}")


class QualityFilter:
    """Filters interactions for training quality"""

    MIN_OUTPUT_LENGTH = 10  # Minimum output length
    MAX_OUTPUT_LENGTH = 10000  # Maximum (avoid huge outputs)

    QUALITY_TOOLS = {
        'llm_generate', 'llm_code_assist', 'llm_complete',
        'memory_store', 'tool_result', 'bash', 'python'
    }

    @classmethod
    def filter(cls, interactions: List[Dict]) -> List[Dict]:
        """Filter interactions for quality"""
        filtered = []

        for item in interactions:
            # Must be successful
            if not item.get('success', True):
                continue

            # Check output length
            output = item.get('output', '') or item.get('result', '')
            if len(output) < cls.MIN_OUTPUT_LENGTH:
                continue
            if len(output) > cls.MAX_OUTPUT_LENGTH:
                continue

            # Prefer certain tools
            tool = item.get('tool', '') or item.get('tool_used', '')
            # For now accept all tools, but weight quality ones higher

            filtered.append(item)

        logger.info(f"Quality filter: {len(interactions)} → {len(filtered)} interactions")
        return filtered


class ContinuousLearner:
    """Main continuous learning orchestrator"""

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.base_dir = Path(config_path).parent
        self.models_dir = self.base_dir / "learning-pipeline" / "models"
        self.data_dir = self.base_dir / "learning-pipeline" / "data" / "continuous"
        self.registry = ModelRegistry(self.models_dir / "registry")

        # Training thresholds
        self.min_samples_for_training = 50  # Minimum new samples before retraining
        self.max_hours_between_training = 24  # Force retrain after this many hours

        # Connect to Redis
        self._connect_redis()

        # Track last collection
        self.last_collection_file = self.data_dir / ".last_collection"
        self.collected_ids_file = self.data_dir / ".collected_ids"

        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _connect_redis(self):
        """Connect to Redis cluster"""
        redis_config = self.config['redis']

        if redis_config.get('cluster_mode', False):
            startup_nodes = [
                ClusterNode(node['host'], node['port'])
                for node in redis_config['nodes']
            ]
            self.redis = RedisCluster(
                startup_nodes=startup_nodes,
                password=redis_config['password'],
                decode_responses=True,
            )
            logger.info("Connected to Redis Cluster")
        else:
            self.redis = redis.Redis(
                host=redis_config['nodes'][0]['host'],
                port=redis_config['nodes'][0]['port'],
                password=redis_config['password'],
                decode_responses=True
            )

    def _get_collected_ids(self) -> set:
        """Get IDs of already collected interactions"""
        if self.collected_ids_file.exists():
            with open(self.collected_ids_file, 'r') as f:
                return set(line.strip() for line in f)
        return set()

    def _add_collected_ids(self, ids: List[str]):
        """Add collected IDs to tracking file"""
        with open(self.collected_ids_file, 'a') as f:
            for id in ids:
                f.write(id + '\n')

    def collect_new_interactions(self) -> List[Dict]:
        """Collect new interactions from Redis queue"""
        queue_name = self.config['learning']['queue_name']
        collected_ids = self._get_collected_ids()

        try:
            # Read all entries from stream
            entries = self.redis.xread({queue_name: '0'}, count=10000)

            if not entries:
                return []

            new_interactions = []
            new_ids = []

            for stream_name, messages in entries:
                for msg_id, msg_data in messages:
                    if msg_id in collected_ids:
                        continue

                    interaction = {
                        'id': msg_id,
                        'timestamp': msg_data.get('timestamp', ''),
                        'tool': msg_data.get('tool_used') or msg_data.get('tool', ''),
                        'input': msg_data.get('user_query') or msg_data.get('input', ''),
                        'output': msg_data.get('result') or msg_data.get('output', ''),
                        'success': str(msg_data.get('success', 'true')).lower() in ('true', '1', 'yes'),
                        'session_id': msg_data.get('session_id', ''),
                    }
                    new_interactions.append(interaction)
                    new_ids.append(msg_id)

            if new_ids:
                self._add_collected_ids(new_ids)
                logger.info(f"Collected {len(new_interactions)} new interactions")

            return new_interactions

        except Exception as e:
            logger.error(f"Error collecting interactions: {e}")
            return []

    def format_training_data(self, interactions: List[Dict]) -> List[Dict]:
        """Format interactions into training examples"""
        training_data = []

        for item in interactions:
            if not item.get('success', True):
                continue

            output = item.get('output', '')
            if not output:
                continue

            # Format as instruction-following
            example = {
                'user_request': item.get('input', f"Execute {item.get('tool', 'command')}"),
                'tool': item.get('tool', 'unknown'),
                'command': item.get('input', ''),
                'output': output,
                'timestamp': item.get('timestamp', ''),
            }
            training_data.append(example)

        return training_data

    def save_training_batch(self, data: List[Dict]) -> Path:
        """Save a training batch and return path"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_file = self.data_dir / f"batch_{timestamp}.jsonl"

        with open(batch_file, 'w') as f:
            for example in data:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Saved {len(data)} examples to {batch_file}")
        return batch_file

    def merge_training_data(self) -> Path:
        """Merge all batch files into one training file"""
        merged_file = self.data_dir / "merged_training.jsonl"
        all_examples = []

        # Load all batch files
        for batch_file in sorted(self.data_dir.glob("batch_*.jsonl")):
            with open(batch_file, 'r') as f:
                for line in f:
                    all_examples.append(json.loads(line))

        # Deduplicate by hashing content
        seen = set()
        unique_examples = []
        for ex in all_examples:
            h = hashlib.md5(json.dumps(ex, sort_keys=True).encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_examples.append(ex)

        # Save merged file
        with open(merged_file, 'w') as f:
            for ex in unique_examples:
                f.write(json.dumps(ex) + '\n')

        logger.info(f"Merged {len(unique_examples)} unique examples into {merged_file}")
        return merged_file

    def should_train(self) -> bool:
        """Check if we should trigger training"""
        # Count available training samples
        total_samples = 0
        for batch_file in self.data_dir.glob("batch_*.jsonl"):
            with open(batch_file, 'r') as f:
                total_samples += sum(1 for _ in f)

        if total_samples >= self.min_samples_for_training:
            logger.info(f"Training threshold met: {total_samples} >= {self.min_samples_for_training}")
            return True

        # Check time since last training
        last_version = self.registry.get_deployed()
        if last_version:
            last_time = datetime.fromisoformat(last_version.created_at)
            hours_since = (datetime.now() - last_time).total_seconds() / 3600
            if hours_since > self.max_hours_between_training and total_samples > 0:
                logger.info(f"Time threshold met: {hours_since:.1f}h > {self.max_hours_between_training}h")
                return True

        logger.info(f"Training not needed: {total_samples} samples (need {self.min_samples_for_training})")
        return False

    def train_new_version(self) -> Optional[ModelVersion]:
        """Train a new model version"""
        # Merge all training data
        training_file = self.merge_training_data()

        # Count samples
        with open(training_file, 'r') as f:
            num_samples = sum(1 for _ in f)

        if num_samples < 10:
            logger.warning(f"Not enough samples for training: {num_samples}")
            return None

        # Generate version ID
        version_id = datetime.now().strftime("v%Y%m%d_%H%M%S")
        version_dir = self.models_dir / "continuous" / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Get base model (use the foundation model or latest deployed)
        base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        deployed = self.registry.get_deployed()

        logger.info(f"Starting training for version {version_id}")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  Training samples: {num_samples}")

        # Create version record
        version = ModelVersion(
            version=version_id,
            created_at=datetime.now().isoformat(),
            base_model=base_model,
            training_samples=num_samples,
            final_loss=0.0,
            gguf_path="",
            lora_path=str(version_dir / "lora"),
            status='training',
        )
        self.registry.add_version(version)

        # Run training
        train_script = self.base_dir / "learning-pipeline" / "scripts" / "train_lora.py"

        cmd = [
            sys.executable, str(train_script),
            "--model", base_model,
            "--dataset", str(training_file),
            "--output", str(version_dir / "lora"),
            "--epochs", "1",  # Quick incremental training
            "--batch-size", "auto",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
                cwd=str(self.base_dir)
            )

            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                version.status = 'failed'
                self.registry._save()
                return None

            # Parse metrics from output
            try:
                metrics = json.loads(result.stdout.strip().split('\n')[-1])
                version.final_loss = metrics.get('train_loss', 0.0)
            except:
                pass

            version.status = 'ready'
            self.registry._save()

            logger.info(f"Training complete: {version_id} (loss: {version.final_loss:.4f})")
            return version

        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            version.status = 'failed'
            self.registry._save()
            return None
        except Exception as e:
            logger.error(f"Training error: {e}")
            version.status = 'failed'
            self.registry._save()
            return None

    def export_to_gguf(self, version: ModelVersion) -> bool:
        """Export model to GGUF format"""
        export_script = self.base_dir / "learning-pipeline" / "scripts" / "export_model.py"
        version_dir = Path(version.lora_path).parent

        logger.info(f"Exporting {version.version} to GGUF...")

        cmd = [
            sys.executable, str(export_script),
            "--base-model", version.base_model,
            "--lora-path", version.lora_path,
            "--output", str(version_dir / "export"),
            "--quant", "Q5_K_M",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=str(self.base_dir)
            )

            if result.returncode != 0:
                logger.error(f"Export failed: {result.stderr}")
                return False

            # Find quantized GGUF file (prefer Q5_K_M over f16)
            export_dir = version_dir / "export"
            q5_file = export_dir / "model-q5_k_m.gguf"
            f16_file = export_dir / "model-f16.gguf"

            if q5_file.exists():
                version.gguf_path = str(q5_file)
            elif f16_file.exists():
                version.gguf_path = str(f16_file)
            else:
                gguf_files = list(export_dir.glob("*.gguf"))
                if gguf_files:
                    version.gguf_path = str(gguf_files[0])

            if version.gguf_path:
                self.registry._save()
                logger.info(f"Exported to: {version.gguf_path}")
                return True

            return False

        except Exception as e:
            logger.error(f"Export error: {e}")
            return False

    def deploy_version(self, version: ModelVersion) -> bool:
        """Deploy a model version (hot-swap)"""
        if not version.gguf_path or not Path(version.gguf_path).exists():
            logger.error("No GGUF file to deploy")
            return False

        # Update symlink for llama-server
        models_dir = self.models_dir / "foundation_7b_export"
        current_link = models_dir / "HiveCoder-7B-current.gguf"

        # Create symlink to new version
        if current_link.exists():
            current_link.unlink()

        current_link.symlink_to(version.gguf_path)

        # Restart llama-server
        logger.info("Restarting llama-server with new model...")
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "hivecoder-llm"],
                check=True,
                timeout=60
            )

            # Wait for service to be ready
            time.sleep(10)

            # Health check
            import requests
            resp = requests.get("http://localhost:8089/health", timeout=10)
            if resp.status_code == 200:
                self.registry.set_deployed(version.version)
                logger.info(f"✅ Deployed version {version.version}")
                return True

        except Exception as e:
            logger.error(f"Deployment failed: {e}")

        return False

    def run_once(self):
        """Run one iteration of continuous learning"""
        logger.info("=" * 60)
        logger.info("Continuous Learning Cycle")
        logger.info("=" * 60)

        # 1. Collect new interactions
        interactions = self.collect_new_interactions()

        if interactions:
            # 2. Filter for quality
            filtered = QualityFilter.filter(interactions)

            if filtered:
                # 3. Format and save
                training_data = self.format_training_data(filtered)
                if training_data:
                    self.save_training_batch(training_data)

        # 4. Check if we should train
        if self.should_train():
            version = self.train_new_version()

            if version and version.status == 'ready':
                # 5. Export to GGUF
                if self.export_to_gguf(version):
                    # 6. Deploy
                    self.deploy_version(version)

        logger.info("Cycle complete")

    def run_daemon(self, interval: int = 300):
        """Run as daemon, checking periodically"""
        logger.info(f"Starting continuous learning daemon (interval: {interval}s)")

        while True:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Error in learning cycle: {e}")

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description='Continuous Learning System')
    parser.add_argument('--config', type=str,
                       default='/var/mnt/build/MCP/hive-mind/config.yaml',
                       help='Path to config.yaml')
    parser.add_argument('--daemon', action='store_true',
                       help='Run as daemon')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (daemon mode)')
    parser.add_argument('--train-now', action='store_true',
                       help='Force training immediately')
    parser.add_argument('--collect-only', action='store_true',
                       help='Only collect data, do not train')
    parser.add_argument('--status', action='store_true',
                       help='Show status and exit')

    args = parser.parse_args()

    learner = ContinuousLearner(args.config)

    if args.status:
        # Show status
        print("\n📊 Continuous Learning Status")
        print("=" * 40)

        # Count pending samples
        total_samples = 0
        for batch_file in learner.data_dir.glob("batch_*.jsonl"):
            with open(batch_file, 'r') as f:
                total_samples += sum(1 for _ in f)

        print(f"Pending samples: {total_samples}")
        print(f"Training threshold: {learner.min_samples_for_training}")

        deployed = learner.registry.get_deployed()
        if deployed:
            print(f"\nDeployed version: {deployed.version}")
            print(f"  Created: {deployed.created_at}")
            print(f"  Samples: {deployed.training_samples}")
            print(f"  Loss: {deployed.final_loss:.4f}")

        print(f"\nTotal versions: {len(learner.registry.versions)}")

    elif args.train_now:
        # Force training
        learner.min_samples_for_training = 1  # Lower threshold
        learner.run_once()

    elif args.collect_only:
        # Just collect data
        interactions = learner.collect_new_interactions()
        if interactions:
            filtered = QualityFilter.filter(interactions)
            if filtered:
                training_data = learner.format_training_data(filtered)
                if training_data:
                    learner.save_training_batch(training_data)

    elif args.daemon:
        learner.run_daemon(args.interval)

    else:
        learner.run_once()


if __name__ == '__main__':
    main()
