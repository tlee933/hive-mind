# Hive-Mind Learning Pipeline Makefile
# Simplifies common operations

.PHONY: help build test run collect train export clean logs shell

# Colors
BLUE := \033[0;34m
GREEN := \033[0;32m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Hive-Mind Learning Pipeline$(NC)"
	@echo "============================"
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""

build: ## Build Docker container
	@echo "$(BLUE)Building learning pipeline container...$(NC)"
	docker-compose build
	@echo "$(GREEN)✅ Build complete$(NC)"

test: ## Run validation tests
	@echo "$(BLUE)Running pipeline tests...$(NC)"
	docker-compose run --rm learning-pipeline bash scripts/test_pipeline.sh

run: ## Run full pipeline (collect + train + export)
	@echo "$(BLUE)Running full learning pipeline...$(NC)"
	docker-compose run --rm learning-pipeline bash scripts/pipeline.sh run

collect: ## Collect training data from Redis
	@echo "$(BLUE)Collecting training data...$(NC)"
	docker-compose run --rm learning-pipeline bash scripts/pipeline.sh collect

train: ## Train LoRA adapter
	@echo "$(BLUE)Training model...$(NC)"
	docker-compose run --rm learning-pipeline bash scripts/pipeline.sh train

export: ## Export merged model
	@echo "$(BLUE)Exporting model...$(NC)"
	docker-compose run --rm learning-pipeline bash scripts/pipeline.sh export

clean: ## Clean generated files
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	rm -rf data/training_data_*.jsonl data/metadata_*.json
	rm -rf models/lora_* models/latest_model.txt
	@echo "$(GREEN)✅ Cleanup complete$(NC)"

logs: ## View TensorBoard logs
	@echo "$(BLUE)Starting TensorBoard...$(NC)"
	@echo "View at http://localhost:6006"
	docker-compose run --rm -p 6006:6006 learning-pipeline \
		tensorboard --logdir /workspace/models --host 0.0.0.0

shell: ## Open interactive shell in container
	@echo "$(BLUE)Opening shell in learning pipeline container...$(NC)"
	docker-compose run --rm learning-pipeline bash

quick-train: ## Quick test training run (1 epoch, small dataset)
	@echo "$(BLUE)Running quick training test...$(NC)"
	docker-compose run --rm -e EPOCHS=1 -e MAX_ITEMS=10 learning-pipeline \
		bash scripts/pipeline.sh run

gpu-check: ## Check GPU availability
	@echo "$(BLUE)Checking GPU...$(NC)"
	docker-compose run --rm learning-pipeline rocm-smi

config-check: ## Validate configuration
	@echo "$(BLUE)Checking configuration...$(NC)"
	docker-compose run --rm learning-pipeline \
		python3 -c "import yaml; print(yaml.safe_load(open('/workspace/config.yaml')))"

stop: ## Stop running containers
	@echo "$(BLUE)Stopping containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Stopped$(NC)"

rebuild: clean build ## Clean and rebuild everything
	@echo "$(GREEN)✅ Rebuild complete$(NC)"

.DEFAULT_GOAL := help
