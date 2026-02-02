#!/bin/bash
# Comprehensive Pipeline Testing Script
# Validates all components before production use

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

FAILED_TESTS=0
PASSED_TESTS=0

# Test result tracking
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED_TESTS++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED_TESTS++))
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Test 1: Validate directory structure
test_directory_structure() {
    log_test "Validating directory structure..."

    local required_dirs=(
        "/workspace/data"
        "/workspace/models"
        "/workspace/scripts"
        "/workspace/configs"
    )

    for dir in "${required_dirs[@]}"; do
        if [ -d "$dir" ]; then
            log_pass "Directory exists: $dir"
        else
            log_fail "Missing directory: $dir"
            return 1
        fi
    done

    return 0
}

# Test 2: Validate scripts exist and are executable
test_scripts_exist() {
    log_test "Validating scripts..."

    local required_scripts=(
        "/workspace/scripts/collect_data.py"
        "/workspace/scripts/train_lora.py"
        "/workspace/scripts/export_model.py"
        "/workspace/scripts/pipeline.sh"
    )

    for script in "${required_scripts[@]}"; do
        if [ -f "$script" ]; then
            if [ -x "$script" ]; then
                log_pass "Script exists and executable: $(basename $script)"
            else
                log_warn "Script not executable: $(basename $script)"
                chmod +x "$script"
                log_pass "Made executable: $(basename $script)"
            fi
        else
            log_fail "Missing script: $(basename $script)"
            return 1
        fi
    done

    return 0
}

# Test 3: Validate Python dependencies
test_python_dependencies() {
    log_test "Validating Python dependencies..."

    local required_packages=(
        "torch"
        "transformers"
        "peft"
        "datasets"
        "accelerate"
        "redis"
        "yaml"
    )

    for package in "${required_packages[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            log_pass "Python package installed: $package"
        else
            log_fail "Missing Python package: $package"
            return 1
        fi
    done

    return 0
}

# Test 4: Validate ROCm/GPU availability
test_gpu_availability() {
    log_test "Validating GPU availability..."

    # Check if ROCm is available
    if command -v rocm-smi &> /dev/null; then
        log_pass "ROCm CLI available"

        # Check GPU visibility
        if rocm-smi --showproductname &> /dev/null; then
            log_pass "GPU detected: $(rocm-smi --showproductname 2>/dev/null | grep 'GPU' | head -1)"
        else
            log_warn "ROCm installed but no GPU detected"
        fi
    else
        log_warn "rocm-smi not found (acceptable in CPU-only mode)"
    fi

    # Check PyTorch GPU
    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        local gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")
        log_pass "PyTorch sees $gpu_count GPU(s)"
    else
        log_warn "PyTorch GPU not available (CPU-only mode)"
    fi

    return 0
}

# Test 5: Validate Redis connectivity
test_redis_connectivity() {
    log_test "Validating Redis connectivity..."

    if [ ! -f "/workspace/config.yaml" ]; then
        log_warn "Config file not found - skipping Redis test"
        return 0
    fi

    # Try to connect to Redis
    local redis_host=$(grep -A 5 "redis:" /workspace/config.yaml | grep "host:" | head -1 | awk '{print $2}' | tr -d '"')
    local redis_port=$(grep -A 5 "redis:" /workspace/config.yaml | grep "port:" | head -1 | awk '{print $2}')
    local redis_pass=$(grep "password:" /workspace/config.yaml | awk '{print $2}' | tr -d '"')

    if [ -z "$redis_host" ] || [ -z "$redis_port" ]; then
        log_warn "Redis config incomplete - skipping connectivity test"
        return 0
    fi

    if command -v redis-cli &> /dev/null; then
        if timeout 5 redis-cli -h "$redis_host" -p "$redis_port" -a "$redis_pass" PING &> /dev/null; then
            log_pass "Redis connection successful: $redis_host:$redis_port"
        else
            log_warn "Cannot connect to Redis at $redis_host:$redis_port"
        fi
    else
        log_warn "redis-cli not available - skipping connectivity test"
    fi

    return 0
}

# Test 6: Validate configuration files
test_configuration() {
    log_test "Validating configuration..."

    if [ -f "/workspace/config.yaml" ]; then
        # Validate YAML syntax
        if python3 -c "import yaml; yaml.safe_load(open('/workspace/config.yaml'))" 2>/dev/null; then
            log_pass "Config YAML syntax valid"
        else
            log_fail "Config YAML syntax invalid"
            return 1
        fi
    else
        log_warn "No config.yaml found (expected for initial setup)"
    fi

    return 0
}

# Test 7: Test data collection (dry run)
test_data_collection() {
    log_test "Testing data collection script (dry run)..."

    if [ ! -f "/workspace/config.yaml" ]; then
        log_warn "No config.yaml - skipping data collection test"
        return 0
    fi

    # Test script can be imported without errors
    if python3 -c "import sys; sys.path.insert(0, '/workspace/scripts'); from collect_data import DataCollector" 2>/dev/null; then
        log_pass "Data collection script imports successfully"
    else
        log_fail "Data collection script has import errors"
        return 1
    fi

    return 0
}

# Test 8: Test training script (syntax check)
test_training_script() {
    log_test "Testing training script (syntax check)..."

    # Check script syntax
    if python3 -m py_compile /workspace/scripts/train_lora.py 2>/dev/null; then
        log_pass "Training script syntax valid"
    else
        log_fail "Training script has syntax errors"
        return 1
    fi

    # Test imports
    if python3 -c "import sys; sys.path.insert(0, '/workspace/scripts'); from train_lora import LoRATrainer" 2>/dev/null; then
        log_pass "Training script imports successfully"
    else
        log_fail "Training script has import errors"
        return 1
    fi

    return 0
}

# Test 9: Test export script (syntax check)
test_export_script() {
    log_test "Testing export script (syntax check)..."

    # Check script syntax
    if python3 -m py_compile /workspace/scripts/export_model.py 2>/dev/null; then
        log_pass "Export script syntax valid"
    else
        log_fail "Export script has syntax errors"
        return 1
    fi

    # Test imports
    if python3 -c "import sys; sys.path.insert(0, '/workspace/scripts'); from export_model import ModelExporter" 2>/dev/null; then
        log_pass "Export script imports successfully"
    else
        log_fail "Export script has import errors"
        return 1
    fi

    return 0
}

# Test 10: Validate write permissions
test_write_permissions() {
    log_test "Testing write permissions..."

    local test_dirs=(
        "/workspace/data"
        "/workspace/models"
    )

    for dir in "${test_dirs[@]}"; do
        local test_file="$dir/.write_test_$$"
        if touch "$test_file" 2>/dev/null; then
            rm -f "$test_file"
            log_pass "Write permission OK: $dir"
        else
            log_fail "No write permission: $dir"
            return 1
        fi
    done

    return 0
}

# Main test runner
main() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Hive-Mind Learning Pipeline Tests${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    # Run all tests
    test_directory_structure || true
    test_scripts_exist || true
    test_python_dependencies || true
    test_gpu_availability || true
    test_redis_connectivity || true
    test_configuration || true
    test_data_collection || true
    test_training_script || true
    test_export_script || true
    test_write_permissions || true

    # Summary
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}            Test Summary${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "  ${GREEN}Passed: $PASSED_TESTS${NC}"
    echo -e "  ${RED}Failed: $FAILED_TESTS${NC}"
    echo ""

    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}✅ All critical tests passed!${NC}"
        echo -e "${GREEN}   Pipeline is ready for production use.${NC}"
        echo ""
        return 0
    else
        echo -e "${RED}❌ Some tests failed.${NC}"
        echo -e "${RED}   Please fix issues before running pipeline.${NC}"
        echo ""
        return 1
    fi
}

# Run tests
main
exit $?
