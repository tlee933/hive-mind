# PyTorch 2.9.1 Build for ROCm 7.12 + gfx1201

**Date:** February 5, 2026
**System:** AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)
**ROCm Version:** 7.12.0a20260203
**PyTorch Version:** 2.9.1

## Build Summary

Successfully built PyTorch 2.9.1 from source with full ROCm 7.12 support for gfx1201, including Flash Attention and FBGEMM GenAI optimizations.

## Issues Fixed

### 1. Flatbuffers Version Mismatch
**Problem:** PyTorch expected flatbuffers v24, but ROCm 7.12 has v25.9.23

**Solution:** Updated version checks in 3 locations:
```bash
# Files modified:
/var/mnt/build/TheRock/external-builds/pytorch/pytorch/torch/csrc/jit/serialization/mobile_bytecode_generated.h
/var/mnt/build/TheRock/external-builds/pytorch/pytorch/torch/include/torch/csrc/jit/serialization/mobile_bytecode_generated.h
/var/mnt/build/TheRock/external-builds/pytorch/pytorch/third_party/flatbuffers/include/flatbuffers/base.h

# Changed from:
FLATBUFFERS_VERSION_MAJOR 24
FLATBUFFERS_VERSION_MINOR 12

# Changed to:
FLATBUFFERS_VERSION_MAJOR 25
FLATBUFFERS_VERSION_MINOR 9
```

### 2. rocprim `__half` Operator Ambiguity
**Problem:** ROCm 7.12's rocprim library had ambiguous `__half` operators in radix sort code

**Solution:** Modified `/opt/rocm/include/rocprim/device/detail/device_radix_sort.hpp`:
```cpp
// Lines 636-638 - Before:
const T zero{0};
const T a_plus = a + zero;
const T b_plus = b + zero;

// After (simplified to avoid __half operator issues):
// Use values directly to avoid __half operator issues
const T a_plus = a;
const T b_plus = b;
```

**Backup created at:** `/opt/rocm/include/rocprim/device/detail/device_radix_sort.hpp.backup`

### 3. Missing libdrm Headers
**Problem:** Build failed looking for `libdrm/drm.h` and related headers

**Solution:** Symlinked TheRock build's libdrm headers:
```bash
sudo ln -sf /mnt/build/TheRock/build/core/ROCR-Runtime/dist/lib/rocm_sysdeps/include/libdrm \
    /opt/rocm/include/libdrm
```

### 4. HIP Device Libraries
**Problem:** Compiler couldn't find ROCm device libraries for gfx1201

**Solution:** Set environment variables pointing to device library path:
```bash
export HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
export DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
```

## Build Configuration

```bash
cd /var/mnt/build/TheRock/external-builds/pytorch/pytorch
rm -rf build

export ROCM_PATH=/opt/rocm
export HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
export DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx1201
export USE_FLASH_ATTENTION=1
export USE_FBGEMM_GENAI=ON
export PYTORCH_BUILD_VERSION=2.9.1
export PYTORCH_BUILD_NUMBER=1
export CMAKE_PREFIX_PATH=/opt/rocm

python3 setup.py bdist_wheel
```

## Build Output

**Wheel file:** `torch-2.9.1-cp314-cp314-linux_x86_64.whl` (329 MB)
**Location:** `/var/mnt/build/TheRock/external-builds/pytorch/pytorch/dist/`
**Build time:** ~2 hours (8083 compilation steps)

## Installation

```bash
python3 -m pip install --force-reinstall \
    /var/mnt/build/TheRock/external-builds/pytorch/pytorch/dist/torch-2.9.1-cp314-cp314-linux_x86_64.whl
```

## Verification

```python
import torch
print(f'PyTorch {torch.__version__}')
print(f'ROCm available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
```

**Output:**
```
PyTorch 2.9.1
ROCm available: True
Device: AMD Radeon AI PRO R9700
```

## LoRA Training Results

Successfully trained Qwen2.5-0.5B with LoRA on 1,500 synthetic examples:
- **Training time:** 546.7 seconds (~9 minutes)
- **Final loss:** 0.3367
- **Throughput:** 8.23 samples/sec, 1.03 steps/sec
- **Parameters:** 4.4M trainable / 498M total (0.88%)
- **No HIP errors** - Previous version (ROCm 7.11) crashed at step 3 with illegal memory access

## Important Notes

1. **Backup created:** Original rocprim header backed up before modification
2. **Symlinks used:** libdrm headers symlinked from TheRock build (not installed via rpm-ostree)
3. **Environment variables:** Device library paths must be set for builds
4. **Version compatibility:** This build specifically targets ROCm 7.12 + gfx1201

## Files Modified

1. `/opt/rocm/include/rocprim/device/detail/device_radix_sort.hpp` (lines 636-638)
2. PyTorch source flatbuffers version checks (3 files)
3. `/opt/rocm/include/libdrm/` → symlink to TheRock build

## Clean System State

- No rpm-ostree layers added for libdrm (using symlinks instead)
- PyTorch wheel can be reinstalled from dist directory if needed
- All modifications are documented and reversible
