#!/usr/bin/env bash
# audit-rocm.sh — phase 1: verify nixpkgs ROCm has gfx906 kernels and flags
# Run this from INSIDE the devshell: nix develop ./nix then ./audit-rocm.sh
#
# Outcome A: all checks pass → nixpkgs ROCm is usable, proceed to pytorch build
# Outcome B: kernel checks fail → edit flake to build rocblas/rocm-device-libs
#            from source with AMDGPU_TARGETS=gfx906 before continuing

set -euo pipefail

PASS=0
FAIL=0
WARN=0

check() {
    local label="$1"
    local cmd="$2"
    local expect="$3"  # grep pattern that should match

    printf "  %-45s " "$label"
    if output=$(eval "$cmd" 2>&1) && echo "$output" | grep -qi "$expect"; then
        echo "OK"
        (( PASS++ )) || true
    else
        echo "FAIL"
        echo "    cmd:    $cmd"
        echo "    output: $(echo "$output" | head -3)"
        (( FAIL++ )) || true
    fi
}

warn() {
    local label="$1"
    local cmd="$2"
    local expect="$3"

    printf "  %-45s " "$label"
    if output=$(eval "$cmd" 2>&1) && echo "$output" | grep -qi "$expect"; then
        echo "OK"
        (( PASS++ )) || true
    else
        echo "WARN (non-fatal)"
        echo "    cmd:    $cmd"
        echo "    output: $(echo "$output" | head -3)"
        (( WARN++ )) || true
    fi
}

echo ""
echo "=== Phase 1: ROCm audit for gfx906 ==="
echo ""

# ------------------------------------------------------------------ #
echo "-- 1. HSA runtime sees the GPU"
# ------------------------------------------------------------------ #
check "rocminfo lists gfx906" \
    "rocminfo" \
    "gfx906"

check "rocminfo sees Vega20/MI50" \
    "rocminfo" \
    "vega20\|MI50\|Radeon VII"

check "HSA_OVERRIDE_GFX_VERSION set" \
    "echo \$HSA_OVERRIDE_GFX_VERSION" \
    "9.0.6"

echo ""

# ------------------------------------------------------------------ #
echo "-- 2. rocm-device-libs has gfx906 bitcode"
# ------------------------------------------------------------------ #
DEVLIBS_PATH=$(nix path-info nixpkgs#rocmPackages.rocm-device-libs 2>/dev/null || true)
if [ -n "$DEVLIBS_PATH" ]; then
    check "device-libs has gfx906 bitcode" \
        "find $DEVLIBS_PATH -name '*gfx906*' -o -name '*906*' 2>/dev/null | head -1" \
        "gfx906\|906"
else
    echo "  rocm-device-libs path-info failed (expected inside devshell)"
    (( WARN++ )) || true
fi

echo ""

# ------------------------------------------------------------------ #
echo "-- 3. rocblas precompiled kernels for gfx906"
# ------------------------------------------------------------------ #
ROCBLAS_PATH=$(nix path-info nixpkgs#rocmPackages.rocblas 2>/dev/null || true)
if [ -n "$ROCBLAS_PATH" ]; then
    KERNEL_COUNT=$(find "$ROCBLAS_PATH" -name "*.co" 2>/dev/null | grep -c "gfx906" || echo 0)
    printf "  %-45s " "rocblas .co kernel objects for gfx906"
    if [ "$KERNEL_COUNT" -gt 0 ]; then
        echo "OK ($KERNEL_COUNT files)"
        (( PASS++ )) || true
    else
        echo "FAIL (0 files — nixpkgs rocblas was NOT built with gfx906)"
        echo "    → You need to rebuild rocblas with AMDGPU_TARGETS=gfx906"
        echo "    → See: flake.nix overrideScope section for how to do this"
        (( FAIL++ )) || true
    fi
else
    echo "  rocblas path-info failed (expected inside devshell)"
    (( WARN++ )) || true
fi

echo ""

# ------------------------------------------------------------------ #
echo "-- 4. HIP toolchain is functional"
# ------------------------------------------------------------------ #
check "hipcc is on PATH" \
    "which hipcc" \
    "hipcc"

check "clang knows amdgcn target" \
    "clang --print-targets" \
    "amdgcn"

# Quick HIP compile sanity — writes to /tmp
TMP_SRC=$(mktemp /tmp/hip_test_XXXX.cpp)
cat > "$TMP_SRC" <<'EOF'
#include <hip/hip_runtime.h>
int main() {
    int n;
    hipGetDeviceCount(&n);
    return 0;
}
EOF
printf "  %-45s " "hipcc compiles trivial HIP program"
if hipcc --offload-arch=gfx906 "$TMP_SRC" -o /tmp/hip_test_bin 2>/tmp/hip_compile_err; then
    echo "OK"
    (( PASS++ )) || true
    rm -f /tmp/hip_test_bin
else
    echo "FAIL"
    cat /tmp/hip_compile_err | head -10 | sed 's/^/    /'
    (( FAIL++ )) || true
fi
rm -f "$TMP_SRC" /tmp/hip_compile_err

echo ""

# ------------------------------------------------------------------ #
echo "-- 5. Python environment"
# ------------------------------------------------------------------ #
check "python3 on PATH" \
    "python3 --version" \
    "Python 3"

warn "numpy importable" \
    "python3 -c 'import numpy; print(numpy.__version__)'" \
    "[0-9]"

echo ""

# ------------------------------------------------------------------ #
echo "=== Summary ==="
# ------------------------------------------------------------------ #
echo "  PASS: $PASS   WARN: $WARN   FAIL: $FAIL"
echo ""

if [ "$FAIL" -eq 0 ]; then
    echo "  All hard checks passed. ROCm looks usable for gfx906."
    echo "  Proceed to: git clone pytorch, then USE_ROCM=1 python setup.py develop"
else
    echo "  $FAIL check(s) failed. Do not proceed to PyTorch build yet."
    echo ""
    echo "  If rocblas kernel check failed:"
    echo "    The nixpkgs rocblas was not built with gfx906 in AMDGPU_TARGETS."
    echo "    Option 1: add rocblas to the overrideScope in flake.nix with"
    echo "              rocblas = ps.rocblas.override { localGpuTargets = [\"gfx906\"]; };"
    echo "    Option 2: build rocblas from source in the devshell and set"
    echo "              ROCBLAS_PATH to point at your build."
    echo ""
    exit 1
fi
