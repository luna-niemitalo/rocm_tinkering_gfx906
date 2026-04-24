# gfx906 ROCm 7.2 + PyTorch — NixOS Build Environment

ROCm 7.2 + PyTorch from source on **Radeon VII / Vega20 / MI50** (gfx906), running NixOS.
gfx906 is not an official ROCm 7.x target, but the hardware support in LLVM/AMDGPU has not been removed — it works if you build from source and know what to set.

---

## Quick start (future me)

```bash
cd ~/tinker/Rocm7_tinkering
# direnv loads the flake automatically on cd
# if .venv doesn't exist yet, direnv creates it via uv

# validate ROCm
nix develop ./nix#rocm
hipcc --offload-arch=gfx906 -c validate_hip.cpp -o /tmp/v.o && echo OK
hipcc --offload-arch=gfx906 validate_hip.cpp -o /tmp/v && /tmp/v

# build PyTorch (default shell)
git clone --recursive --branch v2.9.0 https://github.com/pytorch/pytorch
cd pytorch
uv pip install -r ../requirements.txt  # build deps only, not torch
USE_ROCM=1 USE_MIOPEN=0 USE_CUDA=0 USE_FLASH_ATTN=0 \
  PYTORCH_ROCM_ARCH=gfx906 \
  python setup.py bdist_wheel 2>&1 | tee build.log
cp dist/torch-*.whl ../wheels/
cd ..
uv pip install wheels/torch-*.whl
```

---

## For other gfx906 users

This repo contains a NixOS flake devShell that sets up a working ROCm 7.2 build environment targeting gfx906 exclusively. If you're on a non-NixOS distro, the env vars and build flags documented here still apply — you just need to supply your own ROCm 7.x installation.

**Hardware this was developed and validated on:** dual Radeon VII (gfx906:sramecc-:xnack-), NixOS 25.11, kernel 6.x.

---

## Architecture

```
Nix flake (nix/)
  └── devShells
        ├── default   — PyTorch build environment (what direnv loads)
        └── rocm      — ROCm audit/validation shell (nix develop ./nix#rocm)

Python layer (uv-managed, fully isolated)
  ├── .venv/           — created by direnv on first load
  ├── requirements.txt — build deps (everything except torch)
  └── wheels/          — local torch wheel after build

Project files
  ├── validate_hip.cpp — HIP smoke test (device count, arch string, kernel launch)
  └── .envrc           — direnv config
```

**Design principle:** Nix is the system layer (ROCm libs, compiler toolchain, cmake, git, bootstrap uv). uv is the Python layer (interpreter, venv, all packages including torch). Nothing Python-related leaks into the Nix derivation beyond the bootstrap uv binary.

---

## Shells

### `default` — PyTorch build env

```bash
# entered automatically by direnv, or manually:
nix develop ./nix
```

Contains: full ROCm 7.2 toolchain, cmake, ninja, ccache, git, bootstrap uv. Activates `.venv` if it exists. Sets all ROCm env vars and build flags.

### `rocm` — audit/validation shell

```bash
nix develop ./nix#rocm
```

Same ROCm toolchain, no Python/uv. Use this to debug ROCm itself without the Python layer in the way. Prints all store paths on activation so you can verify what you're actually linked against.

---

## Phase 1 audit — findings

**Question:** Does nixpkgs ROCm 7.2 actually have gfx906 kernels compiled in, or is it just a package that exists?

**Method:**

```bash
# GPU visible to ROCm runtime
rocminfo | grep -i 'gfx906\|Vega20\|MI50'

# rocblas has precompiled Tensile kernel objects for gfx906
find $ROCBLAS_PATH -name '*.co' | grep gfx906 | head -5

# device-libs has gfx906 ISA bitcode
find $ROCM_DEVICE_LIBS_PATH -name '*906*'

# HIP compiler can target gfx906
hipcc --offload-arch=gfx906 -c validate_hip.cpp -o /tmp/v.o && echo OK
```

**Findings:**

- `rocminfo` shows both GPUs as `gfx906:sramecc-:xnack-` ✓
- `rocblas` (7.2.2) has gfx906 Tensile `.co` files including `HH` (fp16) and `DD` (fp64) — the ones that matter for ML ✓
- `rocm-device-libs` has `oclc_isa_version_906.bc` ✓
- `hipcc --offload-arch=gfx906` compiles and links successfully ✓
- Full kernel launch + result verification passes on both GPUs ✓

**Conclusion:** nixpkgs ROCm 7.2 has gfx906 support compiled in. No need to build ROCm from source.

**Note on overrides:** The flake uses `overrideScope` to pin `clr` to `localGpuTargets = [ "gfx906" ]`, which forced a local rebuild of that component (invalidates the binary cache path). This was intentional — it ensures the HIP runtime only contains gfx906 kernel objects rather than the full multi-arch default build. This rebuild is a one-time cost; subsequent `direnv reload` cycles hit the local store.

---

## Key decisions

### ROCm: nixpkgs derivations with scope override, not manual source build

The flake uses `overrideScope` on `rocmPackages` to replace `clr` (gfx906-only
GPU targets) and `rocmClangStdenv` (test/benchmark flags stripped). Because
all math libs (`rocblas`, `hipblas`, `rocrand`, `rocfft`, `hipsparse`) use
`rocmClangStdenv` to compile, they were all rebuilt from source with the new
stdenv — their nixpkgs binary cache paths no longer matched.

The win over a manual ROCm source build: Nix handled the derivations,
dependency wiring, and source fetches. The override just changed two nodes in
the graph and let Nix propagate the consequences. No custom cmake superbuild,
no manually written derivations, no source checkouts to manage.

The one-time rebuild cost was the price of having a coherent, gfx906-only
ROCm stack rather than a multi-arch default build.

### PyTorch: 2.9.x with `USE_FLASH_ATTN=0`

- 2.9.x preferred over 2.7.x for inference performance improvements (better hipBLAS dispatch, memory-efficient attention paths)
- `USE_FLASH_ATTN=0` required: aotriton (the Triton-based flash attention backend) does not support gfx906 as a build target. Attempting to build with it enabled fails mid-compile. Standard attention paths work fine.
- `USE_MIOPEN=0`: MiOpen excluded deliberately. It pulls in a large dependency tree and is not needed for the target workloads. Add it only if a specific op explicitly fails without it.

### Python: 3.12, uv-managed

- AMD's official ROCm 7.2.1 wheel builds target Python 3.12, meaning the most tested combination with this ROCm version
- uv manages the interpreter, venv, and all packages — nothing Python-related in the Nix derivation except the bootstrap uv binary
- This gives a clean separation: flake = system layer, requirements.txt + wheels/ = Python layer

### Wheel-based torch install

`setup.py bdist_wheel` produces a `.whl` file stored in `wheels/`. This goes into `requirements.txt` as a local path. The benefit: rebuilding the environment later is `uv pip install -r requirements.txt` — no re-engineering, just time to recompile if the wheel is lost.

### uv bootstrap pattern

nixpkgs provides a bootstrap uv. The shellHook checks for `.venv/bin/uv` and if found, prepends it to PATH, shadowing the Nix one. This allows installing a custom uv build (e.g. a branch with wheel variant support) into the venv, which then takes precedence on all subsequent shell entries. The Nix uv is only ever used once — to create the venv the first time.

---

## Environment variables set by the shell

| Variable | Value | Purpose |
|---|---|---|
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.6` | Makes ROCm runtime accept gfx906 on ROCm 7.x where it's not an official target |
| `PYTORCH_ROCM_ARCH` | `gfx906` | Restricts PyTorch build to only this arch — cuts compile time significantly |
| `USE_ROCM` | `1` | Enable ROCm backend in PyTorch build |
| `USE_MIOPEN` | `0` | Disable MiOpen — excluded by design |
| `USE_CUDA` / `USE_CUDNN` | `0` | Prevent cmake from hunting for CUDA |
| `USE_FLASH_ATTN` | `0` | Disable aotriton — not supported on gfx906 |
| `ROCM_PATH` | nix store path | Points tools to the ROCm prefix |
| `CCACHE_DIR` | `/media/luna/cache/ccache` | Persistent ccache across rebuilds |

---

## Rebuild procedure

If the system is wiped and needs to be reconstructed:

1. Enter the project directory — direnv loads the flake, creates `.venv`
2. Run the Phase 1 audit commands to confirm ROCm still works
3. Run `validate_hip.cpp` to confirm the compiler and runtime are healthy
4. Clone PyTorch at the same tag, run the build command with the same flags
5. Copy the new wheel to `wheels/`, run `uv pip install -r requirements.txt`

Total time: however long the PyTorch compile takes (~4–8h without ccache warmup, much less with).

### On the ROCm math stack rebuild: it will crash, keep going

The LLVM frontend used by the ROCm build is flaky under sustained parallel
load. During the initial build of this environment, the compile crashed
approximately two dozen times before completing. This is normal.

**The pattern to distinguish "flaky crash" from "real error":**
- Flaky: crash point moves around between runs, or the same point crashes
  once then succeeds next time
- Real error: you get the exact same error at the exact same point two or
  three runs in a row

If it's flaky, just run `nix develop` again. Nix's incremental build picks up
where it left off — you're not recompiling from zero each time. Eventually it
gets through.

The flags that got it over the line:
- `-DBUILD_CLIENTS_TESTS=OFF` / `-DBUILD_CLIENTS_BENCHMARKS=OFF` / `-DBUILD_CLIENTS=OFF`
  — removes a large chunk of compilation that isn't needed at runtime
- `-DTENSILE_COMPILER_FLAGS=-O2` instead of the default `-O3`
  — reduces peak memory and compiler stress during Tensile kernel compilation,
  which is where most of the crashes were happening

---

## Files

| File | Purpose |
|---|---|
| `nix/flake.nix` | Devshell definitions, ROCm override, all env vars |
| `nix/flake.lock` | Pinned nixpkgs rev — do not update casually |
| `.envrc` | direnv config: loads flake, bootstraps .venv |
| `validate_hip.cpp` | HIP smoke test — device enumeration + kernel launch |
| `requirements.txt` | Python build deps + local torch wheel path |
| `wheels/` | Compiled torch .whl — the artifact of the source build |
