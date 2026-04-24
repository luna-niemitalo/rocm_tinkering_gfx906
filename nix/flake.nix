{
  description = "gfx906 ROCm + PyTorch build environment";

  inputs = {
    # Unstable because rocmPackages (7.x) lives here.
    # fox_nest rule: never use nixos-unstable as system channel,
    # but pulling it as a flake input for a devShell is fine.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        # ------------------------------------------------------------------ #
        # GPU target — the only architecture we care about
        # ------------------------------------------------------------------ #
        gpuTarget = "gfx906"; # Radeon VII / Vega20 / MI50

        # ------------------------------------------------------------------ #
        # nixpkgs instance with rocmSupport scoped only to this env.
        # We do NOT set rocmSupport globally — that would drag it into
        # every package that has an optional rocm path (firefox, etc).
        # ------------------------------------------------------------------ #
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            (final: prev: {
              rocmPackages = prev.rocmPackages.overrideScope (
                fs: ps: {
                  # Pin clr (HIP runtime + OpenCL) to only our GPU target.
                  # This avoids building kernel objects for every AMD arch
                  # in the universe, and makes the audit in phase-1 clean:
                  # if gfx906 kernels are present after this, the override worked.
                  clr = ps.clr.override {
                    localGpuTargets = [ gpuTarget ];
                  };
                  # cmake extra flags
                  rocmClangStdenv = ps.rocmClangStdenv.override {
                    extraCMakeFlags = [
                      "-DBUILD_CLIENTS_TESTS=OFF"
                      "-DBUILD_CLIENTS_BENCHMARKS=OFF"
                      "-DBUILD_CLIENTS=OFF"
                      "-DTENSILE_COMPILER_FLAGS=-O2"
                    ];
                  };
                }
              );
            })
          ];
        };

        rocm = pkgs.rocmPackages;

        # ------------------------------------------------------------------ #
        # The packages we actually want in the shell.
        # Deliberately minimal — no miopen, no ml-sdk meta-packages.
        # Add things here only when a build step explicitly fails asking for them.
        # ------------------------------------------------------------------ #
        rocmDeps = with rocm; [
          # Core runtime
          clr # HIP runtime + OpenCL (our overridden, gfx906-only build)
          rocm-runtime # HSA runtime (rocr)
          rocm-toolchain

          # Compiler toolchain
          llvm.clang
          llvm.lld
          llvm.llvm
          rocm-device-libs # bitcode for target GPU

          # Math libs needed by PyTorch
          rocblas
          hipblas
          rocrand
          rocfft
          hipsparse

          # Tools for audit / debugging
          rocminfo
          rocm-smi
        ];


        # ------------------------------------------------------------------ #
        # Shared shellHook fragment — ROCm env vars, ccache, venv.
        # Both shells source this so they stay in sync.
        # ------------------------------------------------------------------ #
        commonShellHook = ''
			# ---------------------------------------------------------------- #
			# ROCm env
			# ---------------------------------------------------------------- #

			# Tell the HIP runtime which GPU we have.
			# Without this, newer ROCm versions refuse to run on gfx906
			# because it's not an "official" target anymore.
			export HSA_OVERRIDE_GFX_VERSION="9.0.6"

			# Prevent PyTorch cmake from going looking for CUDA
			export USE_CUDA=0
			export USE_CUDNN=0

			# We'll add miopen only if something explicitly breaks without it
			export USE_MIOPEN=0

			# Only compile for our arch — speeds up the pytorch build massively
			export PYTORCH_ROCM_ARCH="${gpuTarget}"

			# Tell ROCm where its own prefix is (some tools need this)
			export ROCM_PATH="${rocm.clr}"

			# ---------------------------------------------------------------- #
			# ccache — makes re-runs not take 8 hours
			# ---------------------------------------------------------------- #
			export CCACHE_DIR="/media/luna/cache/ccache"
			export CMAKE_C_COMPILER_LAUNCHER=ccache
			export CMAKE_CXX_COMPILER_LAUNCHER=ccache

			# ---------------------------------------------------------------- #
			# Library paths — exported for cmake and ldd sanity checks
			# ---------------------------------------------------------------- #
			export ROCBLAS_PATH="${rocm.rocblas}"
			export HIPBLAS_PATH="${rocm.hipblas}"
			export ROCRAND_PATH="${rocm.rocrand}"
			export ROCFFT_PATH="${rocm.rocfft}"
			export HIPSPARSE_PATH="${rocm.hipsparse}"
			export ROCM_DEVICE_LIBS_PATH="${rocm.rocm-device-libs}"
			export ROCM_CLR_PATH="${rocm.clr}"

        '';

      in
      {
        # ------------------------------------------------------------------ #
        # DEFAULT shell — PyTorch build env
        # Enter with: nix develop ./nix  (or direnv)
        # ------------------------------------------------------------------ #
        devShells.default = rocm.llvm.rocmClangStdenv.mkDerivation {
			name = "pytorch-rocm-gfx906";

			dontBuild = true;
			dontInstall = true;

			packages = [
				pkgs.glibc_multi
			];

			nativeBuildInputs = [
				pkgs.cmake
				pkgs.ninja
				pkgs.git
				pkgs.ccache
				pkgs.pkg-config
				pkgs.uv          # uv lives here, not in the venv
			];

			buildInputs = rocmDeps ++ [
				pkgs.zlib
				pkgs.zstd
				pkgs.glib
				pkgs.glibc
				pkgs.lld
				pkgs.libffi
			];

			shellHook = commonShellHook + ''
				echo ""
				echo "  gfx906 ROCm + PyTorch build shell"
				echo "  GPU target : ${gpuTarget} (HSA_OVERRIDE_GFX_VERSION=9.0.6)"
				echo "  ROCm path  : $ROCM_PATH"
				echo "  ccache dir : $CCACHE_DIR"
				echo ""
				echo "  Validate HIP before building:"
				echo "    hipcc --offload-arch=${gpuTarget} -c validate_hip.cpp -o /tmp/validate_hip.o && echo OK"
				echo "    hipcc --offload-arch=${gpuTarget} validate_hip.cpp -o /tmp/validate_hip && /tmp/validate_hip"
				echo ""
				echo ""
				echo "  First time setup:"
				echo "    uv venv .venv --python python3.12"
				echo "    source .venv/bin/activate"
				echo "    uv pip install -r requirements.txt"
				echo ""
				echo "  Then clone pytorch and build to wheel:"
				echo "    USE_ROCM=1 USE_MIOPEN=0 USE_CUDA=0 USE_FLASH_ATTN=0 PYTORCH_ROCM_ARCH=gfx906 python setup.py bdist_wheel 2>&1 | tee build.log"
				echo ""
				# ---------------------------------------------------------------- #
				# uv — prefer local build over nixpkgs bootstrap
				# ---------------------------------------------------------------- #
				LOCAL_UV="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/.venv/bin/uv"
				if [ -f "$LOCAL_UV" ]; then
					export PATH="$(dirname $LOCAL_UV):$PATH"
					echo "  uv         : local $($LOCAL_UV --version)"
					else
					echo "  uv         : nixpkgs bootstrap ($(uv --version))"
					echo "               to upgrade: uv pip install 'uv @ git+https://...'"
				fi

				# activate venv if it exists (creation is handled by .envrc)
				if [ -f ".venv/bin/activate" ]; then
					source .venv/bin/activate
				fi
				echo "  python     : $(python --version 2>&1)"

				echo ...
			'';
		};

        # ------------------------------------------------------------------ #
        # ROCm audit shell — for validation and debugging ROCm itself
        # Enter with: nix develop ./nix#rocm
        # ------------------------------------------------------------------ #
        devShells.rocm = rocm.llvm.rocmClangStdenv.mkDerivation {
          name = "rocm-audit-gfx906";

          dontBuild = true;
          dontInstall = true;

          packages = [
            pkgs.glibc_multi
          ];

          nativeBuildInputs = [
            pkgs.cmake
            pkgs.ninja
            pkgs.git
            pkgs.ccache
            pkgs.pkg-config
          ];

          buildInputs = rocmDeps ++ [
            pkgs.zlib
            pkgs.zstd
            pkgs.glib
            pkgs.glibc
          ];

          shellHook = commonShellHook + ''
            echo ""
            echo "  gfx906 ROCm audit shell"
            echo "  GPU target : ${gpuTarget} (HSA_OVERRIDE_GFX_VERSION=9.0.6)"
            echo "  ROCm path  : $ROCM_PATH"
            echo "  ccache dir : $CCACHE_DIR"
            echo ""
            echo "  ROCm paths:"
            echo "    rocblas      : $ROCBLAS_PATH"
            echo "    hipblas      : $HIPBLAS_PATH"
            echo "    rocrand      : $ROCRAND_PATH"
            echo "    rocfft       : $ROCFFT_PATH"
            echo "    hipsparse    : $HIPSPARSE_PATH"
            echo "    device-libs  : $ROCM_DEVICE_LIBS_PATH"
            echo "    clr          : $ROCM_CLR_PATH"
            echo ""
            echo "  Phase 1 audit (uses env vars, not nixpkgs channel):"
            echo "    rocminfo | grep -i 'gfx906\|Vega20\|MI50'"
            echo "    find \$ROCBLAS_PATH -name '*.co' | grep gfx906 | head -5"
            echo "    find \$ROCM_DEVICE_LIBS_PATH -name '*906*'"
            echo ""
            echo "  HIP compile check:"
            echo "    hipcc --offload-arch=${gpuTarget} -c validate_hip.cpp -o /tmp/validate_hip.o && echo OK"
            echo "    hipcc --offload-arch=${gpuTarget} validate_hip.cpp -o /tmp/validate_hip && /tmp/validate_hip"
            echo ""
          '';
        };
      }
    );
}
