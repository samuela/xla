{
  description = "XLA development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
        overlays = [
          # Fix incomplete glibc 2.42 noexcept patch in cuda_nvcc.
          # nixpkgs patches math_functions.h but misses math_functions.hpp.
          # TODO: upstream to nixpkgs
          (final: prev: {
            cudaPackages_12_9 = prev.cudaPackages_12_9.overrideScope (
              cfinal: cprev: {
                cuda_nvcc = cprev.cuda_nvcc.overrideAttrs (oldAttrs: {
                  postInstall = (oldAttrs.postInstall or "") + ''
                    nixLog "Patching math_functions.hpp signatures to match glibc's ones"
                    sed -i \
                      -e 's/__func__(double rsqrt(const double a))/__func__(double rsqrt(const double a) throw())/' \
                      -e 's/__func__(double sinpi(double a))/__func__(double sinpi(double a) throw())/' \
                      -e 's/__func__(double cospi(double a))/__func__(double cospi(double a) throw())/' \
                      -e 's/__func__(float rsqrtf(const float a))/__func__(float rsqrtf(const float a) throw())/' \
                      -e 's/__func__(float sinpif(const float a))/__func__(float sinpif(const float a) throw())/' \
                      -e 's/__func__(float cospif(const float a))/__func__(float cospif(const float a) throw())/' \
                      "''${!outputInclude:?}/include/crt/math_functions.hpp"
                  '';
                });
              }
            );
          })
        ];
      };

      cudaPackages = pkgs.cudaPackages_12_9;

      # Use the CUDA-compatible stdenv (GCC version validated against nvcc).
      stdenv = cudaPackages.backendStdenv;

      # Merged CUDA toolkit tree — combines all split CUDA packages into a
      # single directory with include/, lib/, bin/, nvvm/ subdirs.
      # This is what XLA's LOCAL_CUDA_PATH expects.
      cudaMerged = cudaPackages.cudatoolkit;

      # Merged cuDNN tree for LOCAL_CUDNN_PATH.
      cudnnMerged = pkgs.symlinkJoin {
        name = "cudnn-merged";
        paths = [
          cudaPackages.cudnn.lib
          cudaPackages.cudnn.dev
          cudaPackages.cudnn.include
        ];
      };

      # Merged NCCL tree for LOCAL_NCCL_PATH.
      ncclMerged = pkgs.symlinkJoin {
        name = "nccl-merged";
        paths = [
          cudaPackages.nccl.out
          cudaPackages.nccl.dev
        ];
      };

      # The bazel target used for build verification.
      xlaBuildTarget = "//xla/tools/multihost_hlo_runner:hlo_runner_main";
    in
    {
      devShells.${system}.default = (pkgs.mkShell.override { inherit stdenv; }) {
        packages = [
          pkgs.bazel_7
          pkgs.python3
          pkgs.git
        ];

        # Disable fortify hardening — nvcc can't handle glibc's fortified
        # headers (__pass_object_size__ attributes cause "linkage specification
        # is incompatible" errors).
        hardeningDisable = [ "fortify" ];

        CUDA_MERGED = cudaMerged;
        CUDNN_MERGED = cudnnMerged;
        NCCL_MERGED = ncclMerged;

        shellHook = ''
          export PATH="$CUDA_MERGED/bin:$PATH"
          echo "XLA dev shell (Bazel $(bazel --version 2>&1 | grep -oP '\d+\.\d+\.\d+'), GCC $(gcc --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1), CUDA $(nvcc --version 2>&1 | grep -oP 'V\K[\d.]+'))"
          echo "  CUDA_MERGED=$CUDA_MERGED"
          echo "  Build target: ${xlaBuildTarget}"
          echo ""
          echo "To build:"
          echo "  python3 configure.py --backend CUDA --host_compiler GCC --cuda_compiler NVCC --nccl \\"
          echo "    --local_cuda_path \$CUDA_MERGED --local_cudnn_path \$CUDNN_MERGED \\"
          echo "    --local_nccl_path \$NCCL_MERGED --cuda_compute_capabilities 9.0"
          echo "  bazel build --config cuda ${xlaBuildTarget}"
        '';
      };
    };
}
