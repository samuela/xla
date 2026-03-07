{
  description = "XLA development environment";

  inputs = {
    # Pin to nixos-24.11 for glibc 2.40 compatibility with CUDA/nvcc.
    # glibc >= 2.41 adds C23 math functions (sinpi, cospi, rsqrt, etc.)
    # whose noexcept(true) specifications conflict with CUDA's declarations
    # in math_functions.h. This is a hard error in both GCC and clang and
    # is unfixed as of CUDA 13.1. See:
    # https://forums.developer.nvidia.com/t/323591
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # The bazel target used for build verification.
      xlaBuildTarget = "//xla/tools/multihost_hlo_runner:hlo_runner_main";
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          pkgs.bazel_7
          pkgs.python3
          pkgs.git
          pkgs.llvmPackages_18.clang
          pkgs.llvmPackages_18.lld
        ];

        # Disable fortify hardening — nvcc can't handle glibc's fortified
        # headers (__pass_object_size__ attributes cause "linkage specification
        # is incompatible" errors even with glibc 2.40).
        hardeningDisable = [ "fortify" ];

        shellHook = ''
          echo "XLA dev shell (Bazel $(bazel --version 2>&1 | grep -oP '\d+\.\d+\.\d+'), Clang $(clang --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1))"

          # Derive nix store paths for the configure script.
          export NIX_GCC_CXX_INCLUDE="$(echo | clang++ -xc++ -E -v - 2>&1 | grep -oP '/nix/store/[^ ]*/include/c\+\+/[0-9.]+' | head -1)"
          export NIX_GCC_CXX_PLATFORM_INCLUDE="$NIX_GCC_CXX_INCLUDE/x86_64-unknown-linux-gnu"
          export NIX_CLANG_RESOURCE_INCLUDE="$(clang -print-resource-dir)/include"
          export NIX_GLIBC_DEV_INCLUDE="$(echo | clang -xc -E -v - 2>&1 | grep -oP '/nix/store/[^ ]*glibc[^ ]*-dev/include' | head -1)"
          export NIX_GCC_LIB="$(dirname $(gcc -print-file-name=libstdc++.so.6))"

          # Write a helper script: configure-nix
          # Runs configure.py with the right flags, then appends nix workarounds.
          cat > .configure-nix << 'CONFIGURE_EOF'
          #!/usr/bin/env bash
          set -euo pipefail
          COMPUTE_CAPS="''${1:-sm_90}"

          echo "Configuring XLA for CUDA with compute capability $COMPUTE_CAPS..."
          ./configure.py \
            --backend=CUDA \
            --host_compiler=clang \
            --cuda_compiler=nvcc \
            --clang_path="$(which clang)" \
            --cuda_compute_capabilities="$COMPUTE_CAPS"

          # Append nix-specific workarounds to xla_configure.bazelrc.
          # The nix clang wrapper only adds C++ stdlib include paths when
          # invoked as clang++, but the CUDA crosstool always calls clang.
          # We must add them explicitly.
          cat >> xla_configure.bazelrc << EOF
          build --copt -isystem$NIX_GCC_CXX_INCLUDE
          build --copt -isystem$NIX_GCC_CXX_PLATFORM_INCLUDE
          build --copt -isystem$NIX_CLANG_RESOURCE_INCLUDE
          build --copt -isystem$NIX_GLIBC_DEV_INCLUDE
          build --host_copt -isystem$NIX_GCC_CXX_INCLUDE
          build --host_copt -isystem$NIX_GCC_CXX_PLATFORM_INCLUDE
          build --host_copt -isystem$NIX_CLANG_RESOURCE_INCLUDE
          build --host_copt -isystem$NIX_GLIBC_DEV_INCLUDE
          build --action_env NIX_HARDENING_ENABLE="$NIX_HARDENING_ENABLE"
          build --copt -U_FORTIFY_SOURCE
          build --copt -D_FORTIFY_SOURCE=0
          build --host_copt -U_FORTIFY_SOURCE
          build --host_copt -D_FORTIFY_SOURCE=0
          EOF

          echo "Done. xla_configure.bazelrc updated with nix workarounds."
          CONFIGURE_EOF
          chmod +x .configure-nix
          # Strip leading whitespace from heredoc
          sed -i 's/^          //' .configure-nix

          echo "  Run: ./.configure-nix [compute_capability]  (default: sm_90)"
          echo "  Then: bazel build --config cuda --action_env LD_LIBRARY_PATH=$NIX_GCC_LIB ${xlaBuildTarget}"
        '';
      };
    };
}
