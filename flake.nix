{
  description = "XLA development environment";

  # NOTE: The hermetic XLA build generates scripts with #!/usr/bin/env shebangs
  # (Bazel's py_binary stubs, XNNPACK codegen, etc.). On non-NixOS Linux, add
  # /usr/bin/env to the sandbox in /etc/nix/nix.conf (or nix.custom.conf):
  #
  #   extra-sandbox-paths = /usr/bin/env=/nix/store/<hash>-coreutils-<ver>/bin/coreutils
  #
  # Use bin/coreutils (not bin/env, which is a relative symlink that breaks).
  # Find the path: nix eval nixpkgs#coreutils.outPath --raw

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      lib = nixpkgs.lib;
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

                    # When clang compiles CUDA, host_defines.h redefines __noinline__
                    # as __attribute__((noinline)). This conflicts with libstdc++ >=12
                    # which uses __attribute__((__noinline__)) — the macro expands to
                    # __attribute__((__attribute__((noinline)))) which is invalid.
                    # Clang natively understands __noinline__ as an attribute, so the
                    # macro is unnecessary. Skip it when clang is the compiler.
                    nixLog "Patching host_defines.h to skip __noinline__ macro under clang"
                    sed -i \
                      's/#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)/#if (defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__CUDA_LIBDEVICE__)) \&\& !defined(__clang__)/' \
                      "''${!outputInclude:?}/include/crt/host_defines.h"

                    # Clang 19 CUDA mode: placement new from <new> is __host__ only,
                    # but device code (CUB/CCCL) needs it. Add a header declaring
                    # __host__ __device__ placement new. Force-included via --cxxopt.
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
      # cudaPackages.cudatoolkit doesn't include libnvjitlink, which XLA needs.
      cudaMerged = pkgs.symlinkJoin {
        name = "cuda-merged";
        paths = [
          cudaPackages.cudatoolkit
          cudaPackages.libnvjitlink.lib
          cudaPackages.libnvjitlink.dev
          cudaPackages.libnvjitlink.include
        ];
      };

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

      # libstdc++ path for rpath — needed because cuda_clang config uses
      # -fuse-ld=lld for host tools, and lld doesn't add rpaths automatically.
      libstdcxxPath = "${stdenv.cc.cc.lib}/lib";

      # Clang 19 wrapped to use GCC 14.3.0 (backendStdenv) instead of GCC 15.2.0.
      # GCC 15 headers use [[gnu::noinline]] syntax incompatible with clang's
      # CUDA mode. We override cc-cflags to point at backendStdenv's GCC.
      gcc14 = stdenv.cc.cc;
      gcc15 = pkgs.gcc.cc;  # default nixpkgs GCC (15.2.0)
      clang = pkgs.llvmPackages_19.clang.overrideAttrs (old: {
        postFixup = (old.postFixup or "") + ''
          # Replace all GCC 15 references with GCC 14 in nix-support files
          for f in $out/nix-support/cc-cflags $out/nix-support/libcxx-cxxflags; do
            if [ -f "$f" ]; then
              sed -i "s|${gcc15}|${gcc14}|g; s|${gcc15.version}|${gcc14.version}|g" "$f"
            fi
          done

          # Add device placement new header to clang's resource-root.
          # Clang 19 CUDA mode only has __host__ placement new from <new>;
          # CUB/CCCL need __device__ overloads too. This header is
          # force-included via --cxxopt and is guarded by __CUDA__.
          # resource-root/include is a symlink — replace with a real dir
          # containing the original files plus our header.
          if [ -L "$out/resource-root/include" ]; then
            target=$(readlink -f "$out/resource-root/include")
            rm "$out/resource-root/include"
            mkdir "$out/resource-root/include"
            for f in "$target"/*; do
              ln -s "$f" "$out/resource-root/include/$(basename "$f")"
            done
          fi
          cat > $out/resource-root/include/cuda_device_placement_new.h << 'HEADER_EOF'
#ifndef CUDA_DEVICE_PLACEMENT_NEW_H_
#define CUDA_DEVICE_PLACEMENT_NEW_H_
#if defined(__CUDA__) && defined(__clang__)
#include <new>
__device__ inline void* operator new(__SIZE_TYPE__, void* __p) noexcept { return __p; }
__device__ inline void* operator new[](__SIZE_TYPE__, void* __p) noexcept { return __p; }
__device__ inline void operator delete(void*, void*) noexcept {}
__device__ inline void operator delete[](void*, void*) noexcept {}
#endif
#endif
HEADER_EOF
        '';
      });

      lld = pkgs.llvmPackages_19.lld;

      # The bazel target used for build verification.
      xlaBuildTarget = "//xla/tools/multihost_hlo_runner:hlo_runner_main";
    in
    {
      devShells.${system}.default = (pkgs.mkShell.override { inherit stdenv; }) {
        packages = [
          pkgs.bazel_7
          pkgs.python3
          pkgs.git
          clang
          lld
        ];

        # Disable fortify hardening — nvcc can't handle glibc's fortified
        # headers (__pass_object_size__ attributes cause "linkage specification
        # is incompatible" errors).
        hardeningDisable = [ "fortify" ];

        CUDA_MERGED = cudaMerged;
        CUDNN_MERGED = cudnnMerged;
        NCCL_MERGED = ncclMerged;
        LIBSTDCXX_PATH = libstdcxxPath;

        shellHook = ''
          export PATH="$CUDA_MERGED/bin:$PATH"
          echo "XLA dev shell (Bazel $(bazel --version 2>&1 | grep -oP '\d+\.\d+\.\d+'), clang $(clang --version 2>&1 | grep -oP '\d+\.\d+\.\d+' | head -1), CUDA $(nvcc --version 2>&1 | grep -oP 'V\K[\d.]+'))"
          echo "  CUDA_MERGED=$CUDA_MERGED"
          echo "  Build target: ${xlaBuildTarget}"
          echo ""
          echo "To build:"
          echo "  ./.configure-nix && bazel build --config cuda ${xlaBuildTarget}"

          # Write a configure wrapper that runs configure.py then appends
          # nix-specific workarounds to xla_configure.bazelrc.
          cat > .configure-nix << CONFIGURE_EOF
#!/usr/bin/env bash
set -euo pipefail
CAPS="\''${1:-9.0}"

python3 configure.py \\
  --backend CUDA --host_compiler CLANG --cuda_compiler CLANG --nccl \\
  --clang_path "$(which clang)" --lld_path "$(which ld.lld)" \\
  --local_cuda_path "$CUDA_MERGED" --local_cudnn_path "$CUDNN_MERGED" \\
  --local_nccl_path "$NCCL_MERGED" --cuda_compute_capabilities "\$CAPS"

# Append nix-specific workarounds.
cat >> xla_configure.bazelrc << EOF
build --action_env NIX_HARDENING_ENABLE="$NIX_HARDENING_ENABLE"
build --copt -U_FORTIFY_SOURCE
build --copt -D_FORTIFY_SOURCE=0
build --host_copt -U_FORTIFY_SOURCE
build --host_copt -D_FORTIFY_SOURCE=0
build --copt -Wno-error=unused-command-line-argument
build --copt -Wno-gnu-offsetof-extensions
build --cxxopt=-include --cxxopt=cuda_device_placement_new.h
build --copt -fgpu-defer-diag
build --copt -U_GLIBCXX_HAVE_IS_CONSTANT_EVALUATED
build --linkopt -lm
build --host_linkopt -Wl,-rpath,$LIBSTDCXX_PATH
build --linkopt -Wl,-rpath,$LIBSTDCXX_PATH
EOF

echo "Done. xla_configure.bazelrc updated with nix workarounds."
CONFIGURE_EOF
          chmod +x .configure-nix
        '';
      };

      packages.${system}.xla-pjrt = let
        pythonEnv = pkgs.python3.withPackages (ps: [ ps.numpy ]);
      in pkgs.buildBazelPackage {
        pname = "xla-pjrt";
        version = "0-unstable-2026-03-08";

        src = lib.cleanSource self;

        bazel = pkgs.bazel_7;

        nativeBuildInputs = [
          pkgs.gitMinimal
          pythonEnv
          pkgs.which
          pkgs.patchelf
        ];

        # Libraries needed by the hermetic clang/LLVM/CUDA binaries.
        # These are Ubuntu-built ELF binaries that need Nix glibc + deps.
        buildInputs = [
          pkgs.glibc
          pkgs.gcc.cc.lib  # libstdc++
          pkgs.zlib
          pkgs.ncurses
          pkgs.libxml2
        ];

        postPatch = ''
          rm -f .bazelversion
          patchShebangs .
        '';

        # No configure.py — the hermetic build config downloads its own
        # clang, sysroot, and CUDA toolkit.

        bazelTargets = [
          "//xla/pjrt/c:pjrt_c_api_gpu_plugin.so"
        ];

        # --config must be in bazelFlags (not bazelBuildFlags) so it applies
        # to both the fetch phase (build --nobuild) and the build phase.
        # Without it during fetch, --repo_env=HERMETIC_CUDA_VERSION is unset
        # and the CUDA redistribution JSON repos return empty results.
        #
        # Use local spawn strategy (not sandboxed) so the hermetic toolchain
        # binaries at external/llvm18_linux_x86_64/bin/ are accessible.
        bazelFlags = [
          "--config=pjrt_x86_cuda12_release"
          "--spawn_strategy=local"
          "--genrule_strategy=local"
          "--python_path=${pythonEnv}/bin/python3"
          # Exec-config binaries (tools compiled by hermetic clang) need the
          # Nix dynamic linker — /lib64/ld-linux-x86-64.so.2 doesn't exist in
          # the Nix sandbox. Also set rpath so they find glibc + libstdc++.
          "--host_linkopt=-Wl,--dynamic-linker=${pkgs.glibc}/lib/ld-linux-x86-64.so.2"
          "--host_linkopt=-Wl,-rpath,${pkgs.lib.makeLibraryPath [
            pkgs.glibc
            pkgs.gcc.cc.lib
            pkgs.zlib
          ]}"
        ];

        dontAddBazelOpts = true;
        removeRulesCC = false;
        removeLocal = false;

        fetchAttrs = {
          sha256 = "sha256-7/vMxY6U6t0Nr1V3w1TYCT6e/+pEyaj01TGWctfIabs=";

          # Custom installPhase: the default buildBazelPackage installPhase
          # removes ALL top-level symlinks from $bazelOut/external/ (lines
          # 214-218 of build-bazel-package/default.nix). For the hermetic GPU
          # build, repos like cuda_nvcc, cuda_cudart, cuda_cudnn, the sysroot,
          # and LLVM toolchain are stored as top-level symlinks pointing to the
          # actual downloaded content. Removing them leaves empty wrapper repos.
          #
          # We replace the symlink removal with symlink-to-directory conversion
          # using cp -rL, preserving the hermetic build content in the tarball.
          installPhase = ''
            runHook preInstall

            # Remove machine-local repos that will be regenerated during build.
            rm -rf $bazelOut/external/{local_config_python,\@local_config_python.marker}
            rm -rf $bazelOut/external/{local_config_sh,\@local_config_sh.marker}
            rm -rf $bazelOut/external/{local_config_xcode,\@local_config_xcode.marker}
            rm -rf $bazelOut/external/{local_execution_config_python,\@local_execution_config_python.marker}
            rm -rf $bazelOut/external/{local_jdk,\@local_jdk.marker}

            # Remove built-in external workspaces (Bazel recreates them).
            rm -rf $bazelOut/external/{bazel_tools,\@bazel_tools.marker}
            rm -rf $bazelOut/external/{embedded_jdk,\@embedded_jdk.marker}

            # Clear markers so the nix-hack patch skips validation.
            find $bazelOut/external -name '@*\.marker' -exec sh -c 'echo > {}' \;

            # Remove VCS directories.
            rm -rf $(find $bazelOut/external -type d -name .git)
            rm -rf $(find $bazelOut/external -type d -name .svn)
            rm -rf $(find $bazelOut/external -type d -name .hg)

            # Convert top-level symlinks to real directories instead of
            # removing them. This preserves hermetic CUDA/LLVM/sysroot content.
            find $bazelOut/external -maxdepth 1 -type l | while read symlink; do
              target="$(readlink -f "$symlink")"
              if [ -d "$target" ]; then
                rm "$symlink"
                cp -rL "$target" "$symlink"
              else
                # Non-directory symlink (e.g. pointing to temp paths) — remove it.
                name="$(basename "$symlink")"
                rm "$symlink"
                test -f "$bazelOut/external/@$name.marker" && rm "$bazelOut/external/@$name.marker" || true
              fi
            done

            # Patch remaining symlinks to remove build directory references.
            find $bazelOut/external -type l | while read symlink; do
              new_target="$(readlink "$symlink" | sed "s,$NIX_BUILD_TOP,NIX_BUILD_TOP,")"
              rm "$symlink"
              ln -sf "$new_target" "$symlink"
            done

            echo '${pkgs.bazel_7.name}' > $bazelOut/external/.nix-bazel-version

            (cd $bazelOut/ && tar czf $out --sort=name --mtime='@1' --owner=0 --group=0 --numeric-owner external/)

            runHook postInstall
          '';
        };

        buildAttrs = {
          preConfigure = ''
            # The riegeli repo has no top-level BUILD or WORKSPACE file, so
            # the Bazel nix-hack (which checks for these to decide if a cached
            # repo is valid) falls through and tries to re-fetch.
            touch $bazelOut/external/riegeli/WORKSPACE

            # Patch shebangs in hermetic toolchain wrappers — they use
            # #!/usr/bin/env python3 which doesn't work in the nix sandbox.
            patchShebangs $bazelOut/external/rules_ml_toolchain
            patchShebangs $bazelOut/external/python_3_11_x86_64-unknown-linux-gnu

            # Patch dynamic linker and rpath in hermetic ELF binaries.
            # These are Ubuntu-built binaries that use /lib64/ld-linux-x86-64.so.2
            # and expect system libraries (libz, libncurses, etc.) at FHS paths.
            # Bazel uses `env -` which strips LD_LIBRARY_PATH, so rpath is needed.
            echo "Patching hermetic ELF binaries..."
            NIX_INTERP="$(cat $NIX_CC/nix-support/dynamic-linker)"
            NIX_RPATH="${pkgs.lib.makeLibraryPath [
              pkgs.glibc
              pkgs.gcc.cc.lib
              pkgs.zlib
              pkgs.ncurses5
              pkgs.ncurses
              pkgs.libxml2
              pkgs.elfutils
            ]}"
            # Add hermetic LLVM's own lib dir for bundled libtinfo.so.5
            LLVM_LIB="$bazelOut/external/llvm18_linux_x86_64/lib"
            NIX_RPATH="$LLVM_LIB:$NIX_RPATH"
            for dir in \
              $bazelOut/external/llvm18_linux_x86_64 \
              $bazelOut/external/cuda_nvcc \
              $bazelOut/external/cuda_cupti \
              $bazelOut/external/python_3_11_x86_64-unknown-linux-gnu; do
              if [ -d "$dir" ]; then
                find "$dir" -type f -executable | while read f; do
                  if file "$f" | grep -q 'ELF.*dynamically linked'; then
                    patchelf --set-interpreter "$NIX_INTERP" --set-rpath "$NIX_RPATH" "$f" 2>/dev/null || true
                  fi
                done
              fi
            done
            echo "Done patching hermetic binaries."
          '';

          installPhase = ''
            runHook preInstall

            mkdir -p $out/lib $out/include/xla/pjrt/c
            cp bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so $out/lib/
            cp xla/pjrt/c/pjrt_c_api.h $out/include/xla/pjrt/c/
            cp xla/pjrt/c/pjrt_c_api_macros.h $out/include/xla/pjrt/c/

            runHook postInstall
          '';
        };
      };
    };
}
