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
      };
    in
    {
      packages.${system}.xla-pjrt = let
        pythonEnv = pkgs.python3.withPackages (ps: [ ps.numpy ]);
        cudaPackages = pkgs.cudaPackages_12_9;
        # Runtime CUDA libraries for the GPU plugin.
        # libcuda.so.1 (the driver) is NOT included — it comes from the host
        # via addDriverRunpath (/run/opengl-driver/lib on NixOS).
        gpuRuntimeLibs = [
          pkgs.stdenv.cc.cc.lib  # libstdc++
          cudaPackages.cuda_cupti.lib
          cudaPackages.cuda_cudart
          cudaPackages.libcublas.lib
          cudaPackages.cudnn.lib
          cudaPackages.nccl
          cudaPackages.libcufft.lib
          cudaPackages.libcusparse.lib
          cudaPackages.cuda_nvrtc.lib
          cudaPackages.libnvjitlink.lib
          cudaPackages.libnvshmem
        ];
        gpuRpath = lib.makeLibraryPath gpuRuntimeLibs
          + ":${pkgs.addDriverRunpath.driverLink}/lib";
        cpuRpath = lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
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
          "//xla/pjrt/c:pjrt_c_api_cpu_plugin.so"
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
            cp bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so $out/lib/
            cp bazel-bin/xla/pjrt/c/pjrt_c_api_gpu_plugin.so $out/lib/
            cp xla/pjrt/c/pjrt_c_api.h $out/include/xla/pjrt/c/
            cp xla/pjrt/c/pjrt_c_api_macros.h $out/include/xla/pjrt/c/
            chmod +w $out/lib/*.so

            runHook postInstall
          '';

          # Set RPATH after the fixup phase to prevent Nix from shrinking it.
          # /run/opengl-driver/lib (for the NVIDIA driver on NixOS) would
          # otherwise be removed as "unnecessary".
          dontPatchELF = true;
          postFixup = ''
            patchelf --set-rpath "${cpuRpath}" $out/lib/pjrt_c_api_cpu_plugin.so
            patchelf --set-rpath "${gpuRpath}" $out/lib/pjrt_c_api_gpu_plugin.so
          '';
        };
      };

      checks.${system}.pjrt-cpu-test = pkgs.runCommand "pjrt-cpu-test" {
        nativeBuildInputs = [ pkgs.gcc ];
        pjrt = self.packages.${system}.xla-pjrt;
      } ''
        gcc -o test_pjrt ${./test_pjrt.c} \
          -I$pjrt/include -ldl -lm
        ./test_pjrt $pjrt/lib/pjrt_c_api_cpu_plugin.so
        touch $out
      '';
    };
}
