# XLA PJRT Nix Package

Self-contained Nix package for XLA's PJRT C API plugins (CPU + GPU).

## Quick Start

```bash
# Build
nix build .#xla-pjrt

# Test
./test_pjrt.sh          # auto-detects: runs CPU + GPU if nvidia-smi works
./test_pjrt.sh cpu      # CPU only
./test_pjrt.sh gpu      # GPU only

# Nix check (CPU only — no GPU in Nix sandbox)
nix build .#checks.x86_64-linux.pjrt-cpu-test
```

## Output

```
$out/
  lib/
    pjrt_c_api_cpu_plugin.so        # ~250 MB, CPU plugin
    pjrt_c_api_gpu_plugin.so        # ~407 MB, GPU plugin
  include/xla/pjrt/c/
    pjrt_c_api.h
    pjrt_c_api_macros.h
```

## How It Works

### Build system

Uses `pkgs.buildBazelPackage` with XLA's hermetic build config
(`--config=pjrt_x86_cuda12_release`). The hermetic build downloads its own
clang 18, sysroot (glibc 2.27), CUDA 12.9.1, and cuDNN 9.8.0. No local CUDA
installation is needed to build.

Two-phase build:
1. **Fetch phase** (fixed-output derivation): `bazel build --nobuild` downloads
   all external dependencies. Produces a ~9 GB tarball cached in the Nix store.
2. **Build phase** (sandboxed): compiles from cached deps. ~38 min on 64-core.

### GPU plugin RPATH

The GPU plugin's RPATH points to nixpkgs CUDA 12.9 runtime libraries:
- cuda_cupti, cuda_cudart, libcublas, cudnn, nccl, libcufft, libcusparse,
  cuda_nvrtc, libnvjitlink, libnvshmem
- `/run/opengl-driver/lib` — NVIDIA driver (NixOS convention via `addDriverRunpath`)

The CPU plugin's RPATH points only to libstdc++.

### Compute capabilities

Builds for: `sm_50, sm_60, sm_70, sm_80, sm_90, sm_100, compute_120`

This covers Maxwell through Blackwell GPUs (T4=sm_75 via sm_70 compat,
V100=sm_70, A100=sm_80, H100=sm_90, B200=sm_100).

## Testing Details

### test_pjrt.c

A 337-line C program that exercises the PJRT C API end-to-end:
1. `dlopen` the plugin `.so`
2. `dlsym("GetPjrtApi")` to get the PJRT function table
3. Create a client
4. Compile a StableHLO axpy program: `result = alpha * x + y`
5. Create input buffers (alpha=3.14, x=[1,2,3,4], y=[10.5,20.5,30.5,40.5])
6. Execute and verify results match expected values

### test_pjrt.sh

Wrapper script that handles:
- **Compilation** of test_pjrt.c with the right flags
- **nix glibc interpreter** on non-NixOS (see below)
- **NVIDIA driver isolation** for GPU tests on non-NixOS

### Non-NixOS GPU testing

On non-NixOS Linux, the nixpkgs CUDA runtime libraries are built against nix
glibc (e.g. 2.42), which differs from the system glibc (e.g. 2.39 on Ubuntu
24.04). The test binary must use the nix glibc dynamic linker to avoid version
conflicts. The script handles this by:

1. Linking the test binary with `--dynamic-linker=<nix-glibc>/lib/ld-linux-x86-64.so.2`
2. Creating an isolated directory with only NVIDIA driver library symlinks
   (`libcuda.so*`, `libnvidia-*.so*`). Cannot add `/usr/lib/x86_64-linux-gnu`
   to `LD_LIBRARY_PATH` directly — it would pull in system `libc.so.6`.
3. Setting `LD_LIBRARY_PATH` to that isolated directory.

On NixOS, none of this is needed — the system already uses nix glibc and
`/run/opengl-driver/lib` is in the RPATH.

## Build Troubleshooting

### `/usr/bin/env` not found in sandbox (non-NixOS)

Bazel's py_binary stubs use `#!/usr/bin/env`. On non-NixOS, add to
`/etc/nix/nix.conf` (or `nix.custom.conf`):

```
extra-sandbox-paths = /usr/bin/env=/nix/store/<hash>-coreutils-<ver>/bin/coreutils
```

Find the path: `nix eval nixpkgs#coreutils.outPath --raw`

Use `bin/coreutils` (the real binary), not `bin/env` (a relative symlink that
breaks when bind-mounted).

### Fetch hash changes

If upstream dependencies change, the fetch phase hash
(`sha256-7/vMxY6U6t0Nr1V3w1TYCT6e/+pEyaj01TGWctfIabs=`) will need updating.
Build will fail with a hash mismatch — copy the correct hash from the error.

### dontPatchELF

`dontPatchELF = true` is required. Nix's fixup phase removes RPATH entries it
considers unnecessary, including `/run/opengl-driver/lib` and store paths for
libraries not directly referenced by the output. RPATH is set in `postFixup`.

### Bazel output files are read-only

`chmod +w $out/lib/*.so` is needed before patchelf can modify the `.so` files.
