#!/usr/bin/env bash
# Test XLA PJRT plugins (CPU and/or GPU).
#
# Usage:
#   ./test_pjrt.sh cpu          # test CPU plugin only
#   ./test_pjrt.sh gpu          # test GPU plugin only
#   ./test_pjrt.sh              # test both (GPU skipped if no nvidia-smi)
#
# The script expects `nix build .#xla-pjrt` output in ./result/ (default)
# or pass PJRT_DIR=/path/to/package to override.

set -euo pipefail

PJRT_DIR="${PJRT_DIR:-result}"

if [ ! -d "$PJRT_DIR/lib" ] || [ ! -d "$PJRT_DIR/include" ]; then
  echo "ERROR: $PJRT_DIR does not look like a PJRT package (missing lib/ or include/)"
  echo "Run: nix build .#xla-pjrt"
  exit 1
fi

# Detect NixOS vs non-NixOS.
is_nixos() {
  [ -f /etc/NIXOS ] || [ -d /run/opengl-driver/lib ]
}

# Compile the test binary. On non-NixOS, use nix glibc interpreter so
# the binary uses the same glibc as the nixpkgs CUDA libraries.
compile() {
  local binary="$1"
  local extra_flags=()

  if ! is_nixos; then
    local nix_interp nix_libstdcxx
    nix_interp="$(nix eval --raw 'nixpkgs#glibc')/lib/ld-linux-x86-64.so.2"
    nix_libstdcxx="$(nix eval --raw 'nixpkgs#gcc.cc.lib')/lib"
    extra_flags+=("-Wl,--dynamic-linker=$nix_interp" "-Wl,-rpath,$nix_libstdcxx")
  fi

  gcc -o "$binary" test_pjrt.c \
    -I "$PJRT_DIR/include" -ldl -lm \
    "${extra_flags[@]}"
}

# On non-NixOS, create a directory with only NVIDIA driver libraries
# (libcuda.so, libnvidia-*.so). We can't add /usr/lib/... directly to
# LD_LIBRARY_PATH because that would also pull in system glibc, which
# conflicts with the nix glibc our binary uses.
setup_driver_libs() {
  local driver_dir="$1"
  rm -rf "$driver_dir"
  mkdir -p "$driver_dir"

  local patterns=(
    "/usr/lib/x86_64-linux-gnu/libcuda.so*"
    "/usr/lib/x86_64-linux-gnu/libnvidia-*.so*"
  )
  for pattern in "${patterns[@]}"; do
    for f in $pattern; do
      [ -e "$f" ] && ln -sf "$f" "$driver_dir/$(basename "$f")"
    done
  done
}

run_test() {
  local plugin="$1"
  local name="$2"
  local env_prefix=()

  if [ "$name" = "gpu" ] && ! is_nixos; then
    local driver_dir="/tmp/pjrt-test-driver-libs"
    setup_driver_libs "$driver_dir"
    env_prefix=(env "LD_LIBRARY_PATH=$driver_dir")
  fi

  echo "=== Testing $name plugin: $plugin ==="
  "${env_prefix[@]}" "$BINARY" "$plugin"
  echo ""
}

# Parse arguments.
targets=("${@:-}")
if [ ${#targets[@]} -eq 0 ] || [ -z "${targets[0]}" ]; then
  targets=(cpu)
  if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    targets+=(gpu)
  else
    echo "Note: no GPU detected, skipping GPU test. Pass 'gpu' to force."
  fi
fi

# Compile once.
BINARY="$(mktemp /tmp/test_pjrt.XXXXXX)"
trap 'rm -f "$BINARY"' EXIT
echo "Compiling test binary..."
compile "$BINARY"

# Run requested tests.
for target in "${targets[@]}"; do
  case "$target" in
    cpu)
      run_test "$PJRT_DIR/lib/pjrt_c_api_cpu_plugin.so" cpu
      ;;
    gpu)
      run_test "$PJRT_DIR/lib/pjrt_c_api_gpu_plugin.so" gpu
      ;;
    *)
      echo "Unknown target: $target (use 'cpu' or 'gpu')"
      exit 1
      ;;
  esac
done
