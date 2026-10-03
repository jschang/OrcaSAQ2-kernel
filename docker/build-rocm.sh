#!/usr/bin/env bash
# Build the ROCm serving image. Run from the repo root: bash docker/build-rocm.sh [tag]
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:-orcasaq2-vllm-rocm:0.30.1}
# the daemon socket is root-owned on hosts where the user is not in the docker group
DOCKER=docker; docker info >/dev/null 2>&1 || DOCKER="sudo docker"
$DOCKER build -f docker/Dockerfile-rocm -t "$TAG" --build-arg VLLM_TAG="${VLLM_TAG:-v0.30.0}" .
echo "built $TAG"
echo
echo "Run it with the AMD device nodes, not --gpus all (that flag is nvidia-container-toolkit only):"
echo
echo "  docker run --device /dev/kfd --device /dev/dri --group-add video \\"
echo "    --shm-size=16g -v /data/models/<pack>:/model:ro -p 8000:8000 \\"
echo "    -e PRESET=16gb $TAG"
echo
echo "Two launch gotchas baked in or noted for ROCm (ROCM-DEBUG-2.md session 4):"
echo "  * --shm-size=16g (or --ipc=host): the vLLM engine core mmaps an ~8.5 GiB offload"
echo "    region into /dev/shm during its profile run; Docker's default 64 MiB fails it"
echo "    with 'Insufficient space in /dev/shm' before the model loads."
echo "  * TVM_FFI_DISABLE_TORCH_C_DLPACK=1 is already an image ENV (Dockerfile-rocm): it"
echo "    skips the one-time tvm_ffi torch-C-dlpack addon build that forks hundreds of"
echo "    compiles and can freeze the box on first import in a fresh container."
echo
echo "Presets carry the measured values; prefer them over naming knobs by hand. The 16 GB numbers"
echo "were measured on NVIDIA cards, so re-tune them for your gfx1100 card if you hit OOM."
echo
echo "Add -e API_KEY=... to require a bearer token. Note vLLM guards only /v1 /v2 /inference"
echo "/cohere with it -- /health and /metrics stay open, so publish the port accordingly."
echo "If a 16 GB card OOMs at long context: add -e CPU_OFFLOAD_GB=4   (slow, but it runs)"
