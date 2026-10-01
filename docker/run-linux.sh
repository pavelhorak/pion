#!/usr/bin/env bash
# Run an interactive Linux ARM64 dev shell with the Pion project mounted.
# io_uring enabled via --security-opt seccomp=unconfined (Colima kernel 6.8).
#
# Usage:
#   ./docker/run-linux.sh                    # interactive shell
#   ./docker/run-linux.sh "pixi run build"   # run a single command and exit
#   ./docker/run-linux.sh "pixi run build && ./pion-server -p 1974 -w 4 --independent-workers"

set -e

DOCKER_HOST="unix:///Users/$(whoami)/.colima/default/docker.sock"
export DOCKER_HOST

IMAGE="pion-linux-dev"
PION_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Check Colima is running
if ! colima status 2>/dev/null | grep -q "is running"; then
  echo "Starting Colima..."
  colima start --arch aarch64 --cpu 4 --memory 8
fi

# Build image if not present or Dockerfile changed
if ! docker image inspect "$IMAGE" &>/dev/null || \
   [ "$PION_DIR/Dockerfile.linux" -nt <(docker image inspect "$IMAGE" --format '{{.Created}}' 2>/dev/null || echo 0) ]; then
  echo "Building $IMAGE..."
  docker build \
    -f "$PION_DIR/Dockerfile.linux" \
    -t "$IMAGE" \
    "$PION_DIR"
fi

CMD="${1:-bash --login}"

echo "Starting Linux ARM64 container (kernel $(docker run --rm --security-opt seccomp=unconfined $IMAGE uname -r))..."

docker run -it --rm \
  --security-opt seccomp=unconfined \
  --network host \
  -v "$PION_DIR":/pion \
  -v "$PION_DIR"/.pixi-linux:/pion/.pixi \
  -w /pion \
  -e TERM=xterm-256color \
  "$IMAGE" bash -c "
    export PATH=/root/.pixi/bin:\$PATH
    echo '=== Pion Linux Dev (kernel: \$(uname -r)) ==='
    echo 'io_uring: \$(cat /proc/sys/kernel/io_uring_disabled 2>/dev/null && echo enabled)'
    echo ''
    $CMD
  "
