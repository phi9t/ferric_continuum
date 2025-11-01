#!/bin/bash
# docker-ci-test.sh - Build and test in CI-like Docker environment

set -eu -o pipefail

# Configuration
readonly IMAGE_NAME="ferric-continuum-ci"
readonly CONTAINER_NAME="ferric-continuum-ci-test"

# Create local cache directories if they don't exist
mkdir -p .docker-cache/bazel
mkdir -p .docker-cache/bazelisk

# Build Docker image
# Note: Using legacy builder (deprecation warning is safe to ignore)
docker build --platform linux/amd64 -f Dockerfile.ci -t "${IMAGE_NAME}" .

# Docker run options
DOCKER_RUN_OPTS=(
    --rm
    --platform linux/amd64
    -v "$(pwd):/workspace"
    -v "$(pwd)/.docker-cache/bazel:/root/.cache/bazel"
    -v "$(pwd)/.docker-cache/bazelisk:/root/.cache/bazelisk"
    -w /workspace
    --name "${CONTAINER_NAME}"
)

# Run command
docker run "${DOCKER_RUN_OPTS[@]}" -it "${IMAGE_NAME}" /bin/bash
