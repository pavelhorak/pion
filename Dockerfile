# syntax=docker/dockerfile:1
# Pion — shippable runtime image (gh #108).
#
# Multi-arch: linux/amd64 and linux/arm64.
#   docker buildx build --platform linux/amd64,linux/arm64 -t pion .
#   docker run -p 1974:1974 -v pion-data:/data pion
#
# GPU-free by design: amd64 uses the portable `build-portable` task
# (--target-cpu x86-64-v2 — the ISA floor that runs on any x86-64 host;
# an unpinned Mojo build bakes in the build host's ISA and SIGILLs
# elsewhere). arm64 uses the default GPU-free linux-aarch64 build.
# CUDA images are a separate lane (requires nvcc; see pixi.toml
# [target.linux-64.tasks].build).
#
# io_uring note: Docker's default seccomp profile blocks io_uring syscalls
# (Docker ≥ 25). The default CMD therefore uses --epoll. For io_uring
# performance, run with:
#   docker run --security-opt seccomp=unconfined pion --no-auto-embed
# (no --epoll → Pion auto-selects io_uring on Linux).

FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.pixi/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# gh #417 (I-15): pin the pixi version so the image build is hermetic — the
# installer otherwise serves whatever is latest on build day, while pixi.lock
# was written by a specific pixi, and a lock-format change breaks the build
# with no code change.
ENV PIXI_VERSION=v0.70.2
RUN curl -fsSL https://pixi.sh/install.sh | bash

WORKDIR /pion

# Manifest first: the pixi env layer (Mojo toolchain download) only
# rebuilds when the manifest changes, not on every source edit.
COPY pixi.toml pixi.lock ./
RUN pixi install

COPY . .

ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        pixi run build-portable && mv pion-server-dev pion-server; \
    else \
        pixi run build; \
    fi

# Collect the Mojo runtime libraries the binary links at run time.
#
# DERIVED FROM THE BINARY (scripts/collect_runtime_libs.sh), never hardcoded.
# The list that used to live here named libAsyncRTMojoBindings, which Mojo 1.0
# dropped — and its `|| true` meant the failure mode ran the wrong way: a
# library the binary genuinely needs could go missing and still produce an
# image that builds clean, then dies at `docker run` on a stranger's machine.
# The script exits non-zero on a missing dep, so that is a build failure here.
RUN mkdir -p /out/bin /out/lib && \
    cp pion-server /out/bin/ && \
    bash scripts/collect_runtime_libs.sh pion-server .pixi/envs/default/lib /out/lib

# ── Fast path: wrap an ALREADY-BUILT host binary (seconds, not ~35 min) ──
#   pixi run build   (or download the S3 release tarball and untar)
#   docker build --target runtime-prebuilt -t pion:dev .
# Uses ./pion-server + .pixi runtime libs from the build context. For local/
# private images only — published release images use the from-source default
# target (hermetic provenance).
FROM ubuntu:24.04 AS prebuilt-collect
WORKDIR /pion
COPY pion-server /out/bin/pion-server
COPY scripts/collect_runtime_libs.sh /usr/local/bin/collect_runtime_libs.sh

# Every lib arrives through a BRACKET GLOB, in ONE COPY, into a staging dir.
# Both details are load-bearing, and this target was broken by getting each of
# them wrong:
#
#   - A LITERAL COPY of a path the toolchain no longer ships is a hard build
#     failure. This named libAsyncRTMojoBindings.so, Mojo 1.0 stopped shipping
#     it, and `--target runtime-prebuilt` — the README's "seconds instead of
#     ~35 minutes" path — could not build at all.
#   - The bracket-glob "optional COPY" idiom only tolerates a miss while
#     ANOTHER source in the SAME COPY matches; a COPY whose every pattern
#     matches nothing is an error. libNVPTX and libcudart each sat on a line of
#     their own, so on a GPU-free box they took the build down instead of being
#     skipped. Here they ride along with libKGENCompilerRTShared, always present.
#
# Absence is therefore tolerated at COPY time, and the real check happens below
# against the binary itself.
#
# cudart: CUDA-linked host binaries (the default linux-64 build) need it —
# stage it first with
#   cp /usr/local/cuda/lib64/libcudart.so.12 .pixi/envs/default/lib/
# GPU-free binaries (build-portable, macOS-style) skip it. It is passed through
# rather than derived: the shared derivation ships Mojo runtime libs only, so
# bundling an NVIDIA redistributable stays an explicit act. gh #108.
COPY .pixi/envs/default/lib/libKGENCompilerRTShared.s[o] \
     .pixi/envs/default/lib/libAsyncRTMojoBindings.s[o] \
     .pixi/envs/default/lib/libMSupportGlobals.s[o] \
     .pixi/envs/default/lib/libAsyncRTRuntimeGlobals.s[o] \
     .pixi/envs/default/lib/libNVPTX.s[o] \
     .pixi/envs/default/lib/libcudart.so.1[2] \
     /ctx-lib/

# Ask the binary which of the staged libs it actually needs, and fail the build
# if one is absent — the whole point of a prebuilt image is that it runs, and
# `exit 127` on someone else's box is the most expensive place to find out.
# That 127 is not hypothetical: it is how the missing-cudart bug was found, so
# cudart gets the same treatment as the Mojo libs even though it is bundled by
# hand — needed-and-absent must fail HERE, not at `docker run`.
RUN mkdir -p /out/lib && \
    bash /usr/local/bin/collect_runtime_libs.sh /out/bin/pion-server /ctx-lib /out/lib && \
    if ldd /out/bin/pion-server 2>/dev/null | awk '{print $1}' | grep -q '^libcudart'; then \
        cp /ctx-lib/libcudart.so.12 /out/lib/ 2>/dev/null \
          || { echo "  MISSING: libcudart.so.12 (binary is CUDA-linked; stage it into .pixi/envs/default/lib first)"; exit 1; }; \
        echo "  bundled: libcudart.so.12"; \
    fi

FROM ubuntu:24.04 AS runtime-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /data pion

# gh #412 (I-5): ship the licence and attribution text in the image. liblua.a is
# statically linked (MIT requires the notice in all copies) and the image
# redistributes the Modular runtime dylibs; NOTICE reproduces both verbatim.
COPY LICENSE NOTICE /opt/pion/

# OCI image metadata so `docker inspect` / registries show the source and licence.
LABEL org.opencontainers.image.title="Pion" \
      org.opencontainers.image.description="Deterministically low-latency KV + vector database, wire-compatible with Redis/Valkey" \
      org.opencontainers.image.source="https://github.com/pavelhorak/pion" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.documentation="https://github.com/pavelhorak/pion/blob/main/README.md"

ENV LD_LIBRARY_PATH=/opt/pion/lib
# WAL / snapshots / blob arenas land in the working directory.
WORKDIR /data
VOLUME /data

# 1974 RESP; 1975 binary lane (0xCA5E) when --kvcache is used.
EXPOSE 1974 1975

HEALTHCHECK --interval=15s --timeout=3s --start-period=60s \
    CMD bash -c 'exec 3<>/dev/tcp/127.0.0.1/1974' || exit 1

ENTRYPOINT ["/opt/pion/bin/pion-server"]
# --epoll: io_uring is seccomp-blocked under Docker defaults (see header).
# --no-auto-embed: the slim image ships no Python/torch; embedding-model
# features need the ai image variant or a host install.
# --bind 0.0.0.0 (gh #258): the server now defaults to loopback, and a
# loopback bind inside a container is unreachable through `docker run -p` —
# the port forward arrives on the container's eth0, not its lo. Binding all
# interfaces INSIDE the netns is the normal container posture: what is exposed
# to the host is decided by `-p`, not by this. Note the healthcheck above
# connects via 127.0.0.1 and works either way.
CMD ["--epoll", "--no-auto-embed", "--bind", "0.0.0.0"]

# Fast local target: docker build --target runtime-prebuilt -t pion:dev .
FROM runtime-base AS runtime-prebuilt
COPY --from=prebuilt-collect --chown=root:root /out/bin /opt/pion/bin
COPY --from=prebuilt-collect --chown=root:root /out/lib /opt/pion/lib
USER pion

# Default (last stage = default target): hermetic from-source release image.
FROM runtime-base
COPY --from=builder /out/bin /opt/pion/bin
COPY --from=builder /out/lib /opt/pion/lib
USER pion
