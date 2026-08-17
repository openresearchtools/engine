FROM ubuntu:24.04 AS build

ARG BACKEND=vulkan
ARG PACKAGE_VERSION=0.1.0
ARG CUDA_APT_REPO=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
ARG CUDA_SERIES=13-2
ARG CUDA_DOT=13.2

ENV DEBIAN_FRONTEND=noninteractive
ENV CC=gcc
ENV CXX=g++
ENV CUDAHOSTCXX=g++
ENV CUDAToolkit_ROOT=/usr/local/cuda-${CUDA_DOT}
ENV PATH=/root/.cargo/bin:/usr/local/cuda-${CUDA_DOT}/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      file \
      git \
      gnupg \
      build-essential \
      cmake \
      ninja-build \
      meson \
      pkg-config \
      python3 \
      patchelf \
      dpkg-dev \
      xz-utils \
      libssl-dev \
      libgomp1 \
      libvulkan-dev \
      libvulkan1 \
      mesa-vulkan-drivers \
      glslc \
      spirv-headers \
    && if [ "$BACKEND" = cuda ]; then \
      curl --retry 5 --connect-timeout 20 --max-time 120 -fsSL \
        "$CUDA_APT_REPO/3bf863cc.pub" | gpg --dearmor -o /usr/share/keyrings/nvidia-cuda-archive-keyring.gpg; \
      printf 'deb [signed-by=/usr/share/keyrings/nvidia-cuda-archive-keyring.gpg] %s /\n' \
        "$CUDA_APT_REPO" > /etc/apt/sources.list.d/nvidia-cuda.list; \
      apt-get update; \
      apt-get install -y --no-install-recommends \
        "cuda-nvcc-$CUDA_SERIES" \
        "cuda-cudart-dev-$CUDA_SERIES" \
        "libcublas-dev-$CUDA_SERIES"; \
    fi \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 --retry 5 -fsSL https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /workspace/engine
COPY . .
RUN chmod +x build/linux/*.sh \
    && build/linux/build_engine_deb.sh \
      --backend "$BACKEND" \
      --version "$PACKAGE_VERSION" \
      --build-root "/workspace/ENGINEbuilds/linux-$BACKEND" \
    && mkdir -p /workspace/ENGINEbuilds/container-output \
    && if [ "$BACKEND" = vulkan ]; then \
      cp /workspace/ENGINEbuilds/linux-vulkan/packages/engine-amd64.deb \
        /workspace/ENGINEbuilds/container-output/engine-amd64.deb; \
    else \
      cp /workspace/ENGINEbuilds/linux-cuda/packages/engine-amd64-cuda.deb \
        /workspace/ENGINEbuilds/container-output/engine-amd64-cuda.deb; \
    fi

FROM scratch AS artifact
COPY --from=build /workspace/ENGINEbuilds/container-output/ /
