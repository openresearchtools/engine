FROM ubuntu:24.04 AS build

ARG PACKAGE_VERSION=0.1.0
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/.cargo/bin:${PATH}
ENV CARGO_HOME=/workspace/ENGINEbuilds/cargo-home
ENV CARGO_BUILD_JOBS=4
ENV BUILD_JOBS=4

RUN test "$(dpkg --print-architecture)" = arm64 \
    && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl file git build-essential cmake ninja-build meson \
      pkg-config python3 patchelf dpkg-dev xz-utils libssl-dev libgomp1 \
      libvulkan-dev libvulkan1 mesa-vulkan-drivers glslc spirv-headers \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 --retry 5 -fsSL https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable

ENV PATH=/workspace/ENGINEbuilds/cargo-home/bin:${PATH}
WORKDIR /workspace/engine
COPY . .
RUN build/linux/build_engine_deb.sh \
      --backend vulkan --version "$PACKAGE_VERSION" \
      --build-root /workspace/ENGINEbuilds/linux-arm64-vulkan \
    && mkdir -p /workspace/ENGINEbuilds/container-output \
    && cp /workspace/ENGINEbuilds/linux-arm64-vulkan/packages/engine-arm64.deb \
      /workspace/ENGINEbuilds/container-output/

FROM scratch AS artifact
COPY --from=build /workspace/ENGINEbuilds/container-output/ /
