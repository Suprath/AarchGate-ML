FROM --platform=linux/arm64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install base packages and setup Kitware CMake repository
RUN apt-get update && apt-get install -y ca-certificates gpg wget

# Add Kitware CMake repository (for CMake 3.24+)
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null

# Install build tools and development libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    clang-15 \
    lld-15 \
    cmake \
    ninja-build \
    git \
    pkg-config \
    gdb \
    libacl1-dev \
    libncurses5-dev \
    python3.10-dev \
    libpython3.10-dev \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-setuptools \
    openjdk-11-jdk \
    && ln -sf /usr/bin/clang-15 /usr/bin/clang \
    && ln -sf /usr/bin/clang++-15 /usr/bin/clang++

# Set up compiler and build environment
ENV CC=clang
ENV CXX=clang++
ENV LD=lld
ENV CMAKE_BUILD_PARALLEL_LEVEL=4
ENV CMAKE_MAKE_PROGRAM=/usr/bin/ninja
ENV CXXFLAGS="-O3 -ffast-math -fno-finite-math-only"
ENV CFLAGS="-O3 -ffast-math -fno-finite-math-only"

# Install ML requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set workspace
WORKDIR /workspace

# Copy AarchGate-ML (including submodules)
COPY . .

# Build AarchGate (FORCE CLEAN REBUILD)
RUN rm -rf external/AarchGate/build && \
    mkdir -p external/AarchGate/build && \
    cd external/AarchGate/build && \
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DAPEX_BUILD_FLATBUFFERS=OFF .. && \
    ninja aarchgate aarchgate_python

# Set PYTHONPATH to include the built pybind module
ENV PYTHONPATH="/workspace:/workspace/external/AarchGate/build/bindings/python"

CMD ["/bin/bash"]
