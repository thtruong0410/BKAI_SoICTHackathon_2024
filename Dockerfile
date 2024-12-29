# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for CUDA and PyTorch
ENV CUDA_HOME=/usr/local/cuda-11.3/ \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1" \
    TZ=Asia/Ho_Chi_Minh


ENV TZ=Asia/Ho_Chi_Minh \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install tzdata
# Set the timezone
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create symlinks for Python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    cmake \
    wget \
    llvm \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3-magic \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
# Upgrade pip and install dependencies
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools && \
    pip3 install --no-cache-dir \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip3 install --no-cache-dir \
    terminaltables==3.1.10 \
    pycocotools==2.0.7 \
    fairscale==0.4.13 \
    timm==0.9.16 \
    yapf==0.40.1 \
    scipy==1.10.1 \
    ensemble-boxes==1.0.9 \
    python-magic \
    einops \
    fvcore \
    mmengine \
    ultralytics \
    mmdet==2.25.3 \
    openmim

# Install MMCV
RUN mim install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Optional: Install requirements file (if needed)


# Optional: Cleanup (uncomment if needed)
# RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Default command (replace with your own if needed)
# CMD ["/bin/bash"]