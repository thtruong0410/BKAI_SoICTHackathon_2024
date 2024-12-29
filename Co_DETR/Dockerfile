FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# Set environment variables for CUDA and PyTorch
ENV CUDA_HOME=/usr/local/cuda-11.3/ \
    TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"


RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /base
COPY requirements.txt /base/requirements.txt
RUN pip install -r requirements.txt
# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime

# Update system and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip cmake wget llvm \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx python3-magic \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install essential Python packages
RUN python3 -m pip install --no-cache-dir --upgrade pip wheel setuptools

# Install PyTorch and torchvision (compatible with CUDA 11.3)
RUN pip3 install --no-cache-dir torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

# Install additional Python dependencies
RUN pip3 install --no-cache-dir \
    terminaltables==3.1.10 \
    pycocotools==2.0.7 \
    fairscale==0.4.13 \
    timm==0.9.16 \
    yapf==0.40.2 \
    scipy==1.10.1 \
    ensemble-boxes==1.0.9 \
    python-magic \
    einops \
    fvcore \
    mmengine
RUN pip install yapf==0.40.1
# Install MMCV (for CUDA 11.3 and PyTorch 1.11.0)
RUN pip3 install --no-cache-dir openmim \
    && mim install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# Cleanup unnecessary files
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Default command (optional, replace with your own script)
# CMD bash train.sh
