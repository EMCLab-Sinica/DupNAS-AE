FROM python:3.9-slim

WORKDIR /workspace/DupNAS-AE

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_base.txt .


# Install the required CUDA 11.5 PyTorch build
RUN python3.9 -m pip install --no-cache-dir \
    torch==1.11.0+cu115 \
    torchvision==0.12.0+cu115 \
    torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu115

# Install the remaining dependencies without reinstalling torch
RUN grep -v '^torch==' requirements_base.txt > requirements-docker.txt && \
    grep -v '^torchvision==' requirements-docker.txt > requirements-docker2.txt && \
    grep -v '^torchaudio==' requirements-docker2.txt > requirements-final.txt && \
    python3.9 -m pip install --no-cache-dir -r requirements-final.txt


RUN apt-get update && \
    apt-get install -y --no-install-recommends patchelf && \
    LIB="$(find /usr/local/lib/python3.9/site-packages/torch -name 'libtorch_cpu.so' -print -quit)" && \
    test -n "$LIB" && \
    echo "Clearing executable-stack flag from: $LIB" && \
    patchelf --clear-execstack "$LIB" && \
    rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["bash"]