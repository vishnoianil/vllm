FROM docker.io/nvidia/cuda:12.9.0-devel-ubi9
RUN dnf install -y vim python3.12 python3.12-pip python3.12-devel wget git && \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y ccache

RUN nvcc --version
WORKDIR /src
RUN pip3.12 install --upgrade pip
RUN pip3.12 install uv
# Create virtual env
RUN uv venv venv --python 3.12 --seed
#ENV PATH="/src/venv/bin:/src/venv/lib64/python3.12/site-packages/nvidia/cu13/bin:/usr/local/cuda/bin:$PATH"
ENV VIRTUAL_ENV=/src/venv
ENV UV_LINK_MODE=copy
#RUN --mount=type=cache,target=/root/.cache/uv \
#    uv pip install torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu130
#ENV CUDA_HOME=/usr/local/cuda
ENV VLLM_REPO=https://github.com/vishnoianil/vllm.git
ENV VLLM_BRANCH=quant-gemm-cutedsl

# Download and build vLLM from HEAD using existing torch
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone -b ${VLLM_BRANCH} --single-branch ${VLLM_REPO} && \
    cd vllm && \
    MAX_JOBS=16 TORCH_CUDA_ARCH_LIST='12.0' uv pip install -e .

# Start the container with an interactive shell
CMD ["/bin/bash"]