FROM docker.io/nvidia/cuda:13.1.0-devel-ubi9
RUN dnf install -y vim python3.12 python3.12-pip python3.12-devel wget git && \
    dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm && \
    dnf install -y ccache

RUN nvcc --version
WORKDIR /src
RUN pip3.12 install --upgrade pip
RUN pip3.12 install uv
# Create virtual env
RUN uv venv venv --python 3.12 --seed
ENV PATH="/src/venv/bin:$PATH"
ENV VIRTUAL_ENV=/src/venv
ENV UV_LINK_MODE=copy
ENV VLLM_REPO=https://github.com/vishnoianil/vllm.git
ENV VLLM_BRANCH=vllm-cutile

# Download and build vLLM from HEAD using existing torch
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone ${VLLM_REPO} && \
    cd vllm && \
    git checkout ${VLLM_BRANCH} && \
    MAX_JOBS=16 TORCH_CUDA_ARCH_LIST='10.0+PTX' uv pip install -e .

# Start the container with an interactive shell
CMD ["/bin/bash"]