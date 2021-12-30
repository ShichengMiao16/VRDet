ARG PYTORCH="1.3.1"
ARG CUDA="9.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install VRDet
RUN conda clean --all
RUN git clone https://github.com/ShichengMiao16/VRDet.git /VRDet
WORKDIR /VRDet
ENV FORCE_CUDA="1"
RUN pip install cython --no-cache-dir
RUN pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
RUN pip install --no-cache-dir -e .
