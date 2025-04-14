FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/SWivid/F5-TTS"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && apt-get install -y librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
WORKDIR /workspace

RUN git clone https://github.com/SWivid/F5-TTS.git \
    && cd F5-TTS \
    && git submodule update --init --recursive \
    && pip install -e . --no-cache-dir \
    && pip install fastapi uvicorn python-multipart

# Create a FastAPI application file
COPY src/f5_tts/fastapi_server.py /workspace/F5-TTS/src/f5_tts/

ENV SHELL=/bin/bash

VOLUME /root/.cache/huggingface/hub/

# Change port from 7860 (Gradio) to 8000 (FastAPI standard)
EXPOSE 8000

WORKDIR /workspace/F5-TTS

# Set the entrypoint to run Uvicorn with your FastAPI app
ENTRYPOINT ["uvicorn", "f5_tts.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
