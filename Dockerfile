# Adjust CUDA image depending on your machine
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-distutils \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9-10 1

# Install Xvfb and other dependencies
RUN apt-get update && apt-get install -y \
    xvfb freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

RUN apt-get update && apt-get install -y \
    wget unzip \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.roboti.us/download/mujoco200_linux.zip && unzip mujoco200_linux.zip && \
        mkdir .mujoco && \
        mv mujoco200_linux .mujoco/mujoco200
RUN wget https://www.roboti.us/file/mjkey.txt && \
        mv mjkey.txt .mujoco/mujoco200/bin

ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/app/.mujoco/mujoco200/bin"

# Copy the Python script to the container
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt
