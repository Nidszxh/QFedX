# Dockerfile - QFL HPC image (CPU). For GPU replace base image with CUDA-enabled base and install torch with CUDA.
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps (OpenMPI not required for Flower but useful for MPI fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates libopenblas-dev liblapack-dev \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /workspace/requirements.txt

# Copy project
COPY . /workspace
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV TZ=UTC
RUN mkdir -p /workspace/logs

# Non-root run user (optional)
RUN useradd -ms /bin/bash qfl && chown -R qfl:qfl /workspace
USER qfl

# Entrypoint is chosen at runtime; typically we'll run:
# python fl_server.py  OR python fl_client.py
CMD ["bash", "-lc", "sleep inf"]
COPY entrypoint.sh /entrypoint.sh
# ENTRYPOINT ["/entrypoint.sh"]