FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates libopenblas-dev liblapack-dev \
    libsndfile1 ffmpeg libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r /workspace/requirements.txt

COPY . /workspace
ENV PYTHONPATH=/workspace/src:$PYTHONPATH
ENV TZ=UTC
RUN mkdir -p /workspace/logs /workspace/checkpoints

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN useradd -ms /bin/bash qfl && chown -R qfl:qfl /workspace
USER qfl

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
