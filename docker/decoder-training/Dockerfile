# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM ghcr.io/astral-sh/uv:0.11.8@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 AS uv

FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime@sha256:bfbb4a2b4fdba0fefdb428ea737e626d61bb3daf74a16e1ff935bdb03aa7c3f0

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PATH="/opt/decoder-venv/bin:${PATH}" \
    PYTHONPATH="/opt/document-ocr/src" \
    HF_HOME="/cache/huggingface" \
    TOKENIZERS_PARALLELISM="false"

WORKDIR /opt/document-ocr

COPY environments/decoder-training/pyproject.toml environments/decoder-training/uv.lock ./environments/decoder-training/

# CUDA-matched Torch comes from the immutable base image. Everything else is
# installed from the independent lock without allowing uv to replace that build.
RUN uv venv --system-site-packages --python /usr/bin/python3 /opt/decoder-venv \
    && uv export --quiet --project environments/decoder-training --frozen --no-dev \
      --prune torch --format requirements.txt --output-file /tmp/decoder-requirements.txt \
    && uv pip install --python /opt/decoder-venv/bin/python --no-deps --require-hashes \
      --requirements /tmp/decoder-requirements.txt \
    && /opt/decoder-venv/bin/python -m pip check

COPY src ./src
COPY docker/decoder-training/verify_environment.py ./docker/decoder-training/verify_environment.py

RUN python docker/decoder-training/verify_environment.py

ENTRYPOINT ["python", "-m", "document_ocr.decoder_training.cli"]
CMD ["--help"]
