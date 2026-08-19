# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM ghcr.io/astral-sh/uv:0.11.8@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 AS uv

FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime@sha256:db80a41f8428644cebcb3d75b0b62df334ab6c0e75785951eb25f48bfbd42407

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PATH="/opt/document-ocr-venv/bin:${PATH}" \
    PYTHONPATH="/opt/document-ocr/src"

WORKDIR /opt/document-ocr

COPY pyproject.toml uv.lock README.md ./

# The base image supplies its CUDA-matched PyTorch build. The isolated environment
# inherits only that system Torch, while every other runtime package is installed
# from the lock with hashes. The base image's unreferenced scientific-library
# developer CLI conflicts with current Click, so it is pruned from this runtime image.
# pip's checker evaluates venv and inherited distributions together; uv's checker
# currently ignores the inherited CUDA-matched Torch metadata.
# Source is imported from the immutable image copy.
RUN uv pip uninstall --system --break-system-packages spin \
    && uv venv --system-site-packages --python /usr/bin/python /opt/document-ocr-venv \
    && uv export --quiet --frozen --group train --no-dev --prune torch --no-emit-project \
      --format requirements.txt --output-file /tmp/train-requirements.txt \
    && uv pip install --python /opt/document-ocr-venv/bin/python \
      --no-deps --require-hashes \
      --requirements /tmp/train-requirements.txt \
    && python -m pip check

COPY src ./src
COPY docker/training/verify_environment.py ./docker/training/verify_environment.py

RUN python docker/training/verify_environment.py

ENTRYPOINT ["python", "-m", "document_ocr.training.cli"]
CMD ["--help"]
