# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

FROM ghcr.io/astral-sh/uv:0.11.8@sha256:3b7b60a81d3c57ef471703e5c83fd4aaa33abcd403596fb22ab07db85ae91347 AS uv

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7

COPY --from=uv /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_PROGRESS=1 \
    PATH="/opt/synthesis-venv/bin:${PATH}" \
    PYTHONPATH="/opt/document-ocr/src"

WORKDIR /opt/document-ocr

COPY environments/synthesis/pyproject.toml environments/synthesis/uv.lock ./environments/synthesis/

RUN uv sync --project environments/synthesis --frozen --no-dev \
    && ln -s /opt/document-ocr/environments/synthesis/.venv /opt/synthesis-venv

COPY src ./src

RUN python -c "import sdv; from document_ocr.synthesis.config import SynthesisFoundationConfig"

ENTRYPOINT ["python", "-m", "document_ocr.synthesis.cli"]
CMD ["--help"]
