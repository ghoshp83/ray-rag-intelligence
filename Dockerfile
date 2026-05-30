FROM python:3.10-slim

WORKDIR /app

# build-essential covers source builds for any wheel without a slim-musl binary.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (cached) using only the metadata + package source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY data ./data

ENV PYTHONUNBUFFERED=1
