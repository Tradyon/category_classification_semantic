FROM cr.weaviate.io/semitechnologies/transformers-inference:custom

WORKDIR /app

# HuggingFace model repo for embeddings (E5).
ARG MODEL_NAME=intfloat/multilingual-e5-base

RUN MODEL_NAME="${MODEL_NAME}" ./download.py
