FROM cr.weaviate.io/semitechnologies/reranker-transformers:custom

WORKDIR /app

# HuggingFace model repo for reranking (multilingual, CPU-friendly).
# This is a multilingual cross-encoder trained on mMARCO.
ARG MODEL_NAME=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

RUN MODEL_NAME="${MODEL_NAME}" ./download.py
