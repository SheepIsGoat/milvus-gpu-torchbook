ARG TAG=2.0.0-cuda11.8-ubuntu22.04
ARG BASE_IMAGE=anibali/pytorch:$TAG
FROM ${BASE_IMAGE} as builder

COPY requirements.txt /tmp/

USER root

RUN apt-get update && apt-get install -y python3-pip graphviz \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt \
    && python -m spacy download en_core_web_sm

# COPY . /app/
# RUN rm -f requirements.txt

ENV SHELL=/bin/bash

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888