#!/bin/bash
PY_FILE='cuda-checker/which_cuda.py'

echo Checking Container GPUs
docker run --rm -it --init \
    --gpus=all \
    --ipc=host \
    --user="$(id -u):$(id -g)" \
    --volume="$PWD:/app" \
    anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04 \
    python3 $PY_FILE 
    
echo 

echo Checking host GPUs
python3 $PY_FILE
