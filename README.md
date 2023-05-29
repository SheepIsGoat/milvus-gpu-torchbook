# Milvus GPU Torchbook
## What is Milvus
Milvus is the largest open-source vector database, helping power your machine-learning and search projects with a fast and reliable data store. Common uses include advanced-search (semantic, reverse-image, knowledgebase, audio, molecular, etc.), recommenders, clustering, and classificaiton.

GPU support was recently released in Milvus 2.3.0-beta and allows ~10x improvements in processing speeds.

## What does this repo do?
Run a persistent milvus database alongside jupyter notebooks, all powered by CUDA/NVIDIA GPU acceleration in a fully containerized environment. 

This was made for ubuntu22.04 and tested on 4000 series NVIDIA GPUs, but can be easily extensible to other configurations. If you would like additional support added, please let me know or feel free to contribute.

# Check CUDA compatibility

## Get CUDA Working on Your Host
You'll need CUDA, and CUDA only works with NVIDIA graphics cards, on which you must first install the proper drivers. If you do not have a NVIDIA graphics card or don't want to get CUDA working now, then skip the later jupyter section, and do not try to run milvus on GPU.

### <b>Check your version of CUDA</b>, and make sure these don't throw errors.
```
nvidia-smi
nvcc --version
````

If either one does not work, you may either have a broken installation or a missing installation you will have to fix.

### <b>To Install CUDA</b> 

Go to https://developer.nvidia.com/cuda-downloads, set the options, and copy the commands they give you. This may take many hours to work through if there are issues - CUDA drivers can be very difficult on linux, but hang in there.

## Check that CUDA works in a Container
1. Make sure the output of the following command is the same as the output of `nvidia-smi` from your host.

    `docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi`



2. Make sure pytorch can access your GPUs from a container:
    ```
    docker run --rm -it --init \
        --gpus=all \
        --ipc=host \
        --user="$(id -u):$(id -g)" \
        --volume="$PWD:/app" \
        anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04 \
        python3 cuda-checker/which_cuda.py
    ```


If you want to exec into the container with an interactive shell

```
docker run -it --gpus all anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04 bash
```

# Use Jupyter Container
Build the pytorch-cuda-jupyter image
```
sh pytorch-jupyter/docker/build.sh
```

NOTE: if you're skipping CUDA, you'll have to change `$BASE` in the Dockerfile and add `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` to the run command

First do to the directory and make your script executable
```
cd pytorch-jupyter && chmod ug=rwx run.sh
```
Then run it
```
./run.sh
```

# Run Milvus
Decide which stack you want to use. The intention of this repo is to make working with a GPU milvus stack easy, so I use the `...gpu.yml` files. The `standalone` stack are much easier for development, but for production workloads you may want to use a `cluster` or consider a managed solution.

### Start your stack

```
docker-compose up -f milvus-compose/milvus-standalone-docker-compose-gpu.yml -d
```

You can now connect to milvus via your jupyter notebook, or continue reading for notes on connecting via the Attu GUI.

### Explore via the Attu GUI

Go to http://localhost:8000 in your browser. You will be asked for a Milvus Address to connect to, which will require your local ip.

Find your local host ip using
```
hostname -I
# example output:
# 192.168.1.78 172.17.0.1 172.19.0.1   # along with 3 public IPv6 addresses
```

Based on the above output, in the Attu UI I would set Milvus Address to <b>`172.19.0.1:19530`</b>

After that, you should be able to connect to and manipulate the milvus database via your browser session

# Aknowledgements
Thanks to Anibali for making the base pytorch image

https://github.com/anibali/docker-pytorch