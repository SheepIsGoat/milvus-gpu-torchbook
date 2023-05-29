#!/bin/bash
# Helper scripts for working with Docker image and container.

BASE=$(dirname "$0")

buildBaseImage () {
    ERR_MSG="Please specify 'cuda' or 'nocuda' to build the base image"
    if [ $# -eq 0 ] || [ -z "$1" ]; then
        echo "$ERR_MSG"
        CUDA=$(useCuda)
        echo "CUDA=$CUDA"
        return
    else
        case "$1" in
            [Nn][Oo][Cc][Uu][Dd][Aa]|[Nn][Oo]|[Nn])
                IMAGE="2.0.0-nocuda-ubuntu22.04"
                ;;
            [Cc][Uu][Dd][Aa]|[Cc])
                IMAGE="2.0.0-cuda11.8-ubuntu22.04"
                ;;
            *)
                echo "Sorry, there's no "$1" base image to build"
                echo "$ERR_MSG"
                return
                ;;
        esac
    fi
    echo "Building $IMAGE:latest"
    DIR="$BASE/docker-base/$IMAGE"
    docker build -f "$DIR/Dockerfile" -t "$IMAGE:latest" $DIR
}

buildPipImage () {
    echo "Anabali images can be pulled remotely. If you choose not to use dockerhub images, we'll look for images on your machine"
    echo "Use remote anibali base_image for dev-pip-container?"
    echo "    remote: use remote image"
    echo "    local: use local image"
    read -p "    (remote/local) " response
    case "$response" in
        [Rr][eE][Mm][Oo][Tt][Ee]|[Rr]) 
            BASE_IMG_PREFIX="anibali/pytorch:"
            ;;
        *)
            ;;
    esac
    
    CUDA="$1"
    if [ -z "$CUDA" ] && [ -z "$BASE_IMG_PREFIX" ]; then
        CUDA=$(echo $(useCuda) | awk '{print $NF}')
    fi
    case "$CUDA" in
        [Nn][Oo][Cc][Uu][Dd][Aa]|[Nn][Oo]|[Nn])
            BASE_IMAGE="$BASE_IMG_PREFIX"2.0.0-nocuda-ubuntu22.04
            ;;
        [Cc][Uu][Dd][Aa]|[Cc])
            BASE_IMAGE="$BASE_IMG_PREFIX"2.0.0-cuda11.8-ubuntu22.04
            ;;
        *)
            echo "Sorry, there's no \"$CUDA\" base image to build"
            CUDA=$(echo $(useCuda) | awk '{print $NF}')
            buildPipImage $CUDA
            return
            ;;
    esac
    PREV_PWD="$PWD"
    cd "$BASE"
    cd ..
    echo "Building a pip-dev image with the following packages:"
    echo $(grep -vE "^#|^$" requirements.txt)
    read -p "Continue? (y/n) " response
    case "$response" in
        [yY][eE][sS]|[yY])
            ;;
        *)
            echo "exiting early"
            return
            ;;
    esac

    echo "Building ml-dev-container from BASE_IMAGE=$BASE_IMAGE from dir $PWD"
    docker build \
        --build-arg BASE_IMAGE=$BASE_IMAGE \
        -f "docker/docker-pip/Dockerfile" \
        -t ml-dev-container:latest \
        .
    cd "$PREV_PWD"
}

useCuda () {
    echo "Please specify 'cuda' or 'nocuda' for the base image"
    read -p "cuda or nocuda? (cuda/nocuda) " response
    case "$response" in
        [Nn][Oo][Cc][Uu][Dd][Aa]|[Nn][Oo]|[Nn])
            echo nocuda
            ;;
        [Cc][Uu][Dd][Aa]|[Cc])
            echo cuda
            ;;
        *)
            echo "Sorry, I didn't understand \"$response\""
            echo "$ERR_MSG"
            return
            ;;
    esac
}

# Shows the usage for the script.
showUsage () {
    echo "Description:"
    echo "    Builds, runs and pushes Docker image '$IMAGE_NAME'."
    echo ""
    echo "Options:"
    echo "    base: Builds a base Docker image ('$IMAGE_NAME')."
    echo "              requires second argument 'cuda' or 'nocuda'"
    echo "    pip: Builds an image with pip install -r requirements.txt ('$IMAGE_NAME')."
    echo ""
    echo "Example:"
    echo "    sh docker-task.sh build base cuda"
    echo ""
    echo "    This will:"
    echo "        Build a Docker image named 2.0.0-cuda11.8-ubuntu22.04."
}

if [ $# -eq 0 ]; then
    showUsage
else
    case "$1" in
        "base")
            shift
            buildBaseImage $1
            ;;
        "pip")
            shift
            buildPipImage $1
            ;;
        *)
            showUsage
            ;;
    esac
fi