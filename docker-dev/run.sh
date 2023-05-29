# run from docker-base directory
docker run \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -it \
    docker-dev-base \
    bash
