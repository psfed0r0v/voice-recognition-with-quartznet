#!/bin/bash
image_name=asr_client
container_name=container-${image_name}

docker stop "${container_name}"
docker rm "${container_name}"

docker build -t "$image_name":latest ../asr_client/.
docker run --name "${container_name}" -p 80:80 --network=host ${image_name}
