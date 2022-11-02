#!/bin/bash
docker pull nvcr.io/nvidia/tritonserver:21.09-py3
docker build . --rm -f ../triton/Dockerfile -t quartznet-asr-tritonserver

SCRIPT_DIR=$(cd $(dirname $0); pwd)
PROJECT_DIR=${SCRIPT_DIR}/..
HOST_REPO=${PROJECT_DIR}/triton/model_repository/
echo ${HOST_REPO}
docker run -p8000:8000 -p8001:8001 -p8002:8002 -v/${HOST_REPO}:/models --rm quartznet-asr-tritonserver tritonserver --model-repository=/models --exit-on-error=false
