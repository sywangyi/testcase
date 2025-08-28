#!/bin/bash

# Default variable values
device=xpu
target=base
mount_dir=${PWD}

# Function to display script usage
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo " -h, --help            Display this help message"
  echo " -d, --device          Hardware Device[cpu, xpu, cuda]"
  echo " -t, --target          Target Name[base, ut]"
  echo " -v, --mount_dir       Local directory to be mounted inside the container; default value is the current directory"
  echo " -n, --name            Container name"
}

has_argument() {
  [[ ("$1" == *=* && -n ${1#*=}) || (! -z "$2" && "$2" != -*) ]]
}

extract_argument() {
  echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
  while [ $# -gt 0 ]; do
    case $1 in
    -h | --help)
      usage
      exit 0
      ;;
    -d | --device)
      if ! has_argument $@; then
        echo "Device name not specified." >&2
        usage
        exit 1
      fi

      device=$(extract_argument $@)

      shift
      ;;
    -t | --target)
      if ! has_argument $@; then
        echo "Target image stage not specified" >&2
        usage
        exit 1
      fi

      target=$(extract_argument $@)

      shift
      ;;
    -v | --mount_dir)
      mount_dir=$(extract_argument $@)
      shift
      ;;
    -n | --name)
      name=$(extract_argument $@)
      shift
      ;;
    *)
      echo "Invalid option: $1" >&2
      usage
      exit 1
      ;;
    esac
    shift
  done
}

# Main script execution
handle_options "$@"

if [[ -z "${name}" ]]; then
  name=hf-${device}-${target}
fi

if [[ -z "${HF_HOME}" ]]; then
  HF_HOME=$HOME/.cache/huggingface
fi

if [[ $device == "cuda" ]]; then
  docker run -it \
    --privileged \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e no_proxy=${no_proxy} \
    -v ${HF_HOME}/hub:/root/.cache/huggingface/hub \
    -v ${mount_dir}:/workspace \
    -v /dev/shm:/dev/shm \
    -p 8888:8888 \
    --runtime=nvidia \
    --gpus all \
    --name ${name} \
    appliedml/huggingface:${device}-${target}
elif [[ $device == "xpu" ]]; then
  docker run -it \
    --privileged \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e no_proxy=${no_proxy} \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v ${HF_HOME}/hub:/root/.cache/huggingface/hub \
    -v ${mount_dir}:/workspace \
    -p 8888:8888 \
    --device=/dev/dri \
    --ipc=host \
    --name ${name} \
    appliedml/huggingface:${device}-${target}
elif [[ $device == "cpu" ]]; then
  docker run -it \
    --privileged \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e no_proxy=${no_proxy} \
    -v ${HF_HOME}/hub:/root/.cache/huggingface/hub \
    -v ${mount_dir}:/workspace \
    --ipc=host \
    --name ${name} \
    appliedml/huggingface:${device}-${target}
elif [[ $device == "hpu" ]]; then
  docker run -it \
    --privileged \
    -e http_proxy=${http_proxy} \
    -e https_proxy=${https_proxy} \
    -e no_proxy=${no_proxy} \
    -e HABANA_VISIBLE_DEVICES=all \
    -v ${HF_HOME}/hub:/root/.cache/huggingface/hub \
    -v ${mount_dir}:/workspace \
    --ipc=host \
    --name ${name} \
    appliedml/huggingface:${device}-${target}

else
  echo "the specified device is not supported"
fi
