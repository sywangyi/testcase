#!/bin/bash

# Default variable values
device=xpu
target=base
use_no_proxy=false # Default to not use no_proxy

# Function to display script usage
usage() {
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo " -h, --help            Display this help message"
  echo " -d, --device          Hardware Device[cpu, xpu, cuda]"
  echo " -t, --target          Build stage[base, ut]"
  echo " --use-no-proxy        Include no_proxy argument in the build"
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
    -d | --device*)
      if ! has_argument $@; then
        echo "Device name not specified." >&2
        usage
        exit 1
      fi

      device=$(extract_argument $@)

      shift
      ;;
    -t | --target*)
      if has_argument $@; then
        target=$(extract_argument $@)
      fi
      shift
      ;;
    --use-no-proxy)
      use_no_proxy=true
      shift
      ;;
    esac
    shift
  done
}

# Main script execution
handle_options "$@"

# Build the Docker command
docker_command="docker build \
  --no-cache \
  -f Dockerfile.${device} \
  --target ${target} \
  --build-arg http_proxy=${http_proxy} \
  --build-arg https_proxy=${https_proxy}"

# Conditionally add the no_proxy argument
if [ "$use_no_proxy" = true ]; then
  docker_command+=" --build-arg no_proxy=${no_proxy}"
fi

docker_command+=" -t appliedml/huggingface:${device}-${target} ."

# Execute the Docker command
eval $docker_command
