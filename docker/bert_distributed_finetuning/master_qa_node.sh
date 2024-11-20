#!/bin/bash
#
# Copyright (c) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



function usage_help() {
    echo -e "options:"
    echo -e "  -h Display help"
    echo -e "  -i {image_id}"
    echo -e "  -m {model_name}"
    echo -e "  -p {total process}"
    echo -e "  -n {docker network name}"
    echo -e "  -o: enable offline mode"
}


# Set env param
master_port=29500
ccl_worker_count=1

# Start training BERT
mpi_n=4
mpi_ppn=2
omp_num_threads=24
model_name=bert-large-uncased
dataset_name=squad
per_device_train_batch_size=12
learning_rate=3e-5
num_train_epochs=2
max_seq_length=384
doc_stride=128
output_dir="./tmp/debug_squad/"
cache_dir="./tmp/cache"   # you could put the download to /tmp/cache and then enable offline mode
xpu_backend=ccl
dataloader_pin_memory=False
bf16=False
use_ipex=True
offline=False
# Override args
while getopts "h?r:i:m:p:n:o" OPT; do
    case $OPT in
        h|\?)
            usage_help
            exit 0
            ;;
        i)
            echo -e "Option $OPTIND, image_id = $OPTARG"
            image_id=$OPTARG
            ;;
        m)
            echo -e "Option $OPTIND, model_name = $OPTARG"
            model_name=$OPTARG
            ;;
        p)
            echo -e "Option $OPTIND, pn = $OPTARG"
            mpi_n=$OPTARG
            ;;
        n)
            echo -e "Option $OPTIND, network = $OPTARG"
            network=$OPTARG
            ;;
        o)
            echo -e "enable offline mode"
            offline=True
            ;;
        ?)
            echo -e "Unknown option $OPTARG"
            usage_help
            exit 0
            ;;
    esac
done

if [ ! $image_id ];then
   echo -e "no docker image id indication"
   exit 0
fi

if [ ! $network ];then
   echo -e "no docker network indication"
   exit 0
fi

master_addr=master
docker run --name master -h master --net $network  \
    --privileged --shm-size 800g \
    -v /tmp/:/usr/local/tmp/ \
    -e master_node=True \
    -e learning_rate=${learning_rate} \
    -e max_seq_length=${max_seq_length} \
    -e dataloader_pin_memory=${dataloader_pin_memory} \
    -e model_name=${model_name} \
    -e output_dir=${output_dir} \
    -e mpi_n=${mpi_n} \
    -e mpi_ppn=${mpi_ppn} \
    -e omp_num_threads=${omp_num_threads} \
    -e per_device_train_batch_size=${per_device_train_batch_size} \
    -e learning_rate=${learning_rate} \
    -e num_train_epochs=${num_train_epochs} \
    -e xpu_backend=${xpu_backend} \
    -e doc_stride=${doc_stride}\
    -e ccl_worker_count=${ccl_worker_count} \
    -e master_addr=${master_addr} \
    -e master_port=${master_port} \
    -e dataset_name=${dataset_name} \
    -e use_ipex=${use_ipex} \
    -e bf16=${bf16} \
    -e cache_dir=${cache_dir} \
    -e offline=${offline} \
    ${image_id}

