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
    echo -e "  -n {docker network name}"
    echo -e "  -s {slave_id}"
}

slave_id=0

# Override args
while getopts "h?r:i:n:s:" OPT; do
    case $OPT in
        h|\?)
            usage_help
            exit 0
            ;;
        i)
            echo -e "Option $OPTIND, image_id = $OPTARG"
            image_id=$OPTARG
            ;;
        s)
            echo -e "Option $OPTIND, slave_id = $OPTARG"
            slave_id=$OPTARG
            ;;
        n)
            echo -e "Option $OPTIND, network = $OPTARG"
            network=$OPTARG
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
docker run -d --name slave$slave_id -h slave$slave_id --net $network \
    --privileged --shm-size 800g \
    -v /tmp/:/usr/local/tmp/ \
    -e master_node=False \
    ${image_id}
