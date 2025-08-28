# Run Guide

## 1. Build container from DockerFile 

There are two dockerfiles for internal and external respectively. 
The external one uses stock PyTorch, IPEX, oneCCL, all w/ version 1.12.

### internal build
```bash
$ docker build -f df_internal --build-arg imageVersion=2022_ww32 -t bert_qa:internal .
```
> notes:
> The imageVersion is the version of base image from internal release of PyTorch, IPEX and oneCCL, it may be changed.

### external build
```bash
$ docker build -f df_external --build-arg http_proxy=http://proxy-chain.intel.com:911 --build-arg https_proxy=http://proxy-chain.intel.com:911 --build-arg no_proxy=*.intel.com -t bert_qa:external .
```

## 2. Pull the pre-built docker image from docker hub

### internal
```bash
$ docker pull appliedmlwf/hf:bert_qa_internal_2022ww32
```
### external
```bash
$ docker pull appliedmlwf/hf:bert_qa_external_2022ww33
```

## 3. Docker container deployment (single node)
You can use below command to get help.
```bash
$ ./host_qa_test.sh -h
```
Run training with below command
```bash
$ ./host_qa_test.sh -c 0 -i {image_id} # 2 DDPs in 1 container
```
note:
1. `{image_id}` could be appliedmlwf/hf:bert_qa_external_2022ww33 or appliedmlwf/hf:bert_qa_internal_2022ww32 based on your needs;
2. the default output path is `/tmp/debug_squad`, you could change it in host_qa_test.sh;

If there is a precedental run, please stop and remove existing containers named master before starting a new round, as below:

```bash
$ docker stop master
$ docker rm master
```

## 4. Docker container deployment (multi-node)
Two scripts are provided. `master_qa_node.sh` should be launched in master node and 
`slave_qa_node.sh` should be launched in slave nodes. ***containers in slave nodes must be created before master***

We take 2 nodes as example.

- **In slave node**
```bash
# for help, use:  ./slave_qa_node.sh -h
$ ./slave_qa_node.sh -i {image_id} -n ov_net1 -s 0 # slave0
```

- **In master node**
```bash
# for help, use: ./master_qa_node.sh -h
$ echo "master" > /tmp/hostfile
$ echo "slave0" >> /tmp/hostfile

$ ./master_qa_node.sh -i {image_id} -n ov_net1 -p 4 -n ov_net1 #total 4 DPP, 2 DDP per container instance
```

where `{image_id}` could be appliedmlwf/hf:bert_qa_external_2022ww33 or appliedmlwf/hf:bert_qa_internal_2022ww32 based on your needs

the output is /tmp/debug_squad, you could change it in master_qa_node.sh

please stop and remove existing containers named master and slave0, before starting the test

```bash
$ docker stop master
$ docker rm master
$ docker stop slave0
$ docker rm slave0
```
multiple dockers in multiple nodes utilize "overlay" network mode here. Following is the step to create it.
for example, we have two nodes(10.165.9.49, 10.165.9.48)

**install consul and create consul service**

in 10.165.9.49
```bash
$ docker pull progrium/consul
$ docker run -d -p 8500:8500 -h consul --name consul progrium/consul -server -bootstrap
```
you could open 10.165.9.49:8500 in browser to check the consul status

**configure the docker listening port and consul addr**

in 10.165.9.49
```bash
vi /etc/docker/daemon.json
{
  "hosts":["tcp://0.0.0.0:2376","unix:///var/run/docker.sock"], #listening in port 2376
  "cluster-store": "consul://10.165.9.49:8500", 
  "cluster-advertise": "10.165.9.49:2376"
}
```
in 10.165.9.48

```bash
vi /etc/docker/daemon.json
{
  "hosts":["tcp://0.0.0.0:2376","unix:///var/run/docker.sock"],
  "cluster-store": "consul://10.165.9.49:8500",
  "cluster-advertise": "10.165.9.48:2376"
}
```
change /usr/lib/systemd/system/docker.service in the two nodes  
from  
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock  
to  
ExecStart=/usr/bin/dockerd --containerd=/run/containerd/containerd.sock

**restart docker service**

```bash
$ systemctl restart docker
```

you could see the nodes(10.165.9.48 and 10.165.9.49) shown in the 10.165.9.49:8500

**create the global overlay network**

```bash
$ docker network create -d overlay ov_net1

$ docker network ls # to check its scope
```

and ov_net1 could be used when creating container in section 4

## 5. Perf knobs
omp_num_threads could be configured in master_qa_node.sh and host_qa_test.sh according to your cpu phy cores per socket.
