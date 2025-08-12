# HF Workload Test Guide

**All tested are and should be validated in docker container**

Assume you are running command in the directory of where this README is.

## 1. build docker image

```bash
$ cd ../../HuggingFace/docker
$ bash ./build_image.sh -d <device>
```

`-d` options: "cpu", "xpu" and "cuda"

## 2. launch docker container

```bash
$ bash ./run_docker.sh -d <device>
```
You can run `./run_docker.sh -h` for more options. By default, current directory will be mounted to `/mnt` directory of the container. You can specify your own mount directory.

## 3. run test in container

### 3.1 prepare env

```bash
$ pip install -r requirements.txt
```
**___Note: if the installation of transformers and accelerate fail, you will have to install them from source.___**

Make sure you copied or mounted this repository into container.

### 3.2 inference tests

#### CPU
##### Native DX
By default, We use `BF16` and `BF16 + torch.compile` on CPU for all inference tasks, run the following command:

```bash
bash ./run_cpu.sh
```

After running this command, you can find the data in the `cpu_benmark.log`. Make sure you read the instruction at the beginning of the log.

##### Advanced DX
For advanced DX, `optimum-intel` is required and can be installed with the following commands:

```bash
$ git clone https://github.com/huggingface/optimum-intel.git && cd optimum-intel
$ pip install .
```

Use `--optimum_intel` in `image-classification`, `question-answering`, and `text-generation` can enable `optimum-intel` optimization, for example:

```bash
$ bash ./run.sh -t image-classification -m google/vit-base-patch16-224 --model_dtype bfloat16 --optimum_intel True
```

#### XPU
Before running the test cases, you need to follow [the IPEX official documentation](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=gpu&version=v2.1.10%2Bxpu) to set up the correct environment.

##### Native DX
Run below command:

```bash
$ bash ./run_all_task_xpu.sh --model_dtype float16 --warm_up_steps 10 --run_steps 10 2>&1 | tee xpu_benchmark_raw.log
```

If you want to compare the performance with NV GPU, just add the flag `--device cuda` to the command above.

When the test finishes, you can use the following command to extract the performance data from the log:

```bash
$ python analyse_logs.py --file_names xpu_benchmark_raw.log --out_name xpu_benchmark.log
```
##### Advanced DX
<to be filled>

### Finetune

Please notice that we run the official finetune script on peft and diffusers which are the submodules, please get the latest update by:
```bash
git submodule sync && git submodule update --init --recursive
```
Install from source in the container:
```bash
cd third_party/peft/ && pip install . && cd ../diffusers/ && pip install .
```

We defaultly use bf16 to train [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) in [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) dataset with 4 DDP across 4 instances. Please change the [fine-tune/hostfile](https://github.com/intel-sandbox/HuggingFace/blob/main/tests/workloads/fine-tune/hostfile) to your instances ip and run the following command:

#### CPU

```bash
$ bash ./run.sh -t llm-lora -m meta-llama/Llama-3.1-8B-Instruct --model_dtype bfloat16
```

To train a stable diffusion dreambooth lora finetune, please run:
```bash
bash ./run.sh -t sd-dreambooth-lora -m stable-diffusion-v1-5/stable-diffusion-v1-5
```

#### XPU
source oneAPI first:
```bash
$ source /opt/intel/oneapi/setvars.sh --force
```
need upgrade `transformers>=4.48.0`  
Lora LLM:
```bash
bash ./run.sh -t llm-lora -m meta-llama/Llama-3.1-8B-Instruct --model_dtype bfloat16 --device xpu --parallel_type fsdp
bash ./run.sh -t llm-lora -m hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --model_dtype bfloat16 --device xpu
bash ./run.sh -t llm-lora -m hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4 --model_dtype bfloat16 --device xpu
bash ./run.sh -t llm-lora -m hugging-quants/Meta-Llama-3.1-8B-Instruct-BNB-NF4 --model_dtype bfloat16 --device xpu
```
Stable diffusion dreambooth lora:
```bash
bash ./run.sh -t sd-dreambooth-lora -m stable-diffusion-v1-5/stable-diffusion-v1-5 --device xpu
```

#### CUDA

```bash
$ bash ./run.sh -t llm-lora -m meta-llama/Llama-3.1-8B-Instruct --model_dtype bfloat16 --device cuda
```

```bash
bash ./run.sh -t sd-dreambooth-lora -m stable-diffusion-v1-5/stable-diffusion-v1-5 --device cuda
```

## Notes

### Connection Error
If you cannot connect to huggingface model hub, please try `export HF_ENDPOINT=https://hf-mirror.com`

### Batch Size
The actual batch size is calculated as:
```math
micro\_batch\_size \times gradient\_accumulation\_steps \times world\_size
```

where the `gradient_accumulation_steps` is calculated using the following formula:
```math
gradient\_accumulation\_steps = batch\_size // micro\_batch\_size // world\_size
```
Please make sure that the input batch size is divisible by `micro_batch_size` and `world_size`, otherwise the actual batch size will differ from the the given batch size, e.g. with `batch_size=128`, `micro_batch_size=4` and `world_size=6`, the actual batch size is equal to 120, rather than 128.
