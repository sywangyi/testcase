## Bitsandbytes

## Envs Setup
### Install BNB
#### CPU
Install bitsandbytes with CPU backend in your working directory:
```bash
git clone --branch multi-backend-refactor https://github.com/TimDettmers/bitsandbytes.git && cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=cpu -S .
make
pip install .
```

#### XPU
<TBF>

### Install HF Transformers
Please make sure `transformers >= 4.45.0`

## use case
Go to tests/workloads directory.
### Inference
The following commands defaultly use CPU, please add flag: `--device xpu --model_dtype float16` if you use XPU.
#### bf16(baseline)
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16
```
#### int8
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --bitsandbytes int8
```
#### nf4
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --bitsandbytes nf4
```
#### fp4
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16 --bitsandbytes fp4
```

### Finetune
The following commands defaultly use CPU, please add flag: `--device xpu` if you use XPU.
#### bf16 LoRA(baseline)
```
./run.sh -t fine-tune
```
#### int8 LoRA
```
./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --bitsandbytes int8
```
#### nf4 QLoRA
```
./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --bitsandbytes nf4
```
#### fp4 QLoRA
```
./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf --bitsandbytes fp4
```
