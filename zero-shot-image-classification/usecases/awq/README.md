## AutoAWQ

## Envs Setup
### Install AWQ
#### CPU
Install AutoAWQ with CPU backend in your working directory:
```bash
pip install intel_extension_for_pytorch
pip install git+https://github.com/casper-hansen/AutoAWQ.git
```

## use case
Go to tests/workloads directory.
### Inference
The following commands defaultly use CPU, please add flag: `--device xpu --model_dtype float16` if you use XPU.
#### bf16(baseline)
```
./run.sh -t text-generation -m meta-llama/Llama-2-7b-chat-hf --model_dtype bfloat16
```
#### int4
```
./run.sh -t text-generation -m TheBloke/firefly-llama2-7B-chat-AWQ --autoawq int4
```

### Finetune
The following commands defaultly use CPU, please add flag: `--device xpu` if you use XPU.
#### bf16 LoRA(baseline)
```
./run.sh -t fine-tune -m meta-llama/Llama-2-7b-hf
```
#### int4 LoRA
```
./run.sh -t fine-tune -m TheBloke/firefly-llama2-7B-chat-AWQ --autoawq int4
```
