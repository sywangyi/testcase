# TRL on XPU&GPU
This repository includes examples and training recipes to fine-tune large language models using the ORPO and KTO algorithms with TRL.


## ORPO

### Installtion 
To install the necessary dependencies, run the following command:
```bash
pip install -U transformers datasets accelerate peft trl wandb
```

### Usage 
#### 1. Specify the visible devices

For single-card usage:
```bash
# on XPU
export ZE_AFFINITY_MASK=0
# on CUDA
export CUDA_VISIBLE_DEVICES=0
```

For multi-card usage, e.g. 4 cards
```bash
# on XPU
export ZE_AFFINITY_MASK=0,1,2,3
# on CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

#### 2. Run training 
You can run the ORPO training script with custom arguments. Here is an example:
```bash
python run_orpo.py --base_model meta-llama/Meta-Llama-3-8B --model_save_dir OrpoLlama-3-8B --attn_type eager
```
To find out more options, run:
```bash
python run_orpo.py -h
```

## KTO 



