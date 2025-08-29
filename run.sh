#!/bin/bash

# Default variable values

## test case configurations
task_name=""
model_id=""
model_dtype="float32"
autocast_dtype="float32"
batch_size=1
num_beams=1
input_tokens=32
output_tokens=32
do_sample=False
gradient_checkpointing=False
quant_algo="None"
quant_dtype="None"
parallel_type="ddp"

## runtime configurations
backend="inductor"
device="cpu"
num_processes=4
warm_up_steps=10
run_steps=10
output_dir="./outputs"  # Add default value for output_dir

## optimization options
ipex_optimize=False
ipex_optimize_transformers=False
jit=False
torch_compile=False
optimum_intel=False
optimum_habana=False  # Add default value for optimum_habana

# TP args
tp_plan="auto"
tp_size=1
enable_ep=False

# Compare outputs
compare_outputs=False

# arg parser
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -t, --task            Specify task name"
 echo " -m, --model_id        Specify model id"
 echo " --model_dtype         Indicate the model dtype[float32, bfloat16, float16]"
 echo " --autocast_dtype      Indicate the compute dtype[float32, bfloat16, float16]"
 echo " --batch_size          Input batch size for text-generation"
 echo " --num_beams           The num_beams for text-generation"
 echo " --input_tokens        The input token length for text-generation[32, 64, 128, 256, 512, 1024]"
 echo " --output_tokens       The output token length for text-generation"
 echo " --do_sample           Whether to use sample in text-generation"
 echo " --gradient_checkpointing  Whether to run fine-tuning with gradient checkpoint to save memory, only used for fine-tune task"
 echo " --quant_algo          Use quant_algo to decide quantization method, options are ["bitsandbytes", "autoawq", "gptqmodel"]"
 echo " --quant_dtype         Use quant_dtype to decide quantization data type, like ["int8", "nf4", "fp4"] in bitsandbytes, ["int4"] in autoawq, ["int4"] in gptqmodel"
 echo " --backend             Indicate the torch compile backend[ipex, inductor]"
 echo " --device              Indicate the computation device[cpu, cuda, xpu]"
 echo " --num_processes       The number of data parallelism, only used for CPU fine-tune task"
 echo " --warm_up_steps       The benchmark warm up steps for all tasks"
 echo " --run_steps           The benchmark run steps for all tasks"
 echo " -i, --ipex            Use ipex optimize"
 echo " -j, --jit             Use jit"
 echo " -c, --torch_compile   Use torch compile"
 echo " --ipex_optimize_transformers              Ipex optimize_transformers for text-generation"
 echo " --optimum_intel       Use optimum-intel optimization"
 echo " --parallel_type       Choose parallel type for accelerate[ddp, fsdp]"
 echo " --tp_plan             Run model with tensor parallelism"
 echo " --tp_size             The tensor parallelism size"
 echo " --enable_ep           If enable expert parallelism"
 echo " --compare_outputs     Compare outputs of baseline model and optimized model"
 echo " --optimum_habana      Enable or disable optimum Habana [True, False]"  # Add usage for optimum_habana
 echo " --output_dir          Specify the output directory for results"
}

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
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
      -t | --task*)
        if ! has_argument $@; then
          echo "Script name not specified." >&2
          usage
          exit 1
        fi

        task_name=$(extract_argument $@)

        shift
        ;;
      -m | --model_id*)
        if ! has_argument $@; then
          echo "Model ID not specified." >&2
          usage
          exit 1
        fi

        model_id=$(extract_argument $@)

        shift
        ;;
      -i | --ipex_optimize)
        ipex_optimize=$(extract_argument $@)
        shift
        ;;
      -j | --jit)
        jit=$(extract_argument $@)
        shift
        ;;
      -c | --torch_compile)
        torch_compile=$(extract_argument $@)
        shift
        ;;
      --model_dtype)
        model_dtype=$(extract_argument $@)
        shift
        ;;
      --autocast_dtype)
        autocast_dtype=$(extract_argument $@)
        shift
        ;;
      --backend)
        backend=$(extract_argument $@)
        shift
        ;;
      --parallel_type)
        parallel_type=$(extract_argument $@)
        shift
        ;;
      --device)
        device=$(extract_argument $@)
        shift
        ;;
      --batch_size)
        batch_size=$(extract_argument $@)
        shift
        ;;
      --num_beams)
        num_beams=$(extract_argument $@)
        shift
        ;;
      --input_tokens)
        input_tokens=$(extract_argument $@)
        shift
        ;;
      --output_tokens)
        output_tokens=$(extract_argument $@)
        shift
        ;;
      --do_sample)
        do_sample=$(extract_argument $@)
        shift
        ;;
      --ipex_optimize_transformers)
        ipex_optimize_transformers=$(extract_argument $@)
        shift
        ;;
      --gradient_checkpointing)
        gradient_checkpointing=$(extract_argument $@)
        shift
        ;;
      --num_processes)
        num_processes=$(extract_argument $@)
        shift
        ;;
      --warm_up_steps)
        warm_up_steps=$(extract_argument $@)
        shift
        ;;
      --run_steps)
        run_steps=$(extract_argument $@)
        shift
        ;;
      --optimum_intel)
        optimum_intel=$(extract_argument $@)
        shift
        ;;
      --quant_algo)
        quant_algo=$(extract_argument $@)
        shift
        ;;
      --quant_dtype)
        quant_dtype=$(extract_argument $@)
        shift
        ;;
      --tp_plan)
        tp_plan=$(extract_argument $@)
        shift
        ;;
      --tp_size)
        tp_size=$(extract_argument $@)
        shift
        ;;
      --enable_ep)
        enable_ep=$(extract_argument $@)
        shift
        ;;
      --compare_outputs)
        compare_outputs=$(extract_argument $@)
        shift
        ;;
      --optimum_habana)  # Add parsing for optimum_habana
        optimum_habana=$(extract_argument $@)
        shift
        ;;
      --output_dir)
        output_dir=$(extract_argument $@)
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

CORES=`lscpu | grep 'Core(s) per socket' | awk '{print $4}'`
NUMAS=`lscpu | grep 'NUMA node(s):' | awk '{print $3}'`
SOCKETS=`lscpu | grep 'Socket(s):' | awk '{print $2}'`
CORES=$[CORES*SOCKETS/NUMAS]
export KMP_AFFINITY=granularity=fine,compact,1,0
export TORCHINDUCTOR_FREEZING=1
export TORCHINDUCTOR_CPP_WRAPPER=1
export TRITON_CODEGEN_INTEL_XPU_BACKEND=1

small_model_list_4_cores=("Helsinki-NLP/opus-mt-mul-en" "google-t5/t5-small" "facebook/dinov2-small" "sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L6-v2" "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
for item in "${small_model_list_4_cores[@]}"; do
  if [[ "$item" == "$model_id" ]]; then
    CORES=4
    break
  fi
done

small_model_list_8_cores=("google/vit-base-patch16-224-in21k")
for item in "${small_model_list_8_cores[@]}"; do
  if [[ "$item" == "$model_id" ]]; then
    CORES=8
    break
  fi
done

# These models can get better and stable performance with jemalloc compared to tcmalloc(default malloc)
jemalloc_model_list=("jonatasgrosman/wav2vec2-large-xlsr-53-english" "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech" "openai/clip-vit-large-patch14" "facebook/opt-1.3b" "facebook/dinov2-small")
for item in "${jemalloc_model_list[@]}"; do
  if [[ "$item" == "$model_id" ]]; then
    export LD_PRELOAD=$(echo $LD_PRELOAD | sed "s|/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4|/usr/lib/x86_64-linux-gnu/libjemalloc.so|g")
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    break
  fi
done

echo $LD_PRELOAD
export OMP_NUM_THREADS=${CORES}
export TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE=${CORES}

if [[ "$device" == "cpu" ]]; then
  export CCL_WORKER_COUNT=1
  accelerate_config="fine-tune/cpu_config.yaml"
elif [[ "$device" == "xpu" ]]; then
  if [[ "$parallel_type" == "ddp" ]]; then
    accelerate_config="fine-tune/xpu_config_ddp.yaml"
  elif [[ "$parallel_type" == "fsdp" ]]; then
    accelerate_config="fine-tune/xpu_config_fsdp.yaml"
  fi
elif [[ "$device" == "hpu" ]]; then
  if [[ "$parallel_type" == "ddp" ]]; then
    accelerate_config="fine-tune/hpu_config_ddp.yaml"
  elif [[ "$parallel_type" == "fsdp" ]]; then
    accelerate_config="fine-tune/hpu_config_fsdp.yaml"
  fi
elif [[ "$device" == "cuda" ]]; then
  accelerate_config="fine-tune/cuda_config_ddp.yaml"
fi

# Perform the desired actions based on the provided flags and arguments
if [[ "$task_name" == "llm-lora" ]]; then
  if [[ "$device" == "xpu" || ("$device" == "hpu" && "$parallel_type" == "fsdp") ]]; then
    file="../../third_party/peft/examples/sft/train.py"
  else
    file="../../third_party/peft/examples/olora_finetuning/olora_finetuning.py"
  fi
  if [[ "$parallel_type" == "fsdp" ]]; then
    if [[ "$device" == "xpu" || "$device" == "hpu" ]]; then
      accelerate launch --config_file $accelerate_config $file \
        --model_name_or_path $model_id \
        --seed 100 \
        --dataset_name "smangrul/ultrachat-10k-chatml" \
        --chat_template_format "chatml" \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "train,test" \
        --max_seq_len 2048 \
        --num_train_epochs 1 \
        --logging_steps 5 \
        --log_level "info" \
        --logging_strategy "steps" \
        --eval_strategy "epoch" \
        --save_strategy "epoch" \
        --bf16 True \
        --packing True \
        --learning_rate 1e-4 \
        --lr_scheduler_type "cosine" \
        --weight_decay 1e-4 \
        --warmup_ratio 0.0 \
        --max_grad_norm 1.0 \
        --output_dir "tmp-lora-sft-fsdp" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --gradient_checkpointing True \
        --use_reentrant False \
        --dataset_text_field "content" \
        --use_flash_attn False \
        --use_peft_lora True \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj" \
        --use_4bit_quantization False
    else
      accelerate launch --config_file $accelerate_config $file --base_model $model_id --init_lora_weights gaussian --seed 42 --torch_dtype $model_dtype --device_map $device 
    fi
  else
    if [[ "$device" == "xpu" ]]; then
      accelerate launch --config_file $accelerate_config $file \
        --model_name_or_path $model_id \
        --seed 100 \
        --dataset_name "smangrul/ultrachat-10k-chatml" \
        --chat_template_format "chatml" \
        --add_special_tokens False \
        --append_concat_token False \
        --splits "train,test" \
        --max_seq_len 2048 \
        --num_train_epochs 1 \
        --logging_steps 5 \
        --log_level "info" \
        --logging_strategy "steps" \
        --eval_strategy "epoch" \
        --save_strategy "epoch" \
        --bf16 True \
        --packing True \
        --learning_rate 1e-4 \
        --lr_scheduler_type "cosine" \
        --weight_decay 1e-4 \
        --warmup_ratio 0.0 \
        --max_grad_norm 1.0 \
        --output_dir "tmp-lora-sft-ddp" \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --gradient_checkpointing True \
        --use_reentrant False \
        --dataset_text_field "content" \
        --use_flash_attn False \
        --use_peft_lora True \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.1 \
        --lora_target_modules "q_proj,k_proj,v_proj,o_proj,up_proj,gate_proj" \
        --use_4bit_quantization False
    else
      accelerate launch --config_file $accelerate_config $file \
        --base_model $model_id \
        --init_lora_weights gaussian \
        --seed 42 \
        --torch_dtype $model_dtype \
        --device_map $device \
        --output_dir $output_dir  # Pass output_dir
    fi
  fi
elif [[ "$task_name" == "sd-dreambooth-lora" ]]; then
  file="../../third_party/diffusers/examples/dreambooth/train_dreambooth_lora.py"
  export INSTANCE_DIR="./datasets/dog"
  accelerate launch --config_file $accelerate_config $file \
    --pretrained_model_name_or_path=$model_id  \
    --instance_data_dir=$INSTANCE_DIR \
    --output_dir=$output_dir \
    --instance_prompt="a photo of sks dog" \
    --resolution=512 \
    --prior_generation_precision bf16 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --checkpointing_steps=100 \
    --learning_rate=3e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=200 \
    --validation_prompt="A photo of sks dog in a bucket" \
    --validation_epochs=20 \
    --seed="0"
elif [[ "$task_name" == "tp" ]]; then
  if [[ "$device" == "hpu" ]]; then
    deepspeed --num_gpus $tp_size $task_name/run_"$task_name"_deepspeed.py --model_id $model_id --model_dtype $model_dtype --ipex_optimize $ipex_optimize --autocast_dtype $autocast_dtype --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --do_sample $do_sample --ipex_optimize_transformers $ipex_optimize_transformers --warm_up_steps $warm_up_steps --run_steps $run_steps --optimum_intel $optimum_intel --optimum_habana $optimum_habana
  else
    accelerate launch --config_file "fine-tune/"$device"_tp_config.yaml" $task_name/run_$task_name.py --model_id $model_id --model_dtype $model_dtype --quant_algo $quant_algo --quant_dtype $quant_dtype --jit $jit --ipex_optimize $ipex_optimize --autocast_dtype $autocast_dtype --torch_compile $torch_compile --backend $backend --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --do_sample $do_sample --ipex_optimize_transformers $ipex_optimize_transformers --warm_up_steps $warm_up_steps --run_steps $run_steps --optimum_intel $optimum_intel --tp_plan $tp_plan --enable_ep $enable_ep --optimum_habana $optimum_habana
  fi
else
  numactl -C '0-'$[CORES-1] --membind 0 python $task_name/run_$task_name.py --model_id $model_id --model_dtype $model_dtype --quant_algo $quant_algo --quant_dtype $quant_dtype --jit $jit --ipex_optimize $ipex_optimize --autocast_dtype $autocast_dtype --torch_compile $torch_compile --backend $backend --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --do_sample $do_sample --ipex_optimize_transformers $ipex_optimize_transformers --warm_up_steps $warm_up_steps --run_steps $run_steps --optimum_intel $optimum_intel --compare_outputs $compare_outputs --optimum_habana $optimum_habana
fi
