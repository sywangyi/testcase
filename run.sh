#!/bin/bash

# Default variable values
ipex_optimize=False
jit=False
torch_compile=False
task_name=""
model_id=""
model_dtype="float32"
autocast_dtype="float32"
backend="inductor"
device="cpu"
batch_size=1
num_beams=4
input_tokens=32
output_tokens=32
ipex_optimize_transformers="False"
gradient_checkpointing="False"
num_processes=6
warm_up_steps=10
run_steps=10

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help            Display this help message"
 echo " -t, --task            Specify task name"
 echo " -m, --model_id        Specify model ID "
 echo " -i, --ipex            Use ipex optimize "
 echo " -j, --jit             Use jit "
 echo " -c, --torch_compile   Use torch compile"
 echo " --model_dtype         Indicate the model dtype[float32, bfloat16, float16]"
 echo " --autocast_dtype       Indicate the compute dtype[float32, bfloat16, float16]"
 echo " --backend             Indicate the torch compile backend[ipex, inductor]"
 echo " --device              Indicate the computation device[cpu, cuda, xpu]"
 echo " --batch_size          Input batch size for text-generation"
 echo " --num_beams           The num_beams for text-generation"
 echo " --input_tokens        The input token length for text-generation[32, 64, 128, 256, 512, 1024]"
 echo " --output_tokens       The output token length for text-generation"
 echo " --ipex_optimize_transformers              Ipex optimize_transformers for text-generation"
 echo " --gradient_checkpointing         Whether to run fine-tuning with gradient checkpoint to save memory, only used for fine-tune task"
 echo " --num_processes       The number of data parallelism, only used for CPU fine-tune task"
 echo " --warm_up_steps      The benchmark warm up steps for all tasks"
 echo " --run_steps          The benchmark run steps for all tasks"
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


if [[ "$device" = "cpu" ]]; then
  # Setup environment variables for performance on Xeon
  export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
  export KMP_BLOCKTIME=INF
  export KMP_TPAUSE=0
  export KMP_SETTINGS=1
  export KMP_AFFINITY=granularity=fine,compact,1,0
  export KMP_FORJOIN_BARRIER_PATTERN=dist,dist
  export KMP_PLAIN_BARRIER_PATTERN=dist,dist
  export KMP_REDUCTION_BARRIER_PATTERN=dist,dist
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
  # Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
  export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
fi
export TORCHINDUCTOR_FREEZING=1
export TRITON_CODEGEN_INTEL_XPU_BACKEND=1
export OMP_NUM_THREADS=56

# Perform the desired actions based on the provided flags and arguments
if [[ "$task_name" == "fine-tune" ]]; then
  if [[ "$device" == "cpu" ]]; then
    export CCL_WORKER_COUNT=1
    oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
    source $oneccl_bindings_for_pytorch_path/env/setvars.sh
    mpirun -n $num_processes -ppn 1 -genv OMP_NUM_THREADS=$(($OMP_NUM_THREADS*2/$num_processes)) -genv MASTER_ADDR=127.0.0.1 -genv MASTER_PORT=29500 python $task_name/run_$task_name.py --bf16 True --use_ipex $ipex_optimize
  else
    accelerate launch --config_file $task_name/"$device"_config.yaml $task_name/run_$task_name.py
  fi
else
  numactl -C 0-55 --membind 0 python $task_name/run_$task_name.py --model_id $model_id --model_dtype $model_dtype --jit $jit --ipex_optimize $ipex_optimize --autocast_dtype $autocast_dtype --torch_compile $torch_compile --backend $backend --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --ipex_optimize_transformers $ipex_optimize_transformers --warm_up_steps $warm_up_steps --run_steps $run_steps
fi
