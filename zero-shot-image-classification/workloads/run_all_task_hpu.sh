#!/bin/bash

# Default variable values
ipex_optimize=False
jit=False
torch_compile=False
task_name=""
model_dtype="bfloat16"
autocast_dtype="bfloat16"
backend="inductor"
device="hpu"
batch_size=1
num_beams=4
input_tokens=512
output_tokens=32
ipex_optimize_transformers="False"
warm_up_steps=10
run_steps=10
compare_outputs=True
optimum_habana=False  # Add default value for optimum_habana
log_dir_parent=$(date +"%Y%m%d")  # Default value for log_dir_parent
mode="inference"  # Add default mode to control task type

# Function to display script usage
usage() {
 echo "Usage: $0 [OPTIONS]"
 echo "Options:"
 echo " -h, --help               Display this help message"
 echo " -i, --ipex_optimize      Use ipex optimize "
 echo " -j, --jit                Use jit "
 echo " -c, --torch_compile      Use torch compile"
 echo " --model_dtype            Indicate the model dtype[float32, bfloat16, float16]"
 echo " --autocast_dtype         Indicate the compute dtype[float32, bfloat16, float16]"
 echo " --backend                Indicate the torch compile backend[ipex, inductor]"
 echo " --device                 Indicate the computation device[cpu, cuda, xpu]"
 echo " --batch_size             Input batch size for text-generation"
 echo " --num_beams              The num_beams for text-generation"
 echo " --input_tokens           The input token length for text-generation[32, 64, 128, 256, 512, 1024]"
 echo " --output_tokens          The output token length for text-generation"
 echo " --ipex_optimize_transformers              Ipex optimize_transformers for text-generation"
 echo " --warm_up_steps          The benchmark warm up steps for all tasks"
 echo " --run_steps              The benchmark run steps for all tasks"
 echo " --compare_outputs        Enable or disable output comparison [True, False]"
 echo " --optimum_habana         Enable or disable optimum Habana [True, False]"  # Add usage for optimum_habana
 echo " --log_dir_parent         Parent directory for logs (default: current date in YYYYMMDD format)"
 echo " --mode                  Specify task mode: inference, finetune, or both (default: inference)"
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
      --log_dir_parent)
        log_dir_parent=$(extract_argument $@)
        shift
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
      --warm_up_steps)
        warm_up_steps=$(extract_argument $@)
        shift
        ;;
      --run_steps)
        run_steps=$(extract_argument $@)
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
      --mode)
        mode=$(extract_argument $@)
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

# Get the number of available devices
if [ "$HABANA_VISIBLE_DEVICES" == "all" ]; then
  devices=(0 1 2 3 4 5 6 7)
else
  IFS=',' read -r -a devices <<< "$HABANA_VISIBLE_DEVICES"
fi
num_devices=${#devices[@]}

# Initialize arrays to track successful and failed tasks
successful_tasks=()
failed_tasks=()
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create temporary files to track successful and failed tasks
success_file="success_$timestamp.txt"
fail_file="fail_$timestamp.txt"
touch "$success_file" "$fail_file"
# Ensure temporary files are cleaned up on exit
cleanup() {
  rm -f "$success_file" "$fail_file"
}
trap cleanup EXIT

# Function to run tasks in parallel
run_task() {
  local task=$1
  local model=$2
  local device_id=$3
  local model_dtype=$4
  local jit=$5
  local ipex_optimize=$6
  local torch_compile=$7
  local backend=$8
  local device=$9
  local batch_size=${10}
  local num_beams=${11}
  local input_tokens=${12}
  local output_tokens=${13}
  local ipex_optimize_transformers=${14}
  local warm_up_steps=${15}
  local run_steps=${16}
  local log_dir=${17}
  local autocast_dtype=${18}
  local compare_outputs=${19}
  local optimum_habana=${20}  # Add optimum_habana as a parameter
  local success_file=${21}
  local fail_file=${22}

  local log_file="$log_dir/${task}_${model//\//_}_device${device_id}.log"

  echo -e "Running \e[32m$task\e[0m with model \e[34m$model\e[0m on device \e[31m$device_id\e[0m"

  # Export all environment variables to the log file
  echo "Environment Variables:" > "$log_file"
  printenv >> "$log_file"

  # Add Python package versions to the log
  echo -e "\nPython Package Versions:" >> "$log_file"
  pip list | grep -E "torch|huggingface|trans|habana" >> "$log_file"


  # Build the complete command string
  cmd="HABANA_VISIBLE_DEVICES=$device_id ./run.sh \
    --task $task \
    --model_id $model \
    --model_dtype $model_dtype \
    --jit $jit \
    --ipex_optimize $ipex_optimize \
    --torch_compile $torch_compile \
    --backend $backend \
    --device $device \
    --batch_size $batch_size \
    --num_beams $num_beams \
    --input_tokens $input_tokens \
    --output_tokens $output_tokens \
    --ipex_optimize_transformers $ipex_optimize_transformers \
    --warm_up_steps $warm_up_steps \
    --run_steps $run_steps \
    --autocast_dtype $autocast_dtype \
    --compare_outputs $compare_outputs \
    --optimum_habana $optimum_habana"

  # Print the command to the screen
  echo -e "\nExecuting command:\n$cmd\n" >> "$log_file"

  echo -e "\nCommand Output:" >> "$log_file"

  # Execute the command and redirect output to the log file
  eval "$cmd &>> $log_file"

  # Check if the task failed
  if [ $? -ne 0 ]; then
    mv $log_file $log_dir/error_${log_file##*/}
    echo "$task $model" >> "$fail_file"
    echo -e "Task \e[32m$task\e[0m with model \e[34m$model\e[0m on device \e[31m$device_id\e[0m \e[31mfailed\e[0m"
  else
    echo "$task $model" >> "$success_file"
    echo -e "Task \e[32m$task\e[0m with model \e[34m$model\e[0m on device \e[31m$device_id\e[0m \e[32mcompleted successfully\e[0m"
  fi
}

# Create logs directory with timestamp under log_dir_parent
timestamp=$(date +"%Y%m%d_%H%M%S")
base_log_dir="$log_dir_parent/logs_${timestamp}"
mkdir -p "$base_log_dir"

# Create subdirectories for inference and fine-tuning
if [[ "$mode" == "inference" || "$mode" == "both" ]]; then
  lazy_mode=${PT_HPU_LAZY_MODE:-"default"}  # Read PT_HPU_LAZY_MODE, default to "default" if not set
  op_size=${PT_HPU_MAX_COMPOUND_OP_SIZE:-"default"}  # Read PT_HPU_MAX_COMPOUND_OP_SIZE, default to "default" if not set
  inference_log_dir="$base_log_dir/inference_lazy_${lazy_mode}_torch_compile_${torch_compile}_model_dtype_${model_dtype}_autocast_dtype_${autocast_dtype}_op_size_${op_size}"
  mkdir -p "$inference_log_dir"
fi

if [[ "$mode" == "finetune" || "$mode" == "both" ]]; then
  finetune_log_dir="$base_log_dir/finetune"
  mkdir -p "$finetune_log_dir"
fi

# Print the base log directory for reference
echo "Logs will be saved in: $base_log_dir"

# # Update log_dir variable for inference and fine-tuning tasks
# if [[ "$mode" == "inference" || "$mode" == "both" ]]; then
#   log_dir="$inference_log_dir"
# fi

# if [[ "$mode" == "finetune" || "$mode" == "both" ]]; then
#   log_dir="$finetune_log_dir"
# fi

# Define tasks and models
declare -A tasks
tasks["image-to-image"]="stabilityai/stable-diffusion-xl-refiner-1.0 timbrooks/instruct-pix2pix stabilityai/stable-diffusion-2-inpainting"
tasks["image-to-text"]="nlpconnect/vit-gpt2-image-captioning Salesforce/blip-image-captioning-large Salesforce/blip-image-captioning-base"
tasks["image-feature-extraction"]="google/vit-base-patch16-224-in21k facebook/dinov2-base facebook/dinov2-large"
tasks["text-to-image"]="stabilityai/stable-diffusion-xl-base-1.0 stable-diffusion-v1-5/stable-diffusion-v1-5 stable-diffusion-v1-5/stable-diffusion-inpainting"
tasks["text-to-video"]="THUDM/CogVideoX-5b ByteDance/AnimateDiff-Lightning THUDM/CogVideoX-2b"
tasks["zero-shot-image-classification"]="openai/clip-vit-large-patch14 openai/clip-vit-base-patch16 openai/clip-vit-base-patch32"
tasks["sentence-similarity"]="sentence-transformers/all-mpnet-base-v2 sentence-transformers/all-MiniLM-L6-v2 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tasks["question-answering"]="deepset/roberta-base-squad2 distilbert/distilbert-base-cased-distilled-squad deepset/bert-large-uncased-whole-word-masking-squad2"
tasks["text-generation"]="openai-community/gpt2 meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-14B-Instruct openai-community/gpt2-xl"
tasks["summarization"]="facebook/bart-large-cnn jordiclive/flan-t5-3b-summarizer google/pegasus-xsum"
tasks["translation"]="google-t5/t5-large google-t5/t5-3b facebook/nllb-200-distilled-600M"
tasks["automatic-speech-recognition"]="jonatasgrosman/wav2vec2-large-xlsr-53-english openai/whisper-large-v2 facebook/hubert-large-ls960-ft"
tasks["text-to-speech"]="microsoft/speecht5_tts facebook/hf-seamless-m4t-large suno/bark"
tasks["audio-classification"]="facebook/mms-lid-256 MIT/ast-finetuned-audioset-10-10-0.4593 alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
tasks["visual-question-answering"]="Salesforce/blip-vqa-base dandelin/vilt-b32-finetuned-vqa Salesforce/blip-vqa-capfilt-large"
tasks["document-question-answering"]="impira/layoutlm-document-qa naver-clova-ix/donut-base-finetuned-docvqa impira/layoutlm-invoices"
tasks["image-text-to-text"]="meta-llama/Llama-3.2-11B-Vision-Instruct llava-hf/llava-v1.6-mistral-7b-hf Qwen/Qwen2-VL-7B-Instruct"
tasks["video-text-to-text"]="llava-hf/LLaVA-NeXT-Video-7B-hf THUDM/cogvlm2-llama3-caption"

# Define fine-tuning tasks and models with model_type and parallel_type
declare -A finetune_tasks
#finetune_tasks["llm-lora"]="meta-llama/Llama-3.1-8B-Instruct:bfloat16:ddp meta-llama/Llama-2-70b-hf:bfloat16:fsdp"
finetune_tasks["sd-dreambooth-lora"]="stable-diffusion-v1-5/stable-diffusion-v1-5:bfloat16:ddp"

# Create a list of tasks to run
task_list=()
for task in "${!tasks[@]}"; do
  models=(${tasks[$task]})
  for model in "${models[@]}"; do
    task_list+=("$task $model")
  done
done

# Export the run_task function and necessary variables for xargs
export -f run_task
export HABANA_VISIBLE_DEVICES
#export model_dtype jit ipex_optimize torch_compile backend device batch_size num_beams input_tokens output_tokens ipex_optimize_transformers warm_up_steps run_steps log_dir autocast_dtype

# Create a list of tasks with device IDs and parameters
task_with_device_list=()
for i in "${!task_list[@]}"; do
  task_with_device_list+=("${task_list[$i]} ${devices[$((i % num_devices))]} $model_dtype $jit $ipex_optimize $torch_compile $backend $device $batch_size $num_beams $input_tokens $output_tokens $ipex_optimize_transformers $warm_up_steps $run_steps $inference_log_dir $autocast_dtype $compare_outputs $optimum_habana $success_file $fail_file")
done

# Function to run fine-tuning tasks
run_finetune_task() {
  local task=$1
  local model=$2
  local model_type=$3
  local parallel_type=$4
  local device=$5
  local log_dir=$6

  # Generate a unique output directory for each model
  local output_dir="$log_dir/${task}_${model//\//_}_output"

  local log_file="$log_dir/${task}_${model//\//_}_finetune.log"

  echo -e "Running fine-tuning \e[32m$task\e[0m with model \e[34m$model\e[0m (model_type: \e[33m$model_type\e[0m, parallel_type: \e[33m$parallel_type\e[0m) on device \e[31m$device\e[0m"
  # Export all environment variables to the log file
  echo "Environment Variables:" > "$log_file"
  printenv >> "$log_file"

  # Add Python package versions to the log
  echo -e "\nPython Package Versions:" >> "$log_file"
  pip list | grep -E "torch|huggingface|trans|habana" >> "$log_file"

  # Build the fine-tuning command
  cmd="bash ./run.sh \
    --task $task \
    --model_id $model \
    --device $device \
    --model_dtype $model_type \
    --parallel_type $parallel_type \
    --output_dir $output_dir"  # Pass output_dir

  # Print the command to the screen
  echo -e "\nExecuting command:\n$cmd\n"  >> "$log_file"
  echo -e "\nCommand Output:" >> "$log_file"

  # Execute the command and redirect output to the log file
  eval "$cmd &>> $log_file"

  # Check if the task failed
  if [ $? -ne 0 ]; then
    mv $log_file $log_dir/error_${log_file##*/}
    echo "$task $model" >> "$fail_file"
    echo -e "Fine-tuning task \e[32m$task\e[0m with model \e[34m$model\e[0m on device \e[31m$device\e[0m \e[31mfailed\e[0m"
  else
    echo "$task $model" >> "$success_file"
    echo -e "Fine-tuning task \e[32m$task\e[0m with model \e[34m$model\e[0m on device \e[31m$device\e[0m \e[32mcompleted successfully\e[0m"
  fi
}

# Run tasks based on mode
if [[ "$mode" == "inference" || "$mode" == "both" ]]; then
  # Run tasks in parallel using xargs
  printf "%s\n" "${task_with_device_list[@]}" | xargs -n 22 -P $num_devices bash -c 'run_task "$@"' _
fi

if [[ "$mode" == "finetune" || "$mode" == "both" ]]; then
  for task in "${!finetune_tasks[@]}"; do
    models=(${finetune_tasks[$task]})
    for model_entry in "${models[@]}"; do
      IFS=':' read -r model model_type parallel_type <<< "$model_entry"
      run_finetune_task "$task" "$model" "$model_type" "$parallel_type" "$device" "$finetune_log_dir"
    done
  done
fi

# Print summary of tasks
echo -e "\nSummary of tasks:"
echo -e "\e[32mSuccessful tasks:\e[0m"
if [ -s "$success_file" ]; then
  cat "$success_file"
else
  echo "  None"
fi

echo -e "\n\e[31mFailed tasks:\e[0m"
if [ -s "$fail_file" ]; then
  cat "$fail_file"
else
  echo "  None"
fi

# Count successful and failed tasks
successful_count=$(wc -l < "$success_file" | xargs)
failed_count=$(wc -l < "$fail_file" | xargs)

# Print task counts
echo -e "\n\e[32mTotal successful tasks: $successful_count\e[0m"
echo -e "\e[31mTotal failed tasks: $failed_count\e[0m"

# Generate HTML report after all tasks are completed
python3 logs_parser.py "$base_log_dir"

echo -e "\nPerformance report generated"

echo "All tasks completed."
