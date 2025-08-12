#!/bin/bash

# Default variable values
ipex_optimize=False
jit=False
torch_compile=False
task_name=""
autocast_dtype="float32"
backend="inductor"
device="cpu"
batch_size=1
num_beams=1
input_tokens=512
output_tokens=128
ipex_optimize_transformers="False"
warm_up_steps=10
run_steps=10
# Compare outputs
compare_outputs=False

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
 echo " --compare_outputs     Compare outputs of baseline model and optimized model"
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

echo "test image to image"
declare -a model_list
model_list=("stabilityai/stable-diffusion-2-inpainting" "stabilityai/stable-diffusion-xl-refiner-1.0" "lllyasviel/sd-controlnet-canny" "timbrooks/instruct-pix2pix")

for model in "${model_list[@]}"
do
    bash ./run.sh --task image-to-image --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test image-to-text"
model_list=("nlpconnect/vit-gpt2-image-captioning" "Salesforce/blip-image-captioning-large" "Salesforce/blip-image-captioning-base")

for model in "${model_list[@]}"
do
    bash ./run.sh --task image-to-text --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test image-feature-extraction"
model_list=("google/vit-base-patch16-224-in21k" "facebook/dinov2-base" "facebook/dinov2-small")

for model in "${model_list[@]}"
do
    bash ./run.sh --task image-feature-extraction --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test text-to-image"
model_list=("stable-diffusion-v1-5/stable-diffusion-v1-5" "stable-diffusion-v1-5/stable-diffusion-inpainting" "stabilityai/stable-diffusion-xl-base-1.0")

for model in "${model_list[@]}"
do
    bash ./run.sh --task text-to-image --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test text-to-video"
model_list=("THUDM/CogVideoX-5b" "ByteDance/AnimateDiff-Lightning" "THUDM/CogVideoX-2b")

for model in "${model_list[@]}"
do
    bash ./run.sh --task text-to-video --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test zero-shot-image-classification"
model_list=("openai/clip-vit-large-patch14" "openai/clip-vit-base-patch16" "openai/clip-vit-base-patch32")

for model in "${model_list[@]}"
do
    bash ./run.sh --task zero-shot-image-classification --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test sentence similarity"
model_list=("sentence-transformers/all-mpnet-base-v2" "sentence-transformers/all-MiniLM-L6-v2" "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

for model in "${model_list[@]}"
do
    bash ./run.sh --task sentence-similarity --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test question-answering"
model_list=("deepset/roberta-base-squad2" "distilbert/distilbert-base-cased-distilled-squad" "deepset/bert-large-uncased-whole-word-masking-squad2")

for model in "${model_list[@]}"
do
    bash ./run.sh --task question-answering --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test text-generation"
model_list=("facebook/opt-1.3b" "gpt2")

for model in "${model_list[@]}"
do
    bash ./run.sh --task text-generation --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --ipex_optimize_transformers $ipex_optimize_transformers --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done

model_list=("Qwen/Qwen2.5-1.5B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")

for model in "${model_list[@]}"
do
    bash ./run.sh --task text-generation --model_id $model --model_dtype bfloat16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens --ipex_optimize_transformers $ipex_optimize_transformers --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done

echo "test summarization"
model_list=("facebook/bart-large-cnn" "sshleifer/distilbart-cnn-12-6" "google/pegasus-xsum")

for model in "${model_list[@]}"
do
    bash ./run.sh --task summarization --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs --batch_size $batch_size --num_beams $num_beams --input_tokens $input_tokens --output_tokens $output_tokens
    echo "----------------------------"
done


echo "test translation"
model_list=("google-t5/t5-small" "google-t5/t5-base" "Helsinki-NLP/opus-mt-mul-en")

for model in "${model_list[@]}"
do
    bash ./run.sh --task translation --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs --batch_size $batch_size --num_beams $num_beams --input_tokens 128 --output_tokens 128
    echo "----------------------------"
done


echo "test automatic-speech-recognition"
model_list=("openai/whisper-large-v2" "jonatasgrosman/wav2vec2-large-xlsr-53-english" "openai/whisper-small")

for model in "${model_list[@]}"
do
    bash ./run.sh --task automatic-speech-recognition --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize  --model_dtype $model_dtype --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test text-to-speech"
model_list=("microsoft/speecht5_tts" "suno/bark" "facebook/mms-tts-eng")

for model in "${model_list[@]}"
do
    bash ./run.sh --task text-to-speech --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test audio-classification"
model_list=("facebook/mms-lid-256" "MIT/ast-finetuned-audioset-10-10-0.4593" "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech")

for model in "${model_list[@]}"
do
    bash ./run.sh --task audio-classification --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test visual-question-answering"
model_list=("Salesforce/blip-vqa-capfilt-large" "Salesforce/blip-vqa-base" "dandelin/vilt-b32-finetuned-vqa")

for model in "${model_list[@]}"
do
    bash ./run.sh --task visual-question-answering --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test document-question-answering"
model_list=("impira/layoutlm-document-qa" "naver-clova-ix/donut-base-finetuned-docvqa" "impira/layoutlm-invoices")

for model in "${model_list[@]}"
do
    bash ./run.sh --task document-question-answering --model_id $model --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done


echo "test image-text-to-text"
model_list=("meta-llama/Llama-3.2-11B-Vision-Instruct" "Qwen/Qwen2-VL-7B-Instruct")

for model in "${model_list[@]}"
do
    bash ./run.sh --task image-text-to-text --model_id $model --model_dtype bfloat16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done

bash ./run.sh --task image-text-to-text --model_id llava-hf/llava-v1.6-mistral-7b-hf --model_dtype float16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
echo "----------------------------"

echo "test video-text-to-text"
model_list=("llava-hf/LLaVA-NeXT-Video-7B-hf" "llava-hf/llava-interleave-qwen-7b-hf")

for model in "${model_list[@]}"
do
    bash ./run.sh --task video-text-to-text --model_id $model --model_dtype bfloat16 --jit $jit --ipex_optimize $ipex_optimize --torch_compile $torch_compile --backend $backend --device $device --warm_up_steps $warm_up_steps --run_steps $run_steps --compare_outputs $compare_outputs
    echo "----------------------------"
done
