import os
import io
import sys
import av
import torch
import time
import numpy as np
import logging
import cv2
from PIL import Image
from decord import cpu, VideoReader, bridge

logging.basicConfig(level=logging.INFO)

from transformers.utils import ContextManagers
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlavaNextVideoProcessor,
    LlavaNextVideoForConditionalGeneration,
    LlavaProcessor,
    LlavaForConditionalGeneration,
)

sys.path.append(os.path.dirname(__file__) + "/..")
from common import (
    SEED,
    get_args,
    get_torch_dtype,
    wrap_forward_for_benchmark,
    synchronize_device,
    compute_sentence_similarity,
    log_latency,
)

inference_context = [torch.inference_mode()]


def generate(model, inputs, warm_up_steps, run_steps):
    pipeline_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            # model.forward_time = 0
            set_seed(SEED)
            synchronize_device(model.device.type)
            pre = time.time()
            outputs = model.generate(**inputs, generation_config=generation_config)
            synchronize_device(model.device.type)
            pipeline_times.append((time.time() - pre) * 1000)


    if model_id in (
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "llava-hf/llava-interleave-qwen-7b-hf",
    ):
        generate_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
    elif model_id == "THUDM/cogvlm2-llama3-caption":
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        generate_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generate_text, pipeline_times


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def load_video(video_data):
    bridge.set_bridge("torch")
    mp4_stream = video_data
    num_frames = 24
    decord_vr = VideoReader(io.BytesIO(mp4_stream), ctx=cpu(0))

    frame_id_list = None
    total_frames = len(decord_vr)
    timestamps = decord_vr.get_frame_timestamp(np.arange(total_frames))
    timestamps = [i[0] for i in timestamps]
    max_second = round(max(timestamps)) + 1
    frame_id_list = []
    for second in range(max_second):
        closest_num = min(timestamps, key=lambda x: abs(x - second))
        index = timestamps.index(closest_num)
        frame_id_list.append(index)
        if len(frame_id_list) >= num_frames:
            break

    video_data = decord_vr.get_batch(frame_id_list)
    video_data = video_data.permute(3, 0, 1, 2)
    return video_data


def sample_frames(path, num_frames):
    video = cv2.VideoCapture(path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    frames = []
    for i in range(total_frames):
        ret, frame = video.read()
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not ret:
            continue
        if i % interval == 0:
            frames.append(pil_img)
    video.release()
    return frames[:num_frames]


def get_video_inputs(model_id, torch_dtype):
    video_path = "./datasets/sample_demo_1.mp4"
    container = av.open(video_path)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Why is this video funny?"},
                {"type": "video"},
            ],
        },
    ]
    if model_id == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        # sample uniformly 8 frames from the video, can sample more for longer videos
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        videos = read_video_pyav(container, indices)
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=prompt, videos=videos, padding=True, return_tensors="pt"
        ).to(model.device)
        inputs["pixel_values_videos"] = inputs["pixel_values_videos"].to(torch_dtype)
    elif model_id == "llava-hf/llava-interleave-qwen-7b-hf":
        videos = sample_frames(video_path, 6)
        user_prompt = conversation[0]["content"][0]["text"]
        toks = "<image>" * 6
        prompt = (
            "<|im_start|>user"
            + toks
            + f"\n{user_prompt}<|im_end|><|im_start|>assistant"
        )
        inputs = processor(text=prompt, images=videos, return_tensors="pt").to(
            model.device, model.dtype
        )
    elif model_id == "THUDM/cogvlm2-llama3-caption":
        video_data = open(video_path, "rb").read()
        video = load_video(video_data)
        history = []
        prompt = conversation[0]["content"][0]["text"]
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            query=prompt,
            images=[video],
            history=history,
            template_version="chat",
        )
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[inputs["images"][0].to(model.device).to(model.dtype)]],
        }

    return inputs


def get_model_and_proceessor(model_id, torch_dtype, device):
    model, processor, tokenizer = None, None, None
    if model_id == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    elif model_id == "llava-hf/llava-interleave-qwen-7b-hf":
        processor = LlavaProcessor.from_pretrained(model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype
        ).to(device)
    elif model_id == "THUDM/cogvlm2-llama3-caption":
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )

        model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, trust_remote_code=True
            )
            .eval()
            .to(device)
        )

    return model, processor, tokenizer


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id
    device = args.device
    compare_outputs = args.compare_outputs

    # define a chat history and use `apply_chat_template` to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image", "video")
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    model, processor, tokenizer = get_model_and_proceessor(
        model_id, torch_dtype, device
    )
    # wrap_forward_for_benchmark(model)
    inputs = get_video_inputs(model_id, torch_dtype)

    generation_config = model.generation_config
    generation_config.do_sample = args.do_sample
    generation_config.use_cache = True
    generation_config.temperature = 1.0
    generation_config.max_new_tokens = 10
    generation_config.min_new_tokens = 10
    generation_config.top_p = 1.0
    generation_config.cache_implementation = "static"

    if compare_outputs:
        eager_outputs, _ = generate(model, inputs, 0, 1)

    if args.jit:
        raise ValueError("Image-feature-extraction does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        # import torch._dynamo.config
        # torch._dynamo.config.capture_scalar_outputs = True
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        model.forward = torch.compile(model.forward, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        model = ipex.optimize(model, dtype=torch_dtype, inplace=True)

    if compare_outputs:
        optimized_outputs, _ = generate(model, inputs, 0, 1)

        similarity_score = compute_sentence_similarity(eager_outputs, optimized_outputs)
        logging.info(f"similarity (sentence similarity): {similarity_score}")
        assert similarity_score > 0.99

    generate_text, pipeline_times = generate(model, inputs, warm_up_steps, run_steps)

    log_latency(pipeline_times, warm_up_steps, run_steps)
    logging.info(f"output = {generate_text}")
