import os
import sys
import torch
import time
import logging
logging.basicConfig(level=logging.INFO)

from PIL import Image
from transformers import pipeline
from transformers.utils import ContextManagers

sys.path.append(os.path.dirname(__file__) + "/..")
from common import (
    get_args,
    get_torch_dtype,
    wrap_forward_for_benchmark,
    synchronize_device,
    compute_sentence_similarity,
    log_latency,
)

inference_context = [torch.inference_mode()]


def generate(generator, messages, warm_up_steps, run_steps):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            output = generator(text=messages, generate_kwargs=generate_kwargs, return_full_text=False)
            synchronize_device(generator.device.type)
            pipeline_times.append((time.time() - pre) * 1000)
            forward_times.append(generator.forward_time * 1000)

    generate_text = output[0]["generated_text"]
    return generate_text, pipeline_times, forward_times


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id
    device = args.device
    compare_outputs = args.compare_outputs

    image_path = "./datasets/demo.jpeg"
    raw_image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": raw_image,
                },
                {"type": "text", "text": "Describe this image."},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "A dog"},
            ],
        },
    ]

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    wrap_forward_for_benchmark(pipe)

    generation_config = pipe.model.generation_config
    generation_config.do_sample = args.do_sample
    generation_config.use_cache = True
    generation_config.temperature = 1.0
    generation_config.max_new_tokens = 20
    generation_config.min_new_tokens = 20
    generation_config.top_p = 1.0
    # TODO: This check should be removed once the mllama supports static cache.
    if pipe.model._can_compile_fullgraph:
        generation_config.cache_implementation = "static"
    generate_kwargs = {"generation_config": generation_config}

    if compare_outputs:
        eager_outputs, _, _ = generate(pipe, messages, 0, 1)

    if args.jit:
        raise ValueError("Image-feature-extraction does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        if pipe.model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
            pipe.model.model.language_model.forward = torch.compile(
                pipe.model.model.language_model.forward, backend=args.backend
            )
        else:
            pipe.model.forward = torch.compile(pipe.model.forward, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        pipe.model = ipex.optimize(pipe.model, dtype=torch_dtype, inplace=True)

    if compare_outputs:
        optimized_outputs, _, _ = generate(pipe, messages, 0, 1)

        similarity_score = compute_sentence_similarity(eager_outputs, optimized_outputs)
        logging.info(f"similarity (sentence similarity): {similarity_score}")
        assert similarity_score > 0.99

    generate_text, pipeline_times, forward_times = generate(pipe, messages, warm_up_steps, run_steps)

    log_latency(pipeline_times, warm_up_steps, run_steps, forward_times)
    logging.info(f"output = {generate_text}")
