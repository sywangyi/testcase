import os
import sys
import torch
import time
import requests
import PIL.Image
import logging
logging.basicConfig(level=logging.INFO)

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


def generate(generator, image, warm_up_steps, run_steps):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            output = generator(image)
            synchronize_device(generator.device.type)
            pipeline_times.append((time.time() - pre) * 1000)
            forward_times.append(generator.forward_time * 1000)

    generate_text = output[0]["generated_text"]
    return generate_text, pipeline_times, forward_times


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    model_id = args.model_id
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    device = args.device
    compare_outputs = args.compare_outputs
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    image_to_text = pipeline(
        "image-to-text",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    wrap_forward_for_benchmark(image_to_text)

    image_url = "https://ankur3107.github.io/assets/images/image-captioning-example.png"
    image = PIL.Image.open(requests.get(image_url, stream=True, timeout=3000).raw)

    if compare_outputs:
        eager_outputs, _, _ = generate(image_to_text, image, 0, 1)

    if args.jit:
        raise ValueError("Image-to-text does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        if "Blip" in image_to_text.model.__class__.__name__:
            image_to_text.model.vision_model.forward = torch.compile(image_to_text.model.vision_model.forward, backend=args.backend)
            image_to_text.model.text_decoder.forward = torch.compile(image_to_text.model.text_decoder.forward, backend=args.backend)
        else:
            image_to_text.model.forward = torch.compile(image_to_text.model.forward, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        image_to_text.model = ipex.optimize(
            image_to_text.model,
            dtype=torch_dtype,
            inplace=True,
        )

    if compare_outputs:
        optimized_outputs, _, _ = generate(image_to_text, image, 0, 1)

        similarity_score = compute_sentence_similarity(eager_outputs, optimized_outputs)
        logging.info(f"similarity (sentence similarity): {similarity_score}")
        assert similarity_score > 0.99

    generate_text, pipeline_times, forward_times = generate(image_to_text, image, warm_up_steps, run_steps)

    log_latency(pipeline_times, warm_up_steps, run_steps, forward_times)
    logging.info(f"output = {generate_text}")
