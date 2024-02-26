from transformers import pipeline
import torch
import time
import logging
import requests
import PIL.Image
from transformers.utils import ContextManagers

logging.basicConfig(level=logging.INFO)

import sys
import os

sys.path.append(os.path.dirname(__file__) + "/..")

from common import get_args, get_torch_dtype, wrap_forward_for_benchmark

inference_context = [torch.inference_mode()]

def generate(generator, image, warm_up_steps, run_steps):
    time_costs = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            generator.forward_time = 0
            pre = time.time()
            output = generator(image)
            time_costs.append((time.time() - pre) * 1000)
            forward_times.append(generator.forward_time * 1000)
    average_time = sum(time_costs[warm_up_steps:]) / run_steps
    average_fwd_time = sum(forward_times[warm_up_steps:]) / run_steps
    logging.info(f"total time [ms]: {time_costs}")
    logging.info(
        f"pipeline average time [ms] {average_time}, average fwd time [ms] {average_fwd_time}"
    )
    logging.info(f"output = {output}")


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    model_id = args.model_id
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    image_to_text = pipeline(
        "image-to-text",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )

    wrap_forward_for_benchmark(image_to_text)
    if args.jit:
        raise ValueError("Image-to-text does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        image_to_text.model.generate = torch.compile(
            image_to_text.model.generate, backend=args.backend
        )
        image_to_text.model = torch.compile(image_to_text.model, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        image_to_text.model = ipex.optimize(
            image_to_text.model,
            dtype=torch_dtype,
            inplace=True,
        )

    image_url = "https://ankur3107.github.io/assets/images/image-captioning-example.png"
    image = PIL.Image.open(requests.get(image_url, stream=True, timeout=3000).raw)

    generate(image_to_text, image, warm_up_steps, run_steps)
