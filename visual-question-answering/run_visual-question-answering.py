from PIL import Image
from transformers import pipeline
import torch
import time
import logging
import sys
from transformers.utils import ContextManagers

sys.setrecursionlimit(10000000)

import os

sys.path.append(os.path.dirname(__file__) + "/..")
from common import get_args, get_torch_dtype, wrap_forward_for_benchmark, synchronize_device

logging.basicConfig(level=logging.INFO)

inference_context = [torch.inference_mode()]


def generate(generator, raw_image, question, warm_up_steps, run_steps):
    time_costs = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            output = generator(raw_image, question, topk=1)
            synchronize_device(generator.device.type)
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
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id

    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    image_path = "./datasets/vqa_cats.jpg"
    raw_image = Image.open(image_path).convert("RGB")
    question = "how many dogs are in the picture?"

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    pipe = pipeline(
        "visual-question-answering",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    wrap_forward_for_benchmark(pipe)

    if args.jit:
        raise ValueError("Visual-question-answering does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        if "Blip" in pipe.model.__class__.__name__:
            pipe.model.vision_model.forward = torch.compile(pipe.model.vision_model.forward, backend=args.backend)
            pipe.model.text_encoder.forward = torch.compile(pipe.model.text_encoder.forward, backend=args.backend)
            pipe.model.text_decoder.forward = torch.compile(pipe.model.text_decoder.forward, backend=args.backend)
        else:
            pipe.model.forward = torch.compile(pipe.model.forward, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        pipe.model = ipex.optimize(pipe.model, dtype=torch_dtype, inplace=True)

    generate(pipe, raw_image, question, warm_up_steps, run_steps)
