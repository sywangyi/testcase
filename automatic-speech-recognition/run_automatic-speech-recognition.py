import time
import torch
from transformers import pipeline
from transformers.utils import ContextManagers
from datasets import load_from_disk

import logging

logging.basicConfig(level=logging.INFO)

import sys
import os

sys.path.append(os.path.dirname(__file__) + "/..")

from common import get_args, get_torch_dtype, wrap_forward_for_benchmark

inference_context = [torch.inference_mode()]


def generate(generator, pipe_input, warm_up_steps, run_steps):
    time_costs = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            generator.forward_time = 0
            pre = time.time()
            output = generator(pipe_input)
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

    data = load_from_disk("./datasets/speech_demo")
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32

    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    if args.jit:
        raise ValueError("Automatic-speech-recognition does not support jit trace")

    if "pyannote" not in model_id:
        generator = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        wrap_forward_for_benchmark(generator)
        logging.info(data["train"][0])

        if args.torch_compile:
            if args.backend == "ipex":
                import intel_extension_for_pytorch as ipex
            logging.info(f"using torch compile with {args.backend} backend")
            generator.model.generate = torch.compile(
                generator.model.generate, backend=args.backend
            )
            generator.model = torch.compile(generator.model, backend=args.backend)
        elif args.ipex_optimize:
            import intel_extension_for_pytorch as ipex

            logging.info("Use ipex optimize")
            generator.model = ipex.optimize(generator.model, dtype=torch_dtype, inplace=True)

        generate(
            generator, data["train"][0]["audio"]["array"], warm_up_steps, run_steps
        )
    else:
        from pyannote.audio import Pipeline

        generator = Pipeline.from_pretrained(model_id)
        generator.to(torch.device(device))
        # numpy does not support bf16
        if args.torch_compile:
            logging.info(f"using torch compile with {args.backend} backend")
            if args.backend == "ipex":
                import intel_extension_for_pytorch as ipex
            generator = torch.compile(generator, backend=args.backend)
        elif args.ipex_optimize:
            logging.info("Pyannote do not support ipex optimize")

        generate(generator, "./datasets/speech.wav", warm_up_steps, run_steps)
