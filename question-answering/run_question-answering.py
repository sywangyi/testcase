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


def prepare_jit_inputs(device):
    example_inputs = {
        "input_ids": torch.randint(100, (1, 19)).to(device),
        "token_type_ids": torch.randint(1, (1, 19)).to(device),
        "attention_mask": torch.randint(1, (1, 19)).to(device),
    }

    return example_inputs


def benchmark(pipe, question, context, warm_up_steps, run_steps):
    time_costs = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            pipe.forward_time = 0
            synchronize_device(pipe.device.type)
            pre = time.time()
            output = pipe(question=question, context=context)
            synchronize_device(pipe.device.type)
            time_costs.append((time.time() - pre) * 1000)
            forward_times.append(pipe.forward_time * 1000)

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

    question = "Where do I live?"
    context = "My name is Merve and I live in İstanbul."

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    pipe = pipeline(
        "question-answering",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    wrap_forward_for_benchmark(pipe)

    if args.optimum_intel:
        logging.info("Use optimum-intel")
        from optimum.intel import IPEXModelForQuestionAnswering
        pipe.model = IPEXModelForQuestionAnswering(pipe.model, export=True, torch_dtype=torch_dtype)
    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        pipe.model.forward = torch.compile(pipe.model.forward, backend=args.backend)
    if args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        pipe.model = ipex.optimize(pipe.model, dtype=torch_dtype, inplace=True)
    if args.jit:
        logging.info("using jit trace for acceleration...")
        example_inputs = prepare_jit_inputs(device)
        pipe.model.config.return_dict = False
        with ContextManagers(inference_context):
            pipe.model = torch.jit.trace(
                pipe.model, example_kwarg_inputs=example_inputs, strict=False
            )

        pipe.model = torch.jit.freeze(pipe.model.eval())
        pipe.model(**example_inputs)
        pipe.model(**example_inputs)

    benchmark(pipe, question, context, warm_up_steps, run_steps)
