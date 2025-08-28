import os
import sys
import torch
import time
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
    compute_dict_outputs_mae,
    log_latency,
)

inference_context = [torch.inference_mode()]


def prepare_jit_inputs(device):
    example_inputs = {
        "input_ids": torch.randint(100, (1, 19)).to(device),
        "token_type_ids": torch.randint(1, (1, 19)).to(device),
        "attention_mask": torch.randint(1, (1, 19)).to(device),
    }

    return example_inputs


def benchmark(pipe, question, context, warm_up_steps, run_steps):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            pipe.forward_time = 0
            synchronize_device(pipe.device.type)
            pre = time.time()
            output = pipe(question=question, context=context)
            synchronize_device(pipe.device.type)
            pipeline_times.append((time.time() - pre) * 1000)
            forward_times.append(pipe.forward_time * 1000)

    return output, pipeline_times, forward_times


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id
    device = args.device
    compare_outputs = args.compare_outputs

    question = "Where do I live?"
    context = "My name is Merve and I live in İstanbul."

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    pipe = pipeline(
        "question-answering",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    wrap_forward_for_benchmark(pipe)

    if compare_outputs:
        eager_outputs, _, _ = benchmark(pipe, question, context, 0, 1)

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

    if compare_outputs:
        optimized_outputs, _, _ = benchmark(pipe, question, context, 0, 1)

        if isinstance(eager_outputs, dict):
            eager_outputs, optimized_outputs = [eager_outputs], [optimized_outputs]
        mae = compute_dict_outputs_mae(eager_outputs, optimized_outputs)
        logging.info(f"similarity (1 - MAE): {1 - mae}")
        assert mae < 5e-3

    output, pipeline_times, forward_times = benchmark(
        pipe, question, context, warm_up_steps, run_steps
    )

    log_latency(pipeline_times, warm_up_steps, run_steps, forward_times)
    logging.info(f"output = {output}")
