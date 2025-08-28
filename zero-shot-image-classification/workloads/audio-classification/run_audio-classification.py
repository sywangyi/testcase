import sys
import os
import time
import torch
import logging
logging.basicConfig(level=logging.INFO)

from datasets import load_dataset
from transformers import pipeline, set_seed
from transformers.utils import ContextManagers

sys.path.append(os.path.dirname(__file__) + "/..")
from common import (
    SEED,
    get_args, 
    get_torch_dtype,
    wrap_forward_for_benchmark,
    synchronize_device,
    compute_dict_outputs_mae,
    log_latency,
)

inference_context = [torch.inference_mode()]


def generate(generator, pipe_input, warm_up_steps, run_steps):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            set_seed(SEED)
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            output = generator(pipe_input, top_k=5)
            synchronize_device(generator.device.type)
            pipeline_times.append((time.time() - pre) * 1000)
            forward_times.append(generator.forward_time * 1000)

    return output, pipeline_times, forward_times


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args = {args}")
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id
    device = args.device
    compare_outputs = args.compare_outputs

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32

    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    if args.jit:
        raise ValueError("Automatic-speech-recognition does not support jit trace")

    generator = pipeline(
        "audio-classification",
        model=model_id,
        device=device,
        torch_dtype=torch_dtype,
    )
    wrap_forward_for_benchmark(generator)
    data = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    logging.info(data[0])

    if compare_outputs:
        eager_outputs, _, _ = generate(generator, data[0]["audio"]["array"], 0, 1)

    if args.torch_compile:
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        logging.info(f"using torch compile with {args.backend} backend")
        generator.model.forward = torch.compile(generator.model.forward, backend=args.backend)
    elif args.ipex_optimize:
        import intel_extension_for_pytorch as ipex

        logging.info("Use ipex optimize")
        generator.model = ipex.optimize(generator.model, dtype=torch_dtype, inplace=True)

    if compare_outputs:
        optimized_outputs, _, _ = generate(generator, data[0]["audio"]["array"], 0, 1)

        mae = compute_dict_outputs_mae(eager_outputs, optimized_outputs)
        logging.info(f"similarity (1 - MAE): {1 - mae}")
        assert mae < 5e-3

    output, pipeline_times, forward_times = generate(
        generator, data[0]["audio"]["array"], warm_up_steps, run_steps
    )

    log_latency(pipeline_times, warm_up_steps, run_steps, forward_times)
    logging.info(f"output = {output}")
