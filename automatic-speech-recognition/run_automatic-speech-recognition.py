import sys
import os
import time
import torch
import logging
import numpy as np
logging.basicConfig(level=logging.INFO)

from transformers import pipeline, set_seed
from transformers.utils import ContextManagers

sys.path.append(os.path.dirname(__file__) + "/..")
from common import (
    SEED,
    get_args,
    get_torch_dtype,
    wrap_forward_for_benchmark,
    synchronize_device,
    compute_rouge,
    log_latency,
)

inference_context = [torch.inference_mode()]


def generate(generator, pipe_input, warm_up_steps, run_steps, generate_kwargs=None):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            set_seed(SEED)
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            if generate_kwargs:
                output = generator(pipe_input, generate_kwargs=generate_kwargs)
            else:
                output = generator(pipe_input)
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

    if "pyannote" not in model_id:
        generator = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            torch_dtype=torch_dtype,
        )
        generate_kwargs = None
        if generator.model.can_generate():
            generation_config = generator.model.generation_config
            generation_config.cache_implementation="static"
            generate_kwargs = {"generation_config": generation_config}
        wrap_forward_for_benchmark(generator)

        input_data = np.load("./datasets/asr.npy")
        logging.info(f"input_data: {input_data}")
        logging.info("Label: I would like to set up a joint account with my partner")

        if compare_outputs:
            eager_outputs, _, _ = generate(generator, input_data, 0, 1, generate_kwargs=generate_kwargs)

        if args.torch_compile:
            if args.backend == "ipex":
                import intel_extension_for_pytorch as ipex
            logging.info(f"using torch compile with {args.backend} backend")
            generator.model.forward = torch.compile(generator.model.forward, backend=args.backend)
            if model_id == "jonatasgrosman/wav2vec2-large-xlsr-53-english":
                generator.preprocess = torch.compile(generator.preprocess, backend=args.backend)
        elif args.ipex_optimize:
            import intel_extension_for_pytorch as ipex

            logging.info("Use ipex optimize")
            generator.model = ipex.optimize(generator.model, dtype=torch_dtype, inplace=True)

        if compare_outputs:
            optimized_outputs, _, _ = generate(generator, input_data, 0, 1, generate_kwargs=generate_kwargs)

            rouge_fmeasure = compute_rouge(eager_outputs["text"], optimized_outputs["text"])
            logging.info(f"similarity (rouge_fmeasure): {rouge_fmeasure}")
            assert rouge_fmeasure > 0.99 # Same output

        output, pipeline_times, forward_times = generate(
            generator, input_data, warm_up_steps, run_steps, generate_kwargs=generate_kwargs
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
            generator.model.forward = torch.compile(generator.model.forward, backend=args.backend)
        elif args.ipex_optimize:
            logging.info("Pyannote do not support ipex optimize")

        output, pipeline_times, forward_times = generate(
            generator, "./datasets/speech.wav", warm_up_steps, run_steps
        )

    log_latency(pipeline_times, warm_up_steps, run_steps, forward_times)
    logging.info(f"output = {output}")
