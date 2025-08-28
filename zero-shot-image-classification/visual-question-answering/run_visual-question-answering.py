import os
import sys
import torch
import time
import logging
logging.basicConfig(level=logging.INFO)

from PIL import Image
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


def generate(generator, raw_image, question, warm_up_steps, run_steps):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            set_seed(SEED)
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            output = generator(raw_image, question, topk=1)
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

    image_path = "./datasets/vqa_cats.jpg"
    raw_image = Image.open(image_path).convert("RGB")
    question = "how many dogs are in the picture?"

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    pipe = pipeline(
        "visual-question-answering",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    wrap_forward_for_benchmark(pipe)

    if compare_outputs:
        eager_outputs, _, _ = generate(pipe, raw_image, question, 0, 1)

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

    if compare_outputs:
        optimized_outputs, _, _ = generate(pipe, raw_image, question, 0, 1)

        if "score" in eager_outputs[0]:
            mae = compute_dict_outputs_mae(eager_outputs, optimized_outputs)
            logging.info(f"similarity (1 - MAE): {1 - mae}")
            assert mae < 5e-4
        else:
            assert eager_outputs[0]["answer"] == optimized_outputs[0]["answer"]
            logging.info(f"similarity (sentence similarity): 1.0")

    output, pipeline_times, forward_times = generate(
        pipe, raw_image, question, warm_up_steps, run_steps
    )

    log_latency(pipeline_times, warm_up_steps, run_steps, forward_times)
    logging.info(f"output = {output}")
