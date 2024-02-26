import requests
import torch
import time
import logging
import PIL.Image
from transformers import pipeline
from transformers.utils import ContextManagers

logging.basicConfig(level=logging.INFO)

import os
import sys

sys.path.append(os.path.dirname(__file__) + "/..")
from common import get_args, get_torch_dtype, wrap_forward_for_benchmark

inference_context = [torch.inference_mode()]

SEED = 24
TEXT = ["a photo of a cat", "a photo of a dog"]
IMG_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

MODEL_INPUT_SIZE = {
    "input_ids": (1, 7),
    "pixel_values": (1, 3, 224, 224),
    "attention_mask": (1, 7),
}


def load_model(model_id, seed, model_dtype, device):
    torch.manual_seed(seed)
    classifier = pipeline(
        "zero-shot-image-classification",
        model=model_id,
        torch_dtype=model_dtype,
        device=device,
        return_dict=False,
    )
    return classifier


def benchmark(pipeline, image, labels, seed, nb_pass):
    elapsed_times = []
    forward_times = []
    for _ in range(nb_pass):
        torch.manual_seed(seed)
        pipeline.forward_time = 0
        start = time.time()
        outputs = pipeline(image, candidate_labels=labels)
        duration = time.time() - start
        elapsed_times.append(duration * 1000)
        forward_times.append(pipeline.forward_time * 1000)
        logging.info(outputs)
    return elapsed_times, forward_times


def prepare_jit_inputs(device):
    input_ids_example = torch.randint(200, size=MODEL_INPUT_SIZE["input_ids"]).to(
        device
    )
    pixel_values_example = torch.randn(MODEL_INPUT_SIZE["pixel_values"]).to(device)
    attention_mask_example = torch.randint(
        1, size=MODEL_INPUT_SIZE["attention_mask"]
    ).to(device)

    example_inputs = {
        "input_ids": input_ids_example,
        "pixel_values": pixel_values_example,
        "attention_mask": attention_mask_example,
    }

    return example_inputs


def apply_jit_trace(classifier, device):
    logging.info("using jit trace for acceleration...")
    (
        input_ids_example,
        pixel_values_example,
        attention_mask_example,
    ) = prepare_jit_inputs(device)

    example_inputs = prepare_jit_inputs(device)
    classifier.model.config.return_dict = False
    with ContextManagers(inference_context):
        classifier.model = torch.jit.trace(
            classifier.model, example_kwarg_inputs=example_inputs, strict=False
        )

    classifier.model = torch.jit.freeze(classifier.model.eval())

    classifier.model(**example_inputs)
    classifier.model(**example_inputs)

    return classifier


def optimize_with_ipex(classifier, dtype, device):
    logging.info("using ipex optimize for acceleration...")
    import intel_extension_for_pytorch as ipex

    sample_inputs = tuple(prepare_jit_inputs(device).values())
    with ContextManagers(inference_context):
        classifier.model = ipex.optimize(
            classifier.model,
            dtype=dtype,
            inplace=True,
            sample_input=sample_inputs,
        )

    return classifier


def apply_torch_compile(classifier, backend):
    logging.info(f"using torch compile with {backend} backend for acceleration...")
    if backend == "ipex":
        import intel_extension_for_pytorch as ipex
    classifier.model = torch.compile(classifier.model, backend=backend)
    return classifier


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args={args}")
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id
    use_ipex_optimize = args.ipex_optimize
    use_jit = args.jit
    use_torch_compile = args.torch_compile
    backend = args.backend

    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    dtype = get_torch_dtype(args.autocast_dtype)
    torch_dtype = get_torch_dtype(args.model_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    image = PIL.Image.open(requests.get(IMG_URL, stream=True, timeout=3000).raw)

    classifier = load_model(model_id, SEED, torch_dtype, device)
    wrap_forward_for_benchmark(classifier)

    if use_ipex_optimize:
        classifier = optimize_with_ipex(
            classifier, dtype=torch_dtype, device=device
        )
    if use_jit:
        classifier = apply_jit_trace(
            classifier, device=device
        )
    if use_torch_compile:
        classifier = apply_torch_compile(classifier, backend)

    with ContextManagers(inference_context):
        elapsed_times, forward_times = benchmark(
            classifier, image, TEXT, SEED, warm_up_steps + run_steps
        )

    average_time = sum(elapsed_times[warm_up_steps:]) / run_steps
    average_fwd_time = sum(forward_times[warm_up_steps:]) / run_steps
    logging.info(f"total time [ms]: {elapsed_times}")
    logging.info(
        f"pipeline average time [ms] {average_time}, average fwd time [ms] {average_fwd_time}"
    )
