import time
import torch
import json
from transformers import pipeline, AutoTokenizer
import logging
from transformers.utils import ContextManagers

logging.basicConfig(level=logging.INFO)

import sys
import os

sys.path.append(os.path.dirname(__file__) + "/..")

from common import (
    get_args,
    get_torch_dtype,
    get_bitsandbytes_config,
    wrap_forward_for_benchmark,
    synchronize_device,
    get_batched_prompts,
)

inference_context = [torch.no_grad()]

MODEL_LIST = [
    "gpt-j",
    "llama",
    "gpt-neox",
    "opt",
    "falcon",
    "bloom",
    "baichuan",
    "t5",
    "gpt2",
]

def generate(generator, input_sentence, batch_size, warm_up_steps, run_steps):
    latency = []
    forward_latency = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            output = generator(
                input_sentence, batch_size=batch_size, generation_config=generation_config
            )
            synchronize_device(generator.device.type)
            latency.append((time.time() - pre) * 1000)
            forward_latency.append(generator.forward_time * 1000)

    return (
        sum(latency[warm_up_steps:]) / run_steps,
        output,
        sum(forward_latency[warm_up_steps:]) / run_steps,
    )


def benchmark(
    generator, warm_up_steps, run_steps, input_sentence, output_tokens, batch_size
):
    input_len = len(tokenizer(input_sentence[0])["input_ids"])
    logging.info(f"input tokens length is {input_len}")

    generation_config.max_new_tokens = 1
    generation_config.min_new_tokens = 1

    first_latency, out, _ = generate(
        generator, input_sentence, batch_size, warm_up_steps, run_steps
    )

    logging.info(f"1st token latency = {first_latency} ms")
    logging.info(f"output token nums = {batch_size}")

    generation_config.max_new_tokens = output_tokens
    generation_config.min_new_tokens = output_tokens
    latency, out, forward_latency = generate(
        generator, input_sentence, batch_size, warm_up_steps, run_steps
    )
    out_num = out_num = len(tokenizer(out[0]["summary_text"])["input_ids"])
    logging.info(
        f"2nd+ token latency = {(latency - first_latency) / (out_num - 1)} ms"
    )
    logging.info(f"output token nums = {out_num*batch_size}")
    logging.info(f"output = {out}")
    logging.info(
        f"pipeline average time [ms] {latency}, average fwd time [ms] {forward_latency}"
    )


if __name__ == "__main__":
    args = get_args()
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id

    logging.info(f"args = {args}")

    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    with open("./datasets/prompt.json", "r") as f:
        prompt = json.load(f)

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))
    
    model_kwargs = {}
    quantization_config = get_bitsandbytes_config(args.quant_type)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    generator = pipeline(
        "summarization",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device if quantization_config is None else None,
        tokenizer=tokenizer,
        model_kwargs=model_kwargs,
    )
    generation_config = generator.model.generation_config
    generation_config.do_sample = False
    generation_config.use_cache = True
    generation_config.temperature = 1.0
    generation_config.num_beams = args.num_beams
    generation_config.max_new_tokens = args.output_tokens
    generation_config.min_new_tokens = args.output_tokens
    generation_config.top_p = 1.0
    # Bart model only support list type of kv-cache, will enable it after static cache implementation done.
    # See https://github.com/huggingface/transformers/issues/28981
    # generation_config.cache_implementation="static"

    wrap_forward_for_benchmark(generator)

    model = [name for name in MODEL_LIST if name in model_id.lower()]
    if len(model) == 0:
        model = ["gpt-j"]

    prompt = prompt[model[0]][str(args.input_tokens)]
    input_seq = get_batched_prompts(prompt, args.batch_size)

    if args.optimum_intel:
        from optimum.intel import IPEXModelForCausalLM
        generator.model = IPEXModelForCausalLM(
            generator.model, export=True, torch_dtype=torch_dtype
        )
    elif args.ipex_optimize_transformers:
        import intel_extension_for_pytorch as ipex
        generator.model = ipex.optimize_transformers(
            generator.model, dtype=torch_dtype, device=device
        )
    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        from torch._inductor import config
        torch._inductor.config.cpp_wrapper = True
        # pipeline warmup
        _, _, _ = generate(generator, input_seq, args.batch_size, 1, 1)
        generator.model.forward = torch.compile(
            generator.model.forward, backend=args.backend
        )
        # compile warmup
        _, _, _ = generate(generator, input_seq, args.batch_size, 1, 1)

    benchmark(
            generator,
            warm_up_steps,
            run_steps,
            input_seq,
            output_tokens=args.output_tokens,
            batch_size=args.batch_size,
        )
