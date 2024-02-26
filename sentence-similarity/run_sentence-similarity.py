import torch
import time
import sys
import logging
from transformers import pipeline
import torch.nn.functional as F
from transformers.utils import ContextManagers

import os

sys.path.append(os.path.dirname(__file__) + "/..")

from common import get_args, get_torch_dtype, wrap_forward_for_benchmark

logging.basicConfig(level=logging.INFO)
SEED = 20
SENTENCES = ["This is an example sentence", "Each sentence is converted"]
CHI_SENTENCES = ["如何更换花呗绑定银行卡", "花呗更改绑定银行卡"]

MODEL_INPUT_SIZE = {
    "input_ids": (1, 7),
    "token_type_ids": (1, 7),
    "attention_mask": (1, 7),
}
inference_context = [torch.inference_mode()]


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def benchmark(extractor, sentences, seed, nb_pass):
    elapsed_times = []
    forward_times = []
    for _ in range(nb_pass):
        torch.manual_seed(seed)
        extractor.forward_time = 0
        start = time.time()
        encoded_input = extractor.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        model_output = extractor(sentences, return_tensors=True, batch_size=2)
        sentence_embeddings_1 = F.normalize(
            mean_pooling(model_output[0], encoded_input["attention_mask"][0])
        )
        sentence_embeddings_2 = F.normalize(
            mean_pooling(model_output[1], encoded_input["attention_mask"][1])
        )
        score = torch.inner(sentence_embeddings_1, sentence_embeddings_2)
        duration = time.time() - start
        elapsed_times.append(duration * 1000)
        forward_times.append(extractor.forward_time * 1000)
        logging.info(score)
    return elapsed_times, forward_times


def prepare_jit_inputs(model_id, device):
    input_ids_example = torch.randint(6000, size=MODEL_INPUT_SIZE["input_ids"]).to(
        device
    )
    attention_mask_example = torch.randint(
        1, size=MODEL_INPUT_SIZE["attention_mask"]
    ).to(device)
    example_inputs = {
        "input_ids": input_ids_example,
        "attention_mask": attention_mask_example,
    }

    if model_id != "sentence-transformers/all-mpnet-base-v2":
        token_type_ids_example = torch.randint(
            1, size=MODEL_INPUT_SIZE["token_type_ids"]
        ).to(device)
        example_inputs["token_type_ids"] = token_type_ids_example

    return example_inputs


def apply_jit_trace(extractor, model_id, device):
    logging.info("using jit trace for acceleration...")
    example_inputs = prepare_jit_inputs(model_id, device)

    extractor.model.config.return_dict = False
    with ContextManagers(inference_context):
        extractor.model = torch.jit.trace(
            extractor.model, example_kwarg_inputs=example_inputs, strict=False
        )

    extractor.model = torch.jit.freeze(extractor.model.eval())

    extractor.model(**example_inputs)
    extractor.model(**example_inputs)

    return extractor


def optimize_with_ipex(extractor, dtype):
    logging.info("using ipex optimize for acceleration...")
    import intel_extension_for_pytorch as ipex

    extractor.model = ipex.optimize(extractor.model, dtype=dtype)
    return extractor


def apply_torch_compile(extractor, backend):
    logging.info(f"using torch compile with {backend} backend for acceleration...")
    if backend == "ipex":
        import intel_extension_for_pytorch as ipex
    extractor.model = torch.compile(extractor.model, backend=backend)
    return extractor


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

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    if "shibing624/text2vec-base-chinese" in model_id:
        sentences = CHI_SENTENCES
    else:
        sentences = SENTENCES

    extractor = pipeline(
        "feature-extraction",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
        return_dict=False,
    )
    wrap_forward_for_benchmark(extractor)

    if use_ipex_optimize:
        extractor = optimize_with_ipex(extractor, dtype=torch_dtype)
    if use_jit:
        extractor = apply_jit_trace(
            extractor, model_id, device=device
        )
    if use_torch_compile:
        extractor = apply_torch_compile(extractor, backend)

    with ContextManagers(inference_context):
        elapsed_times, forward_times = benchmark(
            extractor, sentences, SEED, warm_up_steps + run_steps
        )

    average_time = sum(elapsed_times[warm_up_steps:]) / run_steps
    average_fwd_time = sum(forward_times[warm_up_steps:]) / run_steps
    logging.info(f"total time [ms]: {elapsed_times}")
    logging.info(
        f"pipeline average time [ms] {average_time}, average fwd time [ms] {average_fwd_time}"
    )
