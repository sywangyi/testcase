import argparse
import torch
import time
import random
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from pytorch_msssim import ssim
from rouge_score import rouge_scorer
from transformers import AwqConfig, BitsAndBytesConfig, set_seed
from sentence_transformers import SentenceTransformer, util

SEED = 42
set_seed(SEED)


def str2bool(str):
    return True if str.lower() == "true" else False


def synchronize_device(device):
    if device == "xpu":
        torch.xpu.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None, type=str, required=True)
    parser.add_argument("--autocast_dtype", default="float32", type=str)
    parser.add_argument("--ipex_optimize", default="False", type=str2bool)
    parser.add_argument("--jit", default="False", type=str2bool)
    parser.add_argument("--torch_compile", default="False", type=str2bool)
    parser.add_argument("--model_dtype", default="float32", type=str)
    parser.add_argument("--backend", default="inductor", type=str)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_beams", default=4, type=int)
    parser.add_argument(
        "--input_tokens",
        default=32,
        type=int,
        help="choose from [32, 64, 128, 256, 512, 1024]",
    )
    parser.add_argument("--output_tokens", default=32, type=int)
    parser.add_argument("--do_sample", default="False", type=str2bool)
    parser.add_argument("--ipex_optimize_transformers", default="False", type=str2bool)
    parser.add_argument("--warm_up_steps", default=10, type=int)
    parser.add_argument("--run_steps", default=10, type=int)
    parser.add_argument("--optimum_intel", default="False", type=str2bool)
    parser.add_argument("--optimum_habana", default="False", type=str2bool)
    parser.add_argument("--compare_outputs", default="False", type=str2bool)
    parser.add_argument(
        "--quant_algo",
        default=None,
        type=str,
        help="choose from [bitsandbytes, autoawq, gptqmodel]",
    )
    parser.add_argument(
        "--quant_dtype",
        default=None,
        type=str,
        help="choose from [int, nf4, fp4, int4]",
    )
    parser.add_argument(
        "--tp_plan", default="auto", type=str, help="tensor parallelism strategy"
    )
    parser.add_argument("--enable_ep", default="False", type=str2bool)
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    if args.optimum_habana:
        from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi
        adapt_transformers_to_gaudi()

    return args


def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    else:
        return torch.float32


def get_bitsandbytes_config(quant_type):
    if quant_type == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_type in ("nf4", "fp4"):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_awq_config(quant_type):
    if quant_type == "int4":
        quantization_config = AwqConfig(version="ipex")
    else:
        quantization_config = None

    return quantization_config


def wrapped_forward(self, model_inputs, **forward_params):
    start_time = time.time()
    model_outputs = self.__class__._orig_forward(self, model_inputs, **forward_params)
    end_time = time.time()
    self.forward_time += end_time - start_time
    return model_outputs


def wrap_forward_for_benchmark(pipeline):
    pipeline.forward_time = 0
    pipeline.__class__._orig_forward = pipeline.__class__._forward
    pipeline.__class__._forward = wrapped_forward


def get_batched_prompts(prompt, batch_size):
    prompt_list = [prompt]
    token_list = prompt.split(" ")
    assert len(token_list) > 18
    for _ in range(batch_size - 1):
        prompt_len = random.randint(16, len(token_list) - 2)
        new_prompt = " ".join(token_list[:prompt_len])
        prompt_list.append(new_prompt)
    return prompt_list


def compute_ssim(image_1, image_2):
    X = np.array(image_1).astype(np.float32)
    Y = np.array(image_2).astype(np.float32)
    if len(X.shape) == 3:
        X = torch.from_numpy(X).unsqueeze(0).permute(0, 3, 1, 2)
        Y = torch.from_numpy(Y).unsqueeze(0).permute(0, 3, 1, 2)
    elif len(X.shape) == 4:
        X = torch.from_numpy(X).permute(0, 3, 1, 2)
        Y = torch.from_numpy(Y).permute(0, 3, 1, 2)
    else:
        raise ValueError("The image dim should be 3 for single image or 4 for batched images")
    ssim_score = ssim(X, Y, data_range=255)

    return ssim_score


def compute_dict_outputs_mae(input_1, input_2):
    input_1_dict = {}
    input_2_dict = {}
    if "label" in input_1[0]:
        label = "label"
    elif "answer" in input_1[0]:
        label = "answer"
    else:
        raise ValueError("Unrecognized key")

    for i in range(len(input_1)):
        input_1_dict[input_1[i][label]] = input_1[i]["score"]
        input_2_dict[input_2[i][label]] = input_2[i]["score"]

    mae = [abs(input_1_dict[key] - input_2_dict[key]) for key in input_1_dict.keys()]
    mae = sum(mae) / len(mae)

    return mae


def compute_sentence_similarity(text1, text2):
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Load pre-trained model
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2)

    return similarity.item()


def compute_rouge(text1, text2):
    scorer = rouge_scorer.RougeScorer(["rouge1"])
    scores = scorer.score(text1, text2)

    return scores["rouge1"].fmeasure


def log_latency(pipeline_times, warm_up_steps, run_steps, forward_times=None):
    average_time = sum(pipeline_times[warm_up_steps:]) / run_steps
    logging.info(f"total_time [ms]: {pipeline_times}")
    logging.info(f"pipeline_average_time [ms] {average_time}")

    if forward_times:
        average_fwd_time = sum(forward_times[warm_up_steps:]) / run_steps
        logging.info(f"average_fwd_time [ms] {average_fwd_time}")
