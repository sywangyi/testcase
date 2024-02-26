import argparse
import torch
import time


def str2bool(str):
    return True if str.lower() == "true" else False


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
    parser.add_argument("--ipex_optimize_transformers", default="False", type=str2bool)
    parser.add_argument("--warm_up_steps", default=10, type=int)
    parser.add_argument("--run_steps", default=10, type=int)
    args = parser.parse_args()
    return args


def get_torch_dtype(dtype):
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    else:
        return torch.float32


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


args = get_args()
