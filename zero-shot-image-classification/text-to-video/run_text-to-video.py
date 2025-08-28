import os
import sys
import torch
import time
import logging
logging.basicConfig(level=logging.INFO)

from transformers import set_seed
from transformers.utils import ContextManagers
from diffusers import (
    AnimateDiffPipeline,
    CogVideoXPipeline,
    MotionAdapter,
    EulerDiscreteScheduler,
)
from diffusers.utils import export_to_video
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

sys.path.append(os.path.dirname(__file__) + "/..")
from common import (
    SEED,
    get_args,
    get_torch_dtype,
    synchronize_device,
    compute_ssim,
    log_latency,
)

inference_context = [torch.no_grad()]
PROMPT = "An astronaut riding a green horse"


def load_model(model_id, model_dtype, device):
    set_seed(SEED)
    if model_id == "ByteDance/AnimateDiff-Lightning":
        step = 4  # Options: [1,2,4,8]
        ckpt = f"animatediff_lightning_{step}step_diffusers.safetensors"
        base = "emilianJR/epiCRealism"  # Choose to your favorite base model.
        adapter = MotionAdapter().to(device, model_dtype)
        adapter.load_state_dict(load_file(hf_hub_download(model_id ,ckpt), device=device))
        pipe = AnimateDiffPipeline.from_pretrained(base, motion_adapter=adapter, torch_dtype=model_dtype).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")
    elif model_id in ("THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"):
        pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=model_dtype).to(device)
    else:
        raise ValueError("the given model id is not supported currently.")

    return pipe


def benchmark(pipe, prompt, nb_pass):
    elapsed_time = []
    for i in range(nb_pass):
        synchronize_device(pipe.device.type)
        start = time.time()
        set_seed(SEED)
        if model_id == "ByteDance/AnimateDiff-Lightning":
            output = pipe(prompt=prompt, guidance_scale=1.0, num_inference_steps=4).frames[0]
        elif model_id in ("THUDM/CogVideoX-2b", "THUDM/CogVideoX-5b"):
            output = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=20,
                num_frames=10,
                guidance_scale=6,
                generator=torch.Generator(device="cpu").manual_seed(SEED),
            ).frames[0]
        synchronize_device(pipe.device.type)
        duration = time.time() - start
        elapsed_time.append(duration * 1000)

    return elapsed_time, output


def apply_torch_compile(pipe, backend):
    logging.info(f"using torch compile with {backend} backend for acceleration...")
    if backend == "ipex":
        import intel_extension_for_pytorch as ipex
    if pipe.__class__.__name__ == "AnimateDiffPipeline":
        pipe.unet.forward = torch.compile(pipe.unet.forward, backend=backend)
    elif pipe.__class__.__name__ == "CogVideoXPipeline":
        pipe.transformer.forward = torch.compile(pipe.transformer.forward, backend=backend)

    return pipe


if __name__ == "__main__":
    args = get_args()
    logging.info(f"args={args}")
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id
    use_ipex_optimize = args.ipex_optimize
    use_torch_compile = args.torch_compile
    backend = args.backend
    device = args.device
    compare_outputs = args.compare_outputs

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    pipe = load_model(model_id, model_dtype=torch_dtype, device=device)

    if compare_outputs:
        _, eager_video = benchmark(pipe, PROMPT, 1)

    if use_torch_compile:
        pipe = apply_torch_compile(pipe, backend)

    if compare_outputs:
        _, optimized_video = benchmark(pipe, PROMPT, 1)
        ssim_score = compute_ssim(eager_video, optimized_video)
        logging.info(f"similarity (SSIM): {ssim_score}")
        threshold = 0.95 if model_id == "ByteDance/AnimateDiff-Lightning" else 0.9
        assert ssim_score > 0.9

    with ContextManagers(inference_context):
        elapsed_time, output = benchmark(pipe, PROMPT, warm_up_steps + run_steps)

    log_latency(elapsed_time, warm_up_steps, run_steps)
    video_path = "video_" + model_id.replace("/", "_") + ".mp4"
    export_to_video(output, video_path, fps=8)
