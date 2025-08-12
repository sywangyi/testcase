import os
import sys
import torch
import time
import PIL
import logging
import requests
logging.basicConfig(level=logging.INFO)

from transformers import set_seed
from transformers.utils import ContextManagers
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionInpaintPipeline,
)

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
PROMPT_WITH_MASK_IMAGE = "Face of a yellow cat, high resolution, sitting on a park bench"
DOG_BENCH = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
DOG_BENCH_MASK = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
MODEL_INPUT_SIZE = {
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "sample": (2, 4, 128, 128),
        "timestep": 1.0,
        "encoder_hidden_states": (2, 77, 2048),
        "text_embeds": (2, 1280),
        "time_ids": (2, 6),
    },
    "runwayml/stable-diffusion-v1-5": {
        "sample": (2, 4, 64, 64),
        "timestep": 90,
        "encoder_hidden_states": (2, 77, 768),
    },
    "stabilityai/stable-diffusion-2-1": {
        "sample": (2, 4, 96, 96),
        "timestep": 90,
        "encoder_hidden_states": (2, 77, 1024),
    },
}


def load_model(model_id, model_dtype, device):
    set_seed(SEED)
    if model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype, use_safetensors=True
        )
    elif model_id in ["runwayml/stable-diffusion-v1-5", "stable-diffusion-v1-5/stable-diffusion-v1-5"]:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype
        )
    elif model_id == "stabilityai/stable-diffusion-2-1":
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif model_id == "stable-diffusion-v1-5/stable-diffusion-inpainting":
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype
        )
    else:
        raise ValueError("the given model id is not supported currently.")
    pipe.to(device)
    return pipe


def benchmark(pipe, model_id, nb_pass, prompt, **kwargs):
    image = kwargs.get("image", None)
    mask_image = kwargs.get("mask_image", None)
    elapsed_time = []
    for i in range(nb_pass):
        synchronize_device(pipe.device.type)
        start = time.time()
        set_seed(SEED)
        if model_id == "stable-diffusion-v1-5/stable-diffusion-inpainting":
            image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
        else:
            image = pipe(prompt=prompt).images[0]
        synchronize_device(pipe.device.type)
        duration = time.time() - start
        elapsed_time.append(duration * 1000)

    return elapsed_time, image


def prepare_image():
    image = PIL.Image.open(requests.get(DOG_BENCH, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image).convert("RGB")
    mask_image = PIL.Image.open(requests.get(DOG_BENCH_MASK, stream=True).raw)
    mask_image = PIL.ImageOps.exif_transpose(mask_image).convert("RGB")
    return image, mask_image


def prepare_inputs(model_id, jit, dtype, device):
    sample_size = MODEL_INPUT_SIZE[model_id]["sample"]
    timestep_size = MODEL_INPUT_SIZE[model_id]["timestep"]
    encoder_hidden_states_size = MODEL_INPUT_SIZE[model_id]["encoder_hidden_states"]

    sample_example = torch.randn(sample_size, dtype=dtype).to(device)
    timestemp_dtype = torch.int64 if isinstance(timestep_size, int) else torch.float32
    timestep_example = torch.tensor(timestep_size, dtype=timestemp_dtype).to(device)
    encoder_hidden_states_example = torch.randn(
        encoder_hidden_states_size, dtype=dtype
    ).to(device)

    example_inputs = {
        "sample": sample_example,
        "timestep": timestep_example,
        "encoder_hidden_states": encoder_hidden_states_example,
    }
    if not jit:
        example_inputs = tuple(example_inputs.values())

    if model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        text_embeds_size = MODEL_INPUT_SIZE[model_id]["text_embeds"]
        time_ids_size = MODEL_INPUT_SIZE[model_id]["time_ids"]
        text_embeds_example = torch.randn(text_embeds_size, dtype=dtype).to(device)
        time_ids_example = torch.randn(time_ids_size, dtype=dtype).to(device)

        if jit:
            example_inputs["added_cond_kwargs"] = {
                "text_embeds": text_embeds_example,
                "time_ids": time_ids_example,
            }
        else:
            example_inputs = (
                example_inputs
                + (None,) * 4
                + ({"text_embeds": text_embeds_example, "time_ids": time_ids_example},)
            )
    return example_inputs


def apply_jit_trace(pipeline, model_id, attr_list, dtype, device):
    logging.info("using jit trace for acceleration...")
    for name in attr_list:
        model = getattr(pipeline, name)
        model.eval()
        example_inputs = prepare_inputs(model_id, True, dtype, device)

        with ContextManagers(inference_context):
            traced_model = torch.jit.trace(
                model, example_kwarg_inputs=example_inputs, strict=False
            )

        traced_model = torch.jit.freeze(traced_model.eval())

        traced_model(**example_inputs)
        traced_model(**example_inputs)

        setattr(pipeline, name, traced_model)

    return pipeline


def optimize_with_ipex(pipe, model_id, dtype, device):
    logging.info("using ipex optimize for acceleration...")
    import intel_extension_for_pytorch as ipex

    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)

    input_example = prepare_inputs(model_id, False, dtype, device)

    # optimize with IPEX
    pipe.unet = ipex.optimize(
        pipe.unet.eval(), dtype=dtype, inplace=True, sample_input=input_example
    )
    pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=dtype, inplace=True)
    with ContextManagers(inference_context):
        pipe.text_encoder = ipex.optimize(
            pipe.text_encoder.eval(), dtype=dtype, inplace=True
        )

    return pipe


def apply_torch_compile(pipe, backend):
    logging.info(f"using torch compile with {backend} backend for acceleration...")
    if backend == "ipex":
        import intel_extension_for_pytorch as ipex
    pipe.unet.forward = torch.compile(pipe.unet.forward, backend=backend)

    return pipe


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
    compare_outputs = args.compare_outputs

    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    pipe = load_model(model_id, model_dtype=torch_dtype, device=device)

    image = None
    mask_image = None
    prompt = PROMPT
    if model_id == "stable-diffusion-v1-5/stable-diffusion-inpainting":
        image, mask_image = prepare_image()
        prompt = PROMPT_WITH_MASK_IMAGE

    if compare_outputs:
        _, eager_image = benchmark(pipe, model_id, 1, prompt, image=image, mask_image=mask_image)

    if use_ipex_optimize:
        pipe = optimize_with_ipex(
            pipe, model_id, dtype=torch_dtype, device=device
        )
    if use_jit:
        pipe = apply_jit_trace(
            pipe, model_id, ["unet"], dtype=torch_dtype, device=device
        )
    if use_torch_compile:
        pipe = apply_torch_compile(pipe, backend)

    if compare_outputs:
        _, optimized_image = benchmark(pipe, model_id, 1, prompt, image=image, mask_image=mask_image)

        ssim_score = compute_ssim(eager_image, optimized_image)
        logging.info(f"similarity (SSIM): {ssim_score}")
        threshold = 0.9 if model_id == "stabilityai/stable-diffusion-xl-base-1.0" else 0.98
        assert ssim_score > threshold

    with ContextManagers(inference_context):
        elapsed_time, image = benchmark(pipe, model_id, warm_up_steps + run_steps, prompt, image=image, mask_image=mask_image)


    log_latency(elapsed_time, warm_up_steps, run_steps)
    image_path = "image_" + model_id.replace("/", "_") + ".jpg"
    image.save(image_path, "JPEG")