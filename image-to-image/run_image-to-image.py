import PIL
import requests
import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionImageVariationPipeline,
)
import time
from torchvision import transforms
import os
import logging
from transformers.utils import ContextManagers

logging.basicConfig(level=logging.INFO)
import sys

sys.setrecursionlimit(100000)
sys.path.append(os.path.dirname(__file__) + "/..")

from common import get_args, get_torch_dtype

SEED = 20
IMG_URL = "https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg"
SAMPLE_IMAGE = "example.jpg"
PROMPT = "turn him into cyborg"

MODEL_INPUT_SIZE = {
    "timbrooks/instruct-pix2pix": {
        "sample": (3, 8, 64, 64),
        "timestep": 1.0,
        "encoder_hidden_states": (3, 77, 768),
    },
    "stabilityai/stable-diffusion-xl-refiner-1.0": {
        "sample": (2, 4, 64, 64),
        "timestep": 1.0,
        "encoder_hidden_states": (2, 77, 1280),
        "text_embeds": (2, 1280),
        "time_ids": (2, 5),
    },
    "lambdalabs/sd-image-variations-diffusers": {
        "sample": (2, 4, 64, 64),
        "timestep": 90,
        "encoder_hidden_states": (2, 1, 768),
    },
}

inference_context = [torch.no_grad()]


def load_model(model_id, seed, model_dtype, device):
    torch.manual_seed(seed)
    if model_id == "timbrooks/instruct-pix2pix":
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype, safety_checker=None
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    elif model_id == "stabilityai/stable-diffusion-xl-refiner-1.0":
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=model_dtype,
            use_safetensors=True,
            variant="fp16" if model_dtype == torch.float16 else None,
        )
    elif model_id == "lambdalabs/sd-image-variations-diffusers":
        pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            model_id, torch_dtype=model_dtype, revision="v2.0"
        )
    else:
        raise ValueError(
            f"the given model id is not supported currently. it is {model_id}"
        )

    pipe.to(device)

    return pipe


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image


def benchmark(pipe, prompt, image, seed, nb_pass, model_id):
    elapsed_time = []
    for i in range(nb_pass):
        start = time.time()
        torch.manual_seed(seed)
        if model_id == "lambdalabs/sd-image-variations-diffusers":
            new_image = pipe(image, guidance_scale=3).images[0]
        elif model_id == "timbrooks/instruct-pix2pix":
            new_image = pipe(
                prompt, image=image, num_inference_steps=10, image_guidance_scale=1
            ).images[0]
        else:
            new_image = pipe(prompt=prompt, image=image).images[0]
        duration = time.time() - start
        elapsed_time.append(duration * 1000)
        new_image.save(f"img_{i}.jpg", "JPEG")

    return elapsed_time


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

    if model_id == "stabilityai/stable-diffusion-xl-refiner-1.0":
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

    input_example = prepare_inputs(model_id, False, dtype, device)

    # optimize with IPEX
    with ContextManagers(inference_context):
        pipe.unet = ipex.optimize(
            pipe.unet.eval(), dtype=dtype, inplace=True, sample_input=input_example
        )
    pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=dtype, inplace=True)

    return pipe


def apply_torch_compile(pipe, backend):
    logging.info(f"using torch compile with {backend} backend for acceleration...")
    if backend == "ipex":
        import intel_extension_for_pytorch as ipex
    pipe.unet = torch.compile(pipe.unet, backend=backend)
    return pipe


def read_image(model_id, device):
    if model_id == "lambdalabs/sd-image-variations-diffusers":
        image_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "datasets", SAMPLE_IMAGE
        )
        image = PIL.Image.open(image_path)
        tform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=False,
                ),
                transforms.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        image = tform(image).to(device).unsqueeze(0)
    else:
        image = download_image(IMG_URL)

    return image


if __name__ == "__main__":
    args = get_args()
    warm_up_steps = args.warm_up_steps
    run_steps = args.run_steps
    model_id = args.model_id
    use_ipex_optimize = args.ipex_optimize
    use_jit = args.jit
    use_torch_compile = args.torch_compile
    backend = args.backend
    logging.info(f"args = {args}")

    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    image = read_image(model_id, device)
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    enable = dtype != torch.float32
    if enable:
        inference_context.append(torch.autocast(device, dtype, enable))

    pipe = load_model(model_id, SEED, torch_dtype, device)

    if use_ipex_optimize:
        pipe = optimize_with_ipex(
            pipe, model_id, dtype=torch_dtype, device=device
        )
    if use_jit:
        pipe = apply_jit_trace(
            pipe, model_id, ["unet"], dtype=torch_dtype, device=device, enable=enable
        )
    if use_torch_compile:
        pipe = apply_torch_compile(pipe, backend)

    with ContextManagers(inference_context):
        elapsed_time = benchmark(
            pipe, PROMPT, image, SEED, warm_up_steps + run_steps, model_id
        )

    logging.info(f"total time [ms]: {elapsed_time}")
    logging.info(
        f"pipeline average time [ms]: {sum(elapsed_time[warm_up_steps:])/run_steps}"
    )
