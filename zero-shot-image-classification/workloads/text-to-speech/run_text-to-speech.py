import os
import sys
import torch
import time
import soundfile
import logging

logging.basicConfig(level=logging.INFO)

from datasets import load_from_disk
from transformers import pipeline, set_seed
from transformers.utils import ContextManagers

sys.path.append(os.path.dirname(__file__) + "/..")
from common import (
    SEED,
    get_args,
    get_torch_dtype,
    wrap_forward_for_benchmark,
    synchronize_device,
    log_latency,
)

inference_context = [torch.inference_mode()]
PROMPT = "Hello, my dog is cooler than you!"


def generate(generator, forward_params, warm_up_steps, run_steps):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(run_steps + warm_up_steps):
            set_seed(SEED)
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            output = generator(PROMPT, forward_params=forward_params)
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
    synthesiser = pipeline(
        "text-to-speech", model_id, device=device, torch_dtype=torch_dtype
    )
    wrap_forward_for_benchmark(synthesiser)

    embeddings_dataset = load_from_disk("./datasets/speech_vector")
    speaker_embedding = (
        torch.tensor(embeddings_dataset[0]["xvector"])
        .unsqueeze(0)
        .to(device)
        .to(torch_dtype)
    )
    # by default the dtype of speaker_embedding is FP32, if the model dtype is not FP32, we need to manually convert it
    if torch_dtype != torch.float32:
        speaker_embedding = speaker_embedding.to(torch_dtype)

    # You can replace this embedding with your own as well.
    forward_params = (
        {"speaker_embeddings": speaker_embedding} if "t5" in model_id else {}
    )
    if "Bark" in synthesiser.model.__class__.__name__:
        SEED = 777
        forward_params["do_sample"] = True
        forward_params["temperature"] = 0.7
        forward_params["top_k"] = 50
        forward_params["top_p"] = 0.95
    elif synthesiser.model.can_generate():
        forward_params["do_sample"] = False
    if "seamless_m4t" in synthesiser.model.config.model_type:
        forward_params["tgt_lang"] = "eng"

    if compare_outputs:
        eager_outputs, _, _ = generate(synthesiser, forward_params, 0, 1)

    if args.jit:
        raise ValueError("Text-to-speech does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        if "Bark" in synthesiser.model.__class__.__name__:
            synthesiser.model.semantic.forward = torch.compile(
                synthesiser.model.semantic.forward
            )
            synthesiser.model.coarse_acoustics.forward = torch.compile(
                synthesiser.model.coarse_acoustics.forward
            )
            synthesiser.model.fine_acoustics.forward = torch.compile(
                synthesiser.model.fine_acoustics.forward
            )
        elif synthesiser.model.config.model_type == "vits":
            synthesiser.model.decoder.forward = torch.compile(
                synthesiser.model.decoder.forward
            )
        elif "seamless_m4t" in synthesiser.model.config.model_type:
            synthesiser.model = torch.compile(synthesiser.model)
        else:
            synthesiser.model.generate = torch.compile(synthesiser.model.generate)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        synthesiser.model = ipex.optimize(
            synthesiser.model, dtype=torch_dtype, inplace=True
        )

    if compare_outputs:
        optimized_outputs, _, _ = generate(synthesiser, forward_params, 0, 1)

        # The output length might be difference, we can compare the min length.
        eager_outputs = torch.from_numpy(eager_outputs["audio"])
        optimized_outputs = torch.from_numpy(optimized_outputs["audio"])
        min_length = min(eager_outputs.shape[-1], optimized_outputs.shape[-1])
        MAE = torch.nn.L1Loss()
        mae = MAE(
            eager_outputs[..., :min_length], optimized_outputs[..., :min_length]
        ).item()
        logging.info(f"similarity (1 - MAE): {1 - mae}")
        assert mae < 5e-2

    output, pipeline_times, forward_times = generate(
        synthesiser, forward_params, warm_up_steps, run_steps
    )

    log_latency(pipeline_times, warm_up_steps, run_steps, forward_times)
    logging.info(f"output = {output}")

    audio_path = "audio_" + model_id.replace("/", "_") + ".wav"
    audio = output["audio"] if len(output["audio"].shape) == 1 else output["audio"][0]
    soundfile.write(
        audio_path, audio, output["sampling_rate"], format="WAV", subtype="PCM_16"
    )
