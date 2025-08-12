import os
import sys
import torch
import time
import logging
logging.basicConfig(level=logging.INFO)

from PIL import Image
from transformers import pipeline
from transformers.utils import ContextManagers

sys.path.append(os.path.dirname(__file__) + "/..")
from common import (
    get_args,
    get_torch_dtype,
    wrap_forward_for_benchmark,
    synchronize_device,
    compute_dict_outputs_mae,
    log_latency,
)

inference_context = [torch.inference_mode()]


def generate(generator, raw_image, question, word_boxes, warm_up_steps, run_steps):
    pipeline_times = []
    forward_times = []
    with ContextManagers(inference_context):
        for i in range(warm_up_steps + run_steps):
            generator.forward_time = 0
            synchronize_device(generator.device.type)
            pre = time.time()
            output = generator(raw_image, question, word_boxes=word_boxes, topk=1)
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

    image_path = "./datasets/document.png"
    raw_image = Image.open(image_path).convert("RGB")
    question = "What is the invoice number?"
    word_boxes = (['INVOICE', [74, 45, 305, 96]], ['East', [73, 122, 108, 131]], ['Repair', [113, 122, 165, 134]], ['Inc.', [170, 122, 197, 131]], ['1912', [74, 143, 109, 152]], ['Harvest', [116, 143, 172, 152]], ['Lane', [178, 143, 214, 152]], ['New', [73, 160, 106, 169]], ['York,', [110, 160, 149, 171]], ['NY', [156, 160, 178, 169]], ['12210', [184, 160, 229, 169]], ['BILLTO', [61, 192, 130, 220]], ['SHIP', [349, 196, 386, 211]], ['TO', [390, 196, 410, 211]], ['INVOICE', [617, 196, 678, 211]], ['#', [684, 196, 694, 211]], ['us-001', [872, 201, 925, 211]], ['John', [72, 223, 108, 232]], ['Smith', [114, 222, 156, 231]], ['John', [349, 223, 385, 231]], ['Smith', [390, 222, 433, 231]], ['INVOICE', [617, 221, 678, 235]], ['DATE', [684, 221, 724, 235]], ['1110212019', [845, 226, 928, 235]], ['2', [73, 240, 81, 248]], ['Court', [86, 239, 128, 249]], ['Square', [132, 239, 186, 251]], ['3787', [350, 240, 386, 249]], ['Pineview', [392, 240, 458, 249]], ['Drive', [465, 240, 502, 249]], ['Pose', [617, 245, 654, 260]], ['New', [73, 257, 106, 266]], ['York,', [110, 257, 149, 267]], ['NY', [156, 257, 178, 265]], ['12210', [184, 257, 229, 266]], ['Cambridge,', [350, 257, 436, 268]], ['MA', [442, 257, 466, 266]], ['12210', [473, 257, 517, 266]], ['oy', [625, 257, 644, 260]], ['2312/2019', [849, 250, 928, 260]], ['DUE', [617, 270, 646, 284]], ['DATE', [652, 270, 693, 284]], ['26/02/2019', [844, 275, 928, 284]], ['ay', [84, 324, 130, 352]], ['DESCRIPTION', [289, 324, 400, 352]], ['UNIT', [609, 328, 645, 343]], ['PRICE', [650, 327, 694, 343]], ['‘AMOUNT', [845, 327, 913, 343]], ['1', [110, 363, 116, 373]], ['Front', [172, 363, 209, 373]], ['and', [214, 363, 241, 373]], ['rear', [248, 365, 277, 373]], ['brake', [282, 363, 324, 373]], ['cables', [328, 363, 377, 373]], ['100.00', [634, 354, 696, 382]], ['100.00', [864, 363, 913, 373]], ['2', [109, 393, 118, 403]], ['Newset', [172, 393, 294, 406]], ['of', [230, 385, 250, 413]], ['pedal', [257, 393, 294, 406]], ['arms', [301, 396, 337, 404]], ['15.00', [656, 385, 696, 413]], ['30.00', [872, 393, 913, 404]], ['3', [110, 425, 118, 434]], ['Labor', [172, 425, 214, 434]], ['Shrs', [218, 425, 250, 434]], ['5.00', [652, 415, 696, 443]], ['1.00', [873, 425, 913, 434]], ['Subtotal', [629, 457, 696, 485]], ['145.00', [864, 466, 913, 475]], ['Sales', [569, 496, 610, 506]], ['Tax', [616, 496, 644, 506]], ['6.25%', [649, 496, 694, 507]], ['9.06', [881, 496, 913, 506]], ['TOTAL', [641, 521, 696, 549]], ['$154.06', [837, 524, 913, 540]], ['Smith', [818, 595, 929, 638]], ['TERMS', [520, 858, 573, 872]], ['&', [580, 858, 590, 872]], ['CONDITIONS', [594, 858, 689, 872]], ['Payment', [521, 903, 586, 915]], ['is', [590, 903, 602, 913]], ['due', [608, 903, 634, 913]], ['within', [640, 903, 682, 913]], ['15', [689, 903, 706, 913]], ['days', [710, 896, 749, 922]], ['J', [165, 901, 212, 947]], ['hank', [189, 897, 369, 960]], ['you', [384, 917, 501, 968]], ['Plate', [521, 937, 570, 947]], ['ate', [576, 937, 616, 947]], ['cack', [621, 937, 673, 947]], ['payable:', [678, 937, 758, 949]], ['Est', [766, 937, 798, 947]], ['Rep', [804, 937, 853, 949]])
    torch_dtype = get_torch_dtype(args.model_dtype)
    dtype = get_torch_dtype(args.autocast_dtype)
    apply_cast = dtype != torch.float32
    if apply_cast:
        inference_context.append(torch.autocast(device, dtype, apply_cast))

    pipe = pipeline(
        "document-question-answering",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    wrap_forward_for_benchmark(pipe)

    if compare_outputs:
        eager_outputs, _, _ = generate(pipe, raw_image, question, word_boxes, 0, 1)

    if args.jit:
        raise ValueError("Visual-question-answering does not support jit trace")

    if args.torch_compile:
        logging.info(f"Use torch compile with {args.backend} backend")
        if args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
        if pipe.model.__class__.__name__ == "VisionEncoderDecoderModel":
            pipe.model.encoder.forward = torch.compile(pipe.model.encoder.forward, backend=args.backend)
            # pipe.model.decoder.forward = torch.compile(pipe.model.decoder.forward, backend=args.backend)
        else:
            pipe.model.forward = torch.compile(pipe.model.forward, backend=args.backend)
    elif args.ipex_optimize:
        logging.info("Use ipex optimize")
        import intel_extension_for_pytorch as ipex

        pipe.model = ipex.optimize(pipe.model, dtype=torch_dtype, inplace=True)

    if compare_outputs:
        optimized_outputs, _, _ = generate(pipe, raw_image, question, word_boxes, 0, 1)

        if "score" in eager_outputs[0]:
            mae = compute_dict_outputs_mae(eager_outputs, optimized_outputs)
            logging.info(f"similarity (1 - MAE): {1 - mae}")
            assert mae < 5e-2
        else:
            assert eager_outputs[0]["answer"] == optimized_outputs[0]["answer"]
            logging.info(f"similarity (sentence similarity): 1.0")

    output, pipeline_times, forward_times = generate(
        pipe, raw_image, question, word_boxes, warm_up_steps, run_steps
    )

    log_latency(pipeline_times, warm_up_steps, run_steps, forward_times)
    logging.info(f"output = {output}")
