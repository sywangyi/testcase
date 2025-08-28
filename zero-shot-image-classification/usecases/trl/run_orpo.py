import os
import argparse

import torch
import shutil
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format


def load_data(tokenizer, seed):

    dataset_name = "mlabonne/orpo-dpo-mix-40k"
    dataset = load_dataset(dataset_name, split="all")
    dataset = dataset.shuffle(seed=seed).select(
        range(1000)
    )  # Only use 1000 samples for quick demo

    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=os.cpu_count(),
    )
    dataset = dataset.train_test_split(test_size=0.01)

    return dataset


def main(args):
    set_seed(args.seed)

    base_model = args.base_model
    torch_dtype = torch.bfloat16

    # LoRA config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "q_proj",
            "v_proj",
            "o_proj",
        ],
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_type,
    )

    model, tokenizer = setup_chat_format(model, tokenizer)

    tmp_model_dir = "tmp_model"
    model.save_pretrained(tmp_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        tmp_model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=args.attn_type,
    )
    model.gradient_checkpointing_enable()
    shutil.rmtree(tmp_model_dir)

    dataset = load_data(tokenizer, args.seed)

    orpo_args = ORPOConfig(
        learning_rate=8e-6,
        lr_scheduler_type="linear",
        max_length=1024,
        max_prompt_length=512,
        beta=0.1,
        per_device_train_batch_size=args.batch_size_per_device,
        per_device_eval_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="adamw_hf",
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="steps",
        eval_steps=0.2,
        logging_steps=args.logging_steps,
        warmup_steps=10,
        report_to=args.report_to,
        output_dir=args.checkpoint_dir,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.model_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ORPO with TRL")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Model to be fine-tuned",
    )
    parser.add_argument(
        "--model_save_dir",
        type=str,
        default="OrpoLlama-3-8B",
        help="Local path to save the fine-tuned model",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="results",
        help="Local path to save the model checkpoints during training",
    )
    parser.add_argument(
        "--attn_type",
        type=str,
        choices=["eager", "sdpa"],
        default="eager",
        help="Attention implementation",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        choices=["wandb", "tensorboard", "all", "none"],
        default="none",
        help="The platform to report the training results",
    )
    parser.add_argument(
        "--batch_size_per_device",
        type=int,
        default=2,
        help="Per device batch_size for both training and evaluation datasets",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Number of training epoch"
    )
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")

    args = parser.parse_args()
    main(args)
