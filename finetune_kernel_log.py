#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from optimum.neuron import NeuronSFTConfig, NeuronSFTTrainer, NeuronTrainingArguments
from optimum.neuron.models.training import NeuronModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--tokenizer_id", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--dataloader_drop_last", type=bool, default=True)
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    return parser.parse_args()

def formatting_func(examples):
    """Format examples for training."""
    outputs = []
    for messages in examples["messages"]:
        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        outputs.append(formatted)
    return outputs

if __name__ == "__main__":
    args = parse_args()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    train_dataset = load_dataset("json", data_files=args.train_data, split="train")
    val_dataset = load_dataset("json", data_files=args.val_data, split="train")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Load model using NeuronModelForCausalLM which supports tensor parallelism
    # The trn_config will be automatically created from NeuronTrainingArguments
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    
    # We'll create the training args first to get the trn_config
    training_args = NeuronTrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        bf16=args.bf16,
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_drop_last=args.dataloader_drop_last,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        disable_tqdm=args.disable_tqdm,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    
    model = NeuronModelForCausalLM.from_pretrained(
        args.model_id,
        training_args.trn_config,
        torch_dtype=dtype,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Create SFT config
    sft_args = training_args.to_dict()
    sft_config = NeuronSFTConfig(
        max_seq_length=1024,
        packing=False,  # kernel logs are already structured
        **sft_args,
    )
    
    # Initialize SFT trainer with LoRA
    trainer = NeuronSFTTrainer(
        args=sft_config,
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    print(f"Training complete! Checkpoints saved to {args.output_dir}")
    
    # Merge LoRA adapters into base model
    # This requires running the consolidation script in a subprocess
    # to properly handle the Neuron model loading
    import subprocess
    import sys
    from torch_xla.core.xla_model import is_master_ordinal
    
    if is_master_ordinal():
        print("Consolidating and merging LoRA adapters...")
        input_ckpt_dir = os.path.join(
            args.output_dir, f"checkpoint-{args.max_steps}"
        )
        output_ckpt_dir = os.path.join(args.output_dir, "merged_model")
        
        # Set up environment for consolidation
        env = os.environ.copy()
        env["NEURON_RT_VISIBLE_CORES"] = f"0-{args.tensor_parallel_size - 1}"
        
        consolidate_script = os.path.join(
            os.path.dirname(__file__),
            "consolidate_adapter_shards_and_merge_model.py"
        )
        
        if os.path.exists(consolidate_script):
            subprocess.run(
                [
                    sys.executable,
                    consolidate_script,
                    "-i",
                    input_ckpt_dir,
                    "-o",
                    output_ckpt_dir,
                ],
                env=env,
                check=True
            )
            print(f"Merged model saved to {output_ckpt_dir}")
        else:
            print(f"Warning: consolidation script not found at {consolidate_script}")
            print("Skipping model merge step. You can merge manually later.")
