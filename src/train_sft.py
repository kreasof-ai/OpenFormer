# src/train_sft.py

import os

os.environ["VIZ"] = "0" # Set this to 1 if you want to see performance profiler after training

if False: # Set true to enable Vulkan backend via WebGPU
    os.environ["WEBGPU"] = "1"
    os.environ["WEBGPU_BACKEND"] = "WGPUBackendType_Vulkan"

import json
import argparse
from typing import List
from tqdm import tqdm
import random
import time

# Third-party imports
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from datasets import load_dataset
import wandb 

# tinygrad imports
from tinygrad import Tensor, Device, dtypes, TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import GlobalCounters

from model.lfm2_modeling import LFM2ForCausalLM, BaseAttention, LFM2ConvOperator
from model.qwen3_modeling import Qwen3ForCausalLM
from model import MODEL_MAP
from extra.lora import apply_lora_to_model, get_lora_parameters

# --- Dataset and Preprocessing ---

IGNORE_INDEX = -100
 
def data_generator(dataset, tokenizer, max_length, batch_size):
    """
    A custom data generator that processes, tokenizes, and batches conversational data,
    yielding tinygrad Tensors directly.
    """
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    buffer_prompts, buffer_full_texts = [], []

    for idx in indices:
        item = dataset[idx]
        conversation = item.get("conversations", [])
        if not conversation: continue

        messages = [{"role": "user" if turn.get("from") == "human" else "assistant", "content": turn.get("value", "")} for turn in conversation]
        if not messages or messages[-1]["role"] != "assistant": continue

        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, return_dict=False)
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True, return_dict=False)

        buffer_full_texts.append(full_text)
        buffer_prompts.append(prompt_text)

        if len(buffer_full_texts) == batch_size:
            full_tokenized = tokenizer(buffer_full_texts, max_length=max_length, padding="max_length", truncation=True, return_tensors=None, return_dict=False)
            prompt_tokenized = tokenizer(buffer_prompts, max_length=max_length, truncation=True, return_tensors=None, return_dict=False)
            
            input_ids_batch = full_tokenized['input_ids']
            labels_batch = []

            for j in range(batch_size):
                prompt_len = len(prompt_tokenized['input_ids'][j])
                labels = list(input_ids_batch[j])
                labels[:prompt_len] = [IGNORE_INDEX] * prompt_len
                for k, token_id in enumerate(input_ids_batch[j]):
                    if token_id == tokenizer.pad_token_id:
                        labels[k] = IGNORE_INDEX
                labels_batch.append(labels)
                
            yield (
                Tensor(input_ids_batch, dtype=dtypes.int32, device=Device.DEFAULT),
                Tensor(labels_batch, dtype=dtypes.int32, device=Device.DEFAULT)
            )
            buffer_prompts, buffer_full_texts = [], []

def estimate_mfu_flops(model: LFM2ForCausalLM, batch_size: int, seq_len: int) -> int:
    """Estimates the FLOPS for a forward/backward pass of the LFM2 model."""
    config = model.config
    B, S, H, I, K = batch_size, seq_len, config.hidden_size, config.intermediate_size, config.conv_kernel_size
    fwd_flops = 0
    for layer in model.model.layers:
        fwd_flops += 6 * B * S * H * I
        if isinstance(layer.operator, BaseAttention):
            fwd_flops += 4 * B * S * H * H + 4 * B * (S**2) * H
        elif isinstance(layer.operator, LFM2ConvOperator):
            fwd_flops += (2 * B * S * H * (3 * H)) + (2 * B * S * H * H) + 2 * B * H * S * K
    fwd_flops += 2 * B * S * config.vocab_size * H
    return 3 * fwd_flops


def main(args):
    if args.quantize:
        raise NotImplementedError(
            "Quantization is not supported for training. "
            "Please run SFT without the --quantize flag."
        )
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    print(f"--- Starting SFT Training for {args.model_id} on {Device.DEFAULT} ---")

    if args.use_fp16:
        print("\n--- FP16 Training Enabled ---")
    else:
        print("\n--- FP32 Training Enabled ---")

    CausalLM = MODEL_MAP[args.model]
    model = CausalLM.from_pretrained(
        args.model_id,
        torch_dtype="float16" if args.use_fp16 else "float32",
    )

    model.config.use_cache = False

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    print("\n--- Model & MFU Setup ---")
    flops_per_step = 0
    if args.device_peak_flops > 0:
        total_params = sum(p.numel() for p in get_parameters(model))
        flops_per_step = estimate_mfu_flops(model, args.batch_size, args.max_length)
        print(f"  Total Parameters: {total_params / 1e6:.2f}M")
        print(f"  Sequence Length: {args.max_length}, Batch Size: {args.batch_size}")
        print(f"  Estimated FLOPS per step (arch-aware): {flops_per_step / 1e12:.2f} TFLOPS")
        print(f"  Provided Peak Hardware FLOPS: {args.device_peak_flops} TFLOPS")
    else:
        print("  Skipping MFU calculation (device_peak_flops not provided).")

    if args.use_lora:
        print("\n--- LoRA is enabled ---")
        for p in get_parameters(model): p.requires_grad = False
        apply_lora_to_model(model, args.lora_r, args.lora_alpha, args.lora_target_modules)
        params = get_lora_parameters(model)
        print(f"Optimizing {len(params)} LoRA tensors.")
    else:
        print("\n--- Full finetuning enabled ---")
        params = get_parameters(model)
        for p in params: p.requires_grad = True

    print(f"\nLoading dataset '{args.dataset_id}'...")
    dataset = load_dataset(args.dataset_id, split="train")
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))

    optim = AdamW(params, lr=args.learning_rate)

    @TinyJit
    def train_step(input_ids: Tensor, labels: Tensor):
        optim.zero_grad()
        loss = model(input_ids, labels=labels).loss
        loss.cast(dtypes.float32).backward() # Always use FP32 for backward pass stability
        
        # Gradient Clipping
        total_norm = (sum(p.grad.float().square().sum() for p in optim.params if p.grad is not None)).sqrt()
        for p in optim.params:
            if p.grad is not None:
                p.grad = p.grad * (args.gradient_clipping_norm / (total_norm + 1e-6)).clamp(max_=1.0)

        optim.step()
        loss_cpu, lr_cpu, norm_cpu = loss.detach().to("CPU"), optim.lr.to("CPU"), total_norm.detach().to("CPU")
        Tensor.realize(loss_cpu, lr_cpu, norm_cpu)
        return loss_cpu, lr_cpu.item(), norm_cpu

    print("\n--- Starting Training ---")
    train_iterator = iter(data_generator(dataset, tokenizer, args.max_length, args.batch_size))
    pbar = tqdm(range(args.max_steps), desc="Training")
    
    step_times, warmup_steps = [], 5 

    for step in pbar:
        with Tensor.train():
            try:
                input_ids, labels = next(train_iterator)
            except StopIteration:
                train_iterator = iter(data_generator(dataset, tokenizer, args.max_length, args.batch_size))
                input_ids, labels = next(train_iterator)

            start_time = time.perf_counter()
            loss, lr, grad_norm = train_step(input_ids, labels)
            end_time = time.perf_counter()
            
            step_time = end_time - start_time
            if step >= warmup_steps: step_times.append(step_time)

            mfu_str, mfu, achieved_tflops = "N/A", 0.0, 0.0
            if flops_per_step > 0 and step_times:
                avg_step_time = sum(step_times) / len(step_times)
                achieved_tflops = flops_per_step / avg_step_time / 1e12
                mfu = achieved_tflops / args.device_peak_flops
                mfu_str = f"{mfu:.2%}"
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr":  f"{lr:.1e}", "time": f"{step_time*1000:.2f}ms", "MFU": mfu_str})

            if args.use_wandb:
                wandb.log({"train/loss": loss.item(), "train/learning_rate": lr, "train/grad_norm": grad_norm.item(), "perf/step_time_ms": step_time * 1000, "perf/mfu": mfu, "perf/achieved_tflops": achieved_tflops}, step=step)

    print("\n--- Training Complete ---")
    if args.use_wandb: wandb.finish()

    if args.save_weights:
        print("\n--- Saving Weights ---")
        if args.hf_repo is not None:
            model.save_pretrained(f"./my-finetuned-{args.model}", repo_id=args.hf_repo)
            print(f"\nStored weights: ./my-finetuned-{args.model}")
        else:
            model.save_pretrained(f"./my-finetuned-{args.model}")
            print(f"\nStored weights: ./my-finetuned-{args.model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with LFM2 on tinygrad")
    # Model and Data
    parser.add_argument("--model", type=str, default="LFM2", choices=MODEL_MAP.keys(), help="Supported model choice.")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model repository ID")
    parser.add_argument("--dataset_id", type=str, default="mlabonne/FineTome-100k", help="Hugging Face dataset ID for SFT")
    # Training Hyperparameters
    parser.add_argument("--max_length", type=int, default=512, help="Fixed sequence length for training")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimizer learning rate")
    parser.add_argument("--gradient_clipping_norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples from dataset (for quick tests)")
    # LoRA
    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA fine-tuning")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_target_modules", nargs='+', default=["q_proj", "k_proj", "v_proj"], help="Module names to apply LoRA to")
    # Configuration Toggles
    parser.add_argument("--use_fp16", action="store_true", help="Enable FP16 training for lower memory usage")
    # Add quantization argument to prevent errors, but it will be caught in main()
    parser.add_argument("--quantize", type=str, default=None, help="Quantization mode (not supported for SFT).")
    # Performance and Logging
    parser.add_argument("--device_peak_flops", type=float, default=-1.0, help="Peak FP16/BF16 TFLOPS of device. -1 to disable MFU calculation.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="lfm2-tinygrad-sft", help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=f"sft-run-{int(time.time())}", help="Wandb run name")
    # Saving weights
    parser.add_argument("--save_weights", action="store_true", help="Enable Store Weights to Disk")
    parser.add_argument("--hf_repo", type=str, default=None, help="Huggingface Target Repository")
    
    args = parser.parse_args()
    main(args)