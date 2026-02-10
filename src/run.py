# src/run.py

import argparse
from model import MODEL_MAP

# tinygrad imports
from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv
from tinygrad import Device

print("DEVICE:", Device.DEFAULT)

# For reproducible tests
if getenv("SEED"):
    Tensor.manual_seed(getenv("SEED"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LFM2 inference in tinygrad.")
    parser.add_argument("--use_fp16", action="store_true", help="Enable FP16 training for lower memory usage")
    parser.add_argument("--quantize", type=str, default=None, choices=["nf4", "int8"], help="Enable NF4 or INT8 quantization for the model.")
    parser.add_argument("--model", type=str, default="LFM2",  choices=MODEL_MAP.keys(), help="Supported model choice.")
    parser.add_argument("--model_id", type=str, default="LiquidAI/LFM2-350M", help="Hugging Face model repository ID")
    args = parser.parse_args()
    
    # NF4 uses float16 for scales, so set the base model dtype to float16 to avoid excessive casting.
    dtype = dtypes.float16 if args.quantize == "nf4" else dtypes.float32
    
    CausalLM = MODEL_MAP[args.model]
    model = CausalLM.from_pretrained(
        args.model_id,
        quantize=args.quantize,
        torch_dtype="float16" if args.use_fp16 or args.quantize else "float32",
    )

    tokenizer = model.tokenizer

    prompt = "The secret to a long and happy life is"
    input_ids_list = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors=None,
        tokenize=True,
        return_dict=False,
    )
    input_ids = Tensor([input_ids_list], dtype=dtypes.int32)

    print("\n--- Starting Text Generation ---")
    print(tokenizer.decode(input_ids_list), end="", flush=True)

    output_ids = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.3,
        # min_p=0.15, # Not yet supported in tinygrad
        repetition_penalty=1.05,
        max_new_tokens=50,
    )

    # The generate function handles the streaming print, so we just add a final newline.
    # The returned output is the full sequence.
    print("\n\n--- Generation Complete ---")
    full_output_str = tokenizer.decode(output_ids[0].numpy().tolist())
    # print("\nFull output:\n", full_output_str) # Uncomment to see the full decoded string