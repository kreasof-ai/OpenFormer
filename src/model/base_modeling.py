# src/model/base_modeling.py

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

# Third-party imports
from huggingface_hub import HfApi, create_repo, hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
import torch  # used for converting weights and saving
from transformers import AutoTokenizer
from huggingface_hub.utils import EntryNotFoundError

# tinygrad imports
from tinygrad import Tensor, dtypes, Device, TinyJit, UOp
from tinygrad.nn import Embedding, Linear, RMSNorm
from tinygrad.nn.state import load_state_dict, get_parameters, get_state_dict

# Project imports for shared components
from extra.lora import LoRALinear
from extra.quantization import NF4Linear, Int8Linear
from utils.rope import _precompute_rope_cache, apply_rotary_pos_emb
from utils.output import CausalLMOutputWithPast


# --- Base Configuration ---
@dataclass
class BaseConfig(ABC):
    vocab_size: int = 65536
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    head_dim: int = 128
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 4096
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    max_context: int = 4096
    use_cache: bool = True # False for training

    # --- tinygrad specific flags ---
    dtype: Any = dtypes.float32
    quantize: Optional[str] = None

    @classmethod
    @abstractmethod
    def from_hf_config(cls, config_dict: dict) -> "BaseConfig":
        """Creates a config instance from a Hugging Face config dictionary."""
        raise NotImplementedError


# --- Base Model Components ---

class BaseAttention:
    """
    A standardized attention module supporting GQA, QK Norm, and RoPE
    """
    def __init__(self, config: BaseConfig, linear_class: Type = Linear):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = linear_class(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = linear_class(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = linear_class(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # Support for models with QK Norm
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if getattr(config, "qk_norm") else lambda x: x
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if getattr(config, "qk_norm") else lambda x: x

        self._forward_decoding_jit = TinyJit(self._forward_decoding)

    def _forward_decoding(self, hidden_states: Tensor, start_pos:UOp, cos_sin: Tuple[Tensor, Tensor]):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        query_states = self.q_norm(query_states).permute(0, 2, 1, 3)
        key_states = self.k_norm(key_states).permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)

        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        self.cache_kv[:, :, :, start_pos:start_pos+q_len, :].assign(Tensor.stack(key_states, value_states)).realize()
        key_states = self.cache_kv[0, :, :, 0:start_pos+q_len, :]
        value_states = self.cache_kv[1, :, :, 0:start_pos+q_len, :]

        attn_output = Tensor.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=None, enable_gqa=True)
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)

    def _forward_prefill(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Any], cos_sin: Tuple[Tensor, Tensor]):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).reshape(bsz, q_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(bsz, q_len, self.num_key_value_heads, self.head_dim)
        
        query_states = self.q_norm(query_states).permute(0, 2, 1, 3)
        key_states = self.k_norm(key_states).permute(0, 2, 1, 3)
        value_states = value_states.permute(0, 2, 1, 3)

        cos, sin = cos_sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.config.use_cache:
            self.cache_kv = Tensor.zeros(2, bsz, self.num_key_value_heads, self.config.max_context, self.head_dim, dtype=hidden_states.dtype, device=hidden_states.device).contiguous().realize()
            self.cache_kv[:, :, :, 0:q_len, :].assign(Tensor.stack(key_states, value_states)).realize()

        all_key_states, all_value_states = key_states, value_states
        present_kv = (key_states, value_states)

        all_key_states = all_key_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
        all_value_states = all_value_states.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)

        attn_output = Tensor.scaled_dot_product_attention(query_states, all_key_states, all_value_states, attn_mask=attention_mask)
        
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present_kv

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_kv: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos:int|UOp):
        _, q_len, _ = hidden_states.shape

        if q_len > 1:
            output, new_state = self._forward_prefill(hidden_states, attention_mask, past_kv, cos_sin)
        else: # Generation stage (q_len == 1)
            output = self._forward_decoding_jit(hidden_states.contiguous(), start_pos, (cos_sin[0].contiguous(), cos_sin[1].contiguous()))
            new_state = None

        return output, new_state

class BaseMLP:
    """ Standardized SwiGLU MLP. """
    def __init__(self, config: BaseConfig, linear_class: Type = Linear):
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = linear_class(config.hidden_size, config.intermediate_size, bias=mlp_bias)
        self.up_proj = linear_class(config.hidden_size, config.intermediate_size, bias=mlp_bias)
        self.down_proj = linear_class(config.intermediate_size, config.hidden_size, bias=mlp_bias)
    def __call__(self, x: Tensor) -> Tensor:
        return self.down_proj(self.gate_proj(x).silu() * self.up_proj(x))


class BaseModel(ABC):
    """ The stack of decoder layers. """
    def __init__(self, config: BaseConfig, linear_class: Type):
        self.config = config
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = [self._create_decoder_layer(config, linear_class, i) for i in range(config.num_hidden_layers)]
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.head_dim = config.head_dim
        cos_cache, sin_cache = _precompute_rope_cache(dim=self.head_dim, max_seq_len=config.max_position_embeddings, base=config.rope_theta, dtype=config.dtype)
        self.cos_cache = cos_cache
        self.sin_cache = sin_cache

    @abstractmethod
    def _create_decoder_layer(self, config: BaseConfig, linear_class: Type):
        raise NotImplementedError

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int|UOp, output_hidden_states: bool, **kwargs):
        h = self.embed_tokens(input_ids)
        bsz, seq_len, _ = h.shape
        all_hidden_states = (h,) if output_hidden_states else None
        mask = Tensor.full((1, 1, seq_len, seq_len), -float("inf")).triu(1).realize() if seq_len > 1 else None
        
        cos = self.cos_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.head_dim).expand(bsz, 1, seq_len, self.head_dim)
        sin = self.sin_cache[start_pos : start_pos + seq_len].reshape(1, 1, seq_len, self.head_dim).expand(bsz, 1, seq_len, self.head_dim)
        
        new_states_list = []
        for i, layer in enumerate(self.layers):
            past_st = past_states[i] if past_states else None
            h, new_st = layer(h, mask, past_st, (cos, sin), start_pos, **kwargs)
            new_states_list.append(new_st)
            if i + 1 == len(self.layers): h = self.norm(h)
            if output_hidden_states: all_hidden_states += (h,)

        if output_hidden_states: all_hidden_states += (h,)
        return h, new_states_list, all_hidden_states


class BaseForCausalLM(ABC):
    """ Top-level model class with generation and loading capabilities. """
    def __init__(self, config: BaseConfig):
        self.config = config
        self.tokenizer = None
        self.source_model_id = None
        
        if config.quantize == "nf4": self.linear_class = NF4Linear()
        elif config.quantize == "int8": self.linear_class = Int8Linear()
        else: self.linear_class = Linear

        self.model = self._create_model(config, self.linear_class)
        self.lm_head = Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings: self.lm_head.weight = self.model.embed_tokens.weight

    @abstractmethod
    def _create_model(self, config: BaseConfig, linear_class: Type) -> BaseModel:
        raise NotImplementedError

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]] = None, start_pos: int = 0, labels: Optional[Tensor] = None, output_hidden_states: bool = False, **kwargs) -> CausalLMOutputWithPast:
        _past_states = past_states

        hidden_states, new_states, all_hidden_states = self.model(input_ids, _past_states, start_pos, output_hidden_states, **kwargs)
        
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = logits[..., :-1, :].flatten(0, 1).sparse_categorical_crossentropy(labels[..., 1:].flatten(), ignore_index=-100)

        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=new_states,
            hidden_states=all_hidden_states
        )
    
    def _sample(self, logits: Tensor, do_sample: bool, temperature: float, min_p: float, repetition_penalty: float, generated_tokens: List[int]) -> int:
        if repetition_penalty != 1.0 and generated_tokens:
            unique_tokens = Tensor(list(set(generated_tokens)), dtype=dtypes.int32)
            updates = Tensor.where(logits[unique_tokens] > 0, logits[unique_tokens] / repetition_penalty, logits[unique_tokens] * repetition_penalty)
            logits = logits.scatter(0, unique_tokens, updates)
        if not do_sample or temperature == 0: return logits.argmax().item()
        if min_p > 0.0: raise NotImplementedError("Top-p (min_p) sampling is not yet supported in tinygrad.")
        probs = (logits / temperature).softmax()
        return (probs.cumsum() > Tensor.uniform(1).item()).argmax().item()
    
    def _decode_one_token(self, next_token_id: int):
        print(self.tokenizer.decode([next_token_id]), end="", flush=True)

    def generate(self, input_ids: Tensor, max_new_tokens: int, do_sample: bool = False, temperature: float = 1.0, min_p: float = 0.0, repetition_penalty: float = 1.0) -> Tensor:
        assert self.tokenizer is not None, "Tokenizer must be attached to the model."
        Tensor.training = False
        tokens = input_ids[0].numpy().tolist()
        
        v_start_pos = UOp.variable("start_pos", 1, self.config.max_context-1)
        past_states = [None] * len(self.model.layers)
        outputs = self(Tensor([tokens]), past_states, start_pos=0)
        start_pos = len(tokens)
        for _ in range(max_new_tokens):
            past_states = outputs.past_key_values
            logits = outputs.logits[0, -1, :]
            next_token = self._sample(logits, do_sample, temperature, min_p, repetition_penalty, tokens)
            tokens.append(next_token)
            if next_token == self.tokenizer.eos_token_id: break
            self._decode_one_token(next_token)
            outputs = self(Tensor([[next_token]]), past_states, start_pos=v_start_pos.bind(start_pos))
            start_pos += 1
        
        return Tensor([tokens], dtype=dtypes.int32)
    
    @classmethod
    @abstractmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        raise NotImplementedError

    @abstractmethod
    def _get_key_map(self) -> dict:
        raise NotImplementedError
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        print(f"--- Loading Model: {model_id} ---")
        config = cls._from_hf_config(model_id)

        torch_dtype_map = {"bfloat16": dtypes.bfloat16, "float16": dtypes.float16, "float32": dtypes.float32}
        if "torch_dtype" in kwargs: config.dtype = torch_dtype_map.get(str(kwargs["torch_dtype"]).split('.')[-1], dtypes.float32)
        if "quantize" in kwargs: config.quantize = kwargs["quantize"]

        print("\nInitializing model architecture...")
        model = cls(config)
        model.source_model_id = model_id # Store the original repo ID
        
        cls._load_from_hf(model, model_id)

        print("\nLoading tokenizer...")
        model.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        for k in ["device_map", "attn_implementation"]:
            if k in kwargs: print(f"  [Warning] tinygrad: Argument '{k}' is not used.")
        return model

    @classmethod
    def _load_from_hf(cls, model, repo_id: str, filename: str = "model.safetensors"):
        print(f"Fetching weights from {repo_id}...")
        dtype = model.config.dtype
        key_map = model._get_key_map()
        
        tg_state_dict = {}
        print("Loading weights into memory...")

        try:
            # 1. Try to load sharded model index
            index_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors.index.json")
            with open(index_path) as f:
                index_data = json.load(f)
            weight_map = index_data["weight_map"]

            # 2. Group keys by shard
            shards = {}
            for hf_key, shard_filename in weight_map.items():
                if shard_filename not in shards:
                    shards[shard_filename] = []
                shards[shard_filename].append(hf_key)

            # 3. Load from each shard
            print(f"Model is sharded. Found {len(shards)} files.")
            for shard_filename in sorted(shards.keys()):
                hf_keys_in_shard = shards[shard_filename]
                print(f"  Loading shard: {shard_filename}")
                local_shard_path = hf_hub_download(repo_id=repo_id, filename=shard_filename)
                with safe_open(local_shard_path, framework="pt", device="cpu") as f:
                    for hf_key in hf_keys_in_shard:
                        # Find the corresponding tinygrad key
                        if hf_key in key_map:
                            tg_key = key_map[hf_key]
                            tg_state_dict[tg_key] = Tensor(f.get_tensor(hf_key).to(torch.float32).numpy(), requires_grad=False)

        except (EntryNotFoundError, FileNotFoundError):
            # 4. Fallback for non-sharded models
            filename = "model.safetensors"
            print(f"No model.safetensors.index.json found, falling back to single '{filename}' file.")
            local_path = hf_hub_download(repo_id=repo_id, filename=filename)
            with safe_open(local_path, framework="pt", device="cpu") as f:
                for hf_key, tg_key in key_map.items():
                    if hf_key in f.keys():
                        tg_state_dict[tg_key] = Tensor(f.get_tensor(hf_key).to(torch.float32).numpy(), requires_grad=False)

        if model.config.quantize in ["nf4", "int8"]:
            device = getattr(model.model.embed_tokens.weight, 'device', Device.DEFAULT)
            tg_state_dict = model.linear_class.quantize(tg_state_dict, device=device)

        for k in tg_state_dict:
            if tg_state_dict[k].dtype != dtypes.uint8:
                tg_state_dict[k] = tg_state_dict[k].cast(dtype)
        
        print("Assigning weights to model...")
        load_state_dict(model, tg_state_dict, strict=False)
        if model.config.tie_word_embeddings:
            print("Re-tying word embeddings for lm_head...")
            model.lm_head.weight = model.model.embed_tokens.weight
        print("All weights loaded and assigned.")

    def save_pretrained(self, save_directory: str, repo_id: Optional[str] = None):
        """
        Saves the model weights to a local directory and optionally uploads to the Hugging Face Hub.
        LoRA weights are automatically fused into the base model weights.
        Non-weight files (config, tokenizer, etc.) are copied from the original source repository.

        Args:
            save_directory (str): The directory to save the model artifacts.
            repo_id (Optional[str]): The Hugging Face Hub repository ID to upload to (e.g., "your-username/your-model-name").
                                     If provided, the repository will be created if it doesn't exist.
        """
        assert self.source_model_id is not None, "Cannot save model without a source_model_id. Please load with from_pretrained."
        print(f"--- Saving model to {save_directory} ---")
        
        Path(save_directory).mkdir(parents=True, exist_ok=True)

        hf_state_dict = {}
        inverse_key_map = {v: k for k, v in self._get_key_map().items()}
        processed_tg_keys = set()
        
        def get_named_modules(module, prefix=""):
            modules = {prefix: module}
            if not hasattr(module, '__dict__'): return modules
            for name, sub_module in module.__dict__.items():
                if isinstance(sub_module, list):
                    for i, m in enumerate(sub_module):
                        if hasattr(m, '__dict__'):
                            modules.update(get_named_modules(m, prefix=f"{prefix}.{name}.{i}" if prefix else f"{name}.{i}"))
                elif hasattr(sub_module, '__dict__'):
                     modules.update(get_named_modules(sub_module, prefix=f"{prefix}.{name}" if prefix else name))
            return modules
        
        all_modules = get_named_modules(self)

        print("Processing model weights for saving...")
        for path, module in all_modules.items():
            if isinstance(module, LoRALinear):
                print(f"  Fusing LoRA weights for: {path}")
                merged_weight = module.merge_weights()
                
                tg_key = f"{path}.weight" # The key of the original weight tensor
                if tg_key in inverse_key_map:
                    hf_key = inverse_key_map[tg_key]
                    hf_state_dict[hf_key] = torch.from_numpy(merged_weight.numpy())
                    
                    # Mark all related tensors as processed to avoid saving them separately
                    processed_tg_keys.add(tg_key)
                    processed_tg_keys.add(f"{path}.lora_A")
                    processed_tg_keys.add(f"{path}.lora_B")
                    if hasattr(module.linear, 'scale'):
                        processed_tg_keys.add(f"{path}.scale")
            
            elif isinstance(module, (self.linear_class,)): # Catch non-LoRA quantized layers
                if hasattr(module, 'dequantize'):
                    print(f"  Dequantizing layer: {path}")
                    dequant_weight = module.dequantize().realize()
                    
                    tg_key = f"{path}.weight"
                    if tg_key in inverse_key_map:
                        hf_key = inverse_key_map[tg_key]
                        hf_state_dict[hf_key] = torch.from_numpy(dequant_weight.numpy())
                        processed_tg_keys.add(tg_key)
                        processed_tg_keys.add(f"{path}.scale")

        # Handle all other tensors (embeddings, norms, non-quantized linear, etc.)
        full_state_dict = get_state_dict(self)
        for tg_key, tensor in full_state_dict.items():
            if tg_key in processed_tg_keys: continue
            if tg_key in inverse_key_map:
                hf_key = inverse_key_map[tg_key]
                hf_state_dict[hf_key] = torch.from_numpy(tensor.numpy())
        
        weights_path = os.path.join(save_directory, "model.safetensors")
        safe_save_file(hf_state_dict, weights_path, metadata={"format": "pt"})
        print(f"  Weights saved to {weights_path}")

        print(f"  Copying non-weight files from {self.source_model_id}...")
        snapshot_download(
            repo_id=self.source_model_id,
            local_dir=save_directory,
            allow_patterns=["*"],
            ignore_patterns=["*.safetensors", "*.bin*", "*.pth"],
            local_dir_use_symlinks=False,
        )
        print("  Copied configuration and tokenizer files.")

        if repo_id:
            print(f"--- Uploading model to Hugging Face Hub: {repo_id} ---")
            print("  (Ensure you have run 'huggingface-cli login' first)")
            
            # Create the repository if it doesn't exist
            create_repo(repo_id, repo_type="model", exist_ok=True)
            print(f"  Repository '{repo_id}' ensured to exist.")
            
            api = HfApi()
            api.upload_folder(
                folder_path=save_directory,
                repo_id=repo_id,
                repo_type="model",
            )
            print("  Upload complete!")