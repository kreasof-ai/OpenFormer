# src/model/lfm2_modeling.py

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type

# Third-party imports
from huggingface_hub import hf_hub_download

# tinygrad imports
from tinygrad import Tensor, dtypes, TinyJit
from tinygrad.nn import Conv1d, Linear, RMSNorm, Embedding

# Project imports
from model.base_modeling import BaseConfig, BaseAttention, BaseMLP, BaseForCausalLM
from utils.rope import _precompute_rope_cache

# --- Configuration ---
@dataclass
class LFM2Config(BaseConfig):
    conv_kernel_size: int = 3
    full_attn_idxs: List[int] = field(default_factory=lambda: [2, 5, 8, 10, 12, 14])
    initializer_range: float = 0.02
    qk_norm: bool = True # Custom flag to indicate QK norm is used

    @classmethod
    def from_hf_config(cls, config_dict: dict) -> "LFM2Config":
        intermediate_size = config_dict.get("block_ff_dim", config_dict.get("intermediate_size"))
        if config_dict.get("block_auto_adjust_ff_dim", False):
            intermediate_size = int(2 * intermediate_size / 3)
            multiple_of = config_dict.get("block_multiple_of", 256)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        
        layer_types = config_dict.get("layer_types")
        full_attn_idxs = [i for i, t in enumerate(layer_types) if t == "full_attention"] if layer_types else config_dict["full_attn_idxs"]
        head_dim = config_dict["hidden_size"] // config_dict.get("num_attention_heads", config_dict.get("num_heads"))

        return cls(
            vocab_size=config_dict["vocab_size"],
            hidden_size=config_dict["hidden_size"],
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_hidden_layers=config_dict["num_hidden_layers"],
            num_attention_heads=config_dict.get("num_attention_heads", config_dict.get("num_heads")),
            num_key_value_heads=config_dict["num_key_value_heads"],
            full_attn_idxs=full_attn_idxs,
            conv_kernel_size=config_dict.get("conv_L_cache", 3),
            max_position_embeddings=config_dict["max_position_embeddings"],
            rms_norm_eps=config_dict.get("norm_eps", config_dict.get("block_norm_eps")),
            rope_theta=config_dict["rope_theta"],
        )

# --- Model Components (LFM2 Specific) ---
class LFM2ConvOperator:
    def __init__(self, config: LFM2Config, linear_class: Type = Linear):
        self.hidden_size = config.hidden_size
        self.kernel_size = config.conv_kernel_size
        self.in_proj = linear_class(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.conv = Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.kernel_size - 1, # Causal padding
            groups=config.hidden_size,
            bias=False
        )
        self.out_proj = linear_class(config.hidden_size, config.hidden_size, bias=False)

        self._forward_decoding_jit = TinyJit(self._forward_decoding)

    def _forward_decoding(self, x: Tensor, past_conv_state: Optional[Tensor]):
        """
        Decoding only expect fixed shape so we can do JIT
        """
        bsz, seq_len, _ = x.shape
        B, C, x_proj = self.in_proj(x).chunk(3, dim=-1)
        x_gated = B * x_proj

        x_gated_permuted = x_gated.permute(0, 2, 1)

        assert past_conv_state is not None
        assert past_conv_state.shape[2] == self.kernel_size
        new_conv_state = Tensor.cat(past_conv_state[:, :, 1:], x_gated_permuted, dim=2)
        conv_weights = self.conv.weight.reshape(1, self.hidden_size, self.kernel_size)
        conv_out = (new_conv_state * conv_weights).sum(axis=2, keepdim=True)

        conv_out = conv_out.permute(0, 2, 1)
        output = self.out_proj(C * conv_out)
        return output, new_conv_state
    
    def _forward_prefill(self, x: Tensor, past_conv_state: Optional[Tensor]):
        """
        Prefill expect variable length, so no JIT
        """
        bsz, seq_len, _ = x.shape
        B, C, x_proj = self.in_proj(x).chunk(3, dim=-1)
        x_gated = B * x_proj

        x_gated_permuted = x_gated.permute(0, 2, 1)

        conv_out = self.conv(x_gated_permuted)[:, :, :seq_len]
        if seq_len < self.kernel_size:
            pad_amount = self.kernel_size - seq_len
            padding = Tensor.zeros(bsz, self.hidden_size, pad_amount, device=x.device, dtype=x.dtype)
            new_conv_state = Tensor.cat(padding, x_gated_permuted, dim=2)
        else:
            new_conv_state = x_gated_permuted[:, :, -self.kernel_size:]

        conv_out = conv_out.permute(0, 2, 1)
        output = self.out_proj(C * conv_out)
        return output, new_conv_state
        
    def __call__(self, x: Tensor, past_conv_state: Optional[Tensor]):
        _, seq_len, _ = x.shape

        if seq_len > 1:
            output, new_conv_state = self._forward_prefill(x, past_conv_state)
        else: # Generation stage (seq_len == 1)
            output, new_conv_state = self._forward_decoding_jit(x.contiguous(), past_conv_state.contiguous())

        return output, new_conv_state

class LFM2MLP(BaseMLP): # Renaming SwiGLU to match base class
    def __init__(self, config: LFM2Config, linear_class: Type = Linear):
        super().__init__(config, linear_class)
        # Alias for weight loading compatibility
        self.w1, self.w3, self.w2 = self.gate_proj, self.up_proj, self.down_proj

class LFM2DecoderLayer:
    def __init__(self, config: LFM2Config, is_attention_block: bool, linear_class: Type):
        self.is_attention_block = is_attention_block
        self.operator = BaseAttention(config, linear_class) if is_attention_block else LFM2ConvOperator(config, linear_class)
        self.feed_forward = LFM2MLP(config, linear_class)
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self._ffn_jit = TinyJit(self._ffn)

    def _ffn(self, hidden_states: Tensor):
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states

    def __call__(self, hidden_states: Tensor, attention_mask: Optional[Tensor], past_state: Optional[Any], cos_sin: Tuple[Tensor, Tensor], start_pos: int, **kwargs):
        _, seq_len, _ = hidden_states.shape
        residual = hidden_states
        normed_hidden = self.operator_norm(hidden_states)
        if self.is_attention_block:
            op_out, new_state = self.operator(normed_hidden, attention_mask, past_state, cos_sin, start_pos, **kwargs)
        else:
            op_out, new_state = self.operator(normed_hidden, past_state)
        
        hidden_states = op_out + residual

        if seq_len > 1:
            hidden_states = self._ffn(hidden_states)
        else:
            hidden_states = self._ffn_jit(hidden_states)
        
        return hidden_states, new_state

class LFM2Model: # Not inheriting from BaseModel because layer structure is too different
    def __init__(self, config: LFM2Config, linear_class: Type):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [LFM2DecoderLayer(config, i in config.full_attn_idxs, linear_class) for i in range(config.num_hidden_layers)]
        
        self.head_dim = config.hidden_size // config.num_attention_heads
        cos_cache, sin_cache = _precompute_rope_cache(dim=self.head_dim, max_seq_len=config.max_position_embeddings, base=config.rope_theta, dtype=config.dtype)
        self.cos_cache, self.sin_cache = cos_cache, sin_cache

    def __call__(self, input_ids: Tensor, past_states: Optional[List[Any]], start_pos: int, output_hidden_states: bool, **kwargs):
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
        return h, new_states_list, all_hidden_states

class LFM2ForCausalLM(BaseForCausalLM):
    def _create_model(self, config: BaseConfig, linear_class: Type) -> Any:
        return LFM2Model(config, linear_class)

    def __call__(self, *args, **kwargs):
        # Override to handle LFM2's special cache update for conv layers
        output = super().__call__(*args, **kwargs)
        return output
    
    @classmethod
    def _from_hf_config(cls, model_id: str) -> BaseConfig:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f: config_dict = json.load(f)
        return LFM2Config.from_hf_config(config_dict)

    def _get_key_map(self) -> dict:
        key_map = {
            "model.embedding_norm.weight": "model.norm.weight", # HF calls it embedding_norm
            "model.embed_tokens.weight": "model.embed_tokens.weight"
        }
        for i, layer in enumerate(self.model.layers):
            p, op_p = f"model.layers.{i}", f"model.layers.{i}.operator"
            key_map.update({
                f"{p}.operator_norm.weight": f"{p}.operator_norm.weight",
                f"{p}.ffn_norm.weight": f"{p}.ffn_norm.weight",
                f"{p}.feed_forward.w1.weight": f"{p}.feed_forward.gate_proj.weight",
                f"{p}.feed_forward.w2.weight": f"{p}.feed_forward.down_proj.weight",
                f"{p}.feed_forward.w3.weight": f"{p}.feed_forward.up_proj.weight",
            })
            if layer.is_attention_block:
                key_map.update({
                    f"{p}.self_attn.q_proj.weight": f"{op_p}.q_proj.weight", f"{p}.self_attn.k_proj.weight": f"{op_p}.k_proj.weight",
                    f"{p}.self_attn.v_proj.weight": f"{op_p}.v_proj.weight", f"{p}.self_attn.out_proj.weight": f"{op_p}.o_proj.weight",
                    f"{p}.self_attn.q_layernorm.weight": f"{op_p}.q_norm.weight", f"{p}.self_attn.k_layernorm.weight": f"{op_p}.k_norm.weight"
                })
            else:
                key_map.update({
                    f"{p}.conv.in_proj.weight": f"{op_p}.in_proj.weight", f"{p}.conv.conv.weight": f"{op_p}.conv.weight",
                    f"{p}.conv.out_proj.weight": f"{op_p}.out_proj.weight"
                })
        return key_map