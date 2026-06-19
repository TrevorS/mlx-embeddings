"""EuroBERT encoder for embeddings (e.g. jina-embeddings-v5-text-nano).

EuroBERT is a *bidirectional* encoder built on a Llama-style backbone:
RoPE + SwiGLU MLP + RMSNorm, pre-norm blocks, no attention bias and — unlike
Qwen3 — no per-head q_norm/k_norm. The embedding head uses last-token pooling
followed by L2 normalization, matching Jina's v5-text-nano checkpoint.

The only structural difference from a decoder backbone is the attention mask:
the encoder applies a padding mask but *no causal mask*, so every token may
attend to every other token. Weight layout matches the checkpoint exactly
(``model.embed_tokens``, ``model.layers.N.*``, ``model.norm``), so no key
remapping is required.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings


def last_token_pool(
    last_hidden_states: mx.array, attention_mask: Optional[mx.array] = None
) -> mx.array:
    """Pool the hidden state of the last non-padding token of each sequence.

    Handles both left- and right-padded batches; with no mask it takes the
    final position.
    """
    if attention_mask is None:
        return last_hidden_states[:, -1]

    # Left padding -> every sequence ends with a real token at position -1.
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = (attention_mask.sum(axis=1) - 1).astype(mx.int32)
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[mx.arange(batch_size), sequence_lengths]


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "eurobert"
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 8192
    vocab_size: int = 128256

    rms_norm_eps: float = 1e-5
    rope_theta: float = 1000000.0

    attention_bias: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False

    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    architectures: List[str] = field(default_factory=lambda: ["EuroBertModel"])

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            if self.hidden_size % self.num_attention_heads != 0:
                raise ValueError(
                    f"hidden_size ({self.hidden_size}) must be divisible by "
                    f"num_attention_heads ({self.num_attention_heads})"
                )
            self.head_dim = self.hidden_size // self.num_attention_heads


class EuroBertMLP(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        dim, hidden = config.hidden_size, config.intermediate_size
        bias = config.attention_bias
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class EuroBertAttention(nn.Module):
    """Multi-head attention with RoPE and optional GQA; no q_norm/k_norm.

    Bidirectional — the caller supplies a padding-only mask (or None).
    """

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5

        dim = config.hidden_size
        bias = config.attention_bias
        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=bias)

        self.rotary_emb = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rotary_emb(queries)
        keys = self.rotary_emb(keys)

        if self.num_key_value_groups > 1:
            keys = mx.repeat(keys, self.num_key_value_groups, axis=1)
            values = mx.repeat(values, self.num_key_value_groups, axis=1)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class EuroBertLayer(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.self_attn = EuroBertAttention(config)
        self.mlp = EuroBertMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class EuroBertModel(nn.Module):
    """Bidirectional EuroBERT encoder stack (no causal mask)."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [EuroBertLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> mx.array:
        h = self.embed_tokens(input_ids)

        mask = None
        if attention_mask is not None:
            # (B, L) padding mask -> additive (B, 1, 1, L); bidirectional, no causal term.
            padding = mx.where(attention_mask == 0, -mx.inf, 0.0)
            mask = padding[:, None, None, :].astype(h.dtype)

        for layer in self.layers:
            h = layer(h, mask)

        return self.norm(h)


class Model(nn.Module):
    """EuroBERT embedding model: encoder + last-token pooling + L2 normalize."""

    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = EuroBertModel(config)

    def __call__(
        self, input_ids: mx.array, attention_mask: Optional[mx.array] = None
    ) -> BaseModelOutput:
        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be 2D, got shape {input_ids.shape}")

        last_hidden_state = self.model(input_ids, attention_mask=attention_mask)
        pooled = last_token_pool(last_hidden_state, attention_mask)
        text_embeds = normalize_embeddings(pooled)

        return BaseModelOutput(
            text_embeds=text_embeds, last_hidden_state=last_hidden_state
        )

    def sanitize(self, weights: dict) -> dict:
        # Weight keys already match (model.*); drop any tied lm_head and rotary buffers.
        return {
            k: v
            for k, v in weights.items()
            if not k.endswith("lm_head.weight") and "rotary_emb.inv_freq" not in k
        }
