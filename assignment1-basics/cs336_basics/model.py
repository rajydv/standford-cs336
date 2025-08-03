import torch
from torch import nn
from torch import Tensor
from einops import einsum, rearrange
import einx
from jaxtyping import Float, Int
import math
from cs336_basics.nn_utils import(
    softmax
)
import logging
from jaxtyping import Float, Bool, Int

from collections.abc import Callable, Iterable
from typing import Optional

logger = logging.getLogger(__name__)

def get_std(
    in_feature: int,
    out_feature: int
):
    return torch.sqrt(torch.tensor(2)/(in_feature + out_feature))

def silu(
    x: torch.tensor
) -> torch.tensor:
    return einsum(x, torch.sigmoid(x), "..., ... -> ...")


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,      
):
    sqrt_dk = torch.sqrt(torch.tensor(K.shape[-1]))
    raw_attention = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")/sqrt_dk

    
    if mask is not None:
        raw_attention = torch.where(mask == 0, torch.tensor(float('-inf')), raw_attention)

    attention_weights = softmax(raw_attention, dim=-1)
    return einsum(attention_weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device: torch.device=None,
            dtype: torch.device=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        std = get_std(in_features, out_features)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(out_features, in_features, **factory_kwargs),
                mean = 0,
                std=std,
                a=-3*std,
                b=3*std
            )
        )

    
    def forward(self, x: torch.tensor) -> torch.tensor:
        return einsum(x, self.weight,  "... in_feature, out_feature in_feature -> ... out_feature")
    
    

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device=None,
        dtype: torch.device=None
    ) -> None:
        super().__init__()
        self.device = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        std = get_std(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim, **factory_kwargs),
                mean = 0,
                std=std,
                a=-3*std,
                b=3*std
            ),
        )
    
    def forward(self, token_ids: torch.tensor) -> torch.tensor:
        return self.weight[token_ids]



class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device  = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty(self.d_model, **factory_kwargs))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_x =  torch.sqrt(einsum(x**2, "... d_model -> ...") / self.d_model + self.eps)
        scale_x = einsum(x, 1/rms_x, "... d_model, ... -> ... d_model")
        rmsnorm_x = einsum(scale_x, self.weight, "... d_model, d_model -> ... d_model")
        return rmsnorm_x.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device  = device
        self.dtype = dtype
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w3 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, d_model)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # FNN(x) = (SiLU(xW1)*xW3)W2
        ff1 = einsum(silu(self.w1(x)), self.w3(x), "... d_ff, ... d_ff -> ... d_ff")
        return self.w2(ff1)


class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache",
            RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )
    
    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[torch.Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = theta ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[torch.Tensor, " ... seq d"], pos_ids: Int[torch.Tensor, " ... seq"]) -> Float[torch.Tensor, " ... seq d"]:
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result
    
    def extra_repr(self):
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}"


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

            
        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        Q, K, V = (
            rearrange(item, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for item in (Q, K, V)
        )  # fmt: skip


        
        if self.positional_encoder:
            if token_positions is None:
                token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))

            # Duplicate token positions for each head
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        # Construct causal mask
        seq = torch.arange(sequence_length, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask = qi >= kj  # (query, key)
        # Suppose seq len is 4
        # [
        #     [True, False, False, False],
        #     [True, True, False, False],
        #     [True, True, True, False],
        #     [True, True, True, True]
        # ]

        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()
        output = self.output_proj(attn_output)
        return output

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding | None
    ) -> None:
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            positional_encoder
        )

        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, " batch sequence_length d_model"]
    ) -> Float[Tensor, " batch sequence_length d_model"]:
        # sub layer 1
        x_attn = self.attn(self.ln1(x))
        attn_sublayer = x_attn + x

        # sub layer 2
        x_ffn = self.ffn(self.ln2(attn_sublayer))
        ffn_sublayer = attn_sublayer + x_ffn
        return ffn_sublayer


class BasicTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        rope_theta: float,
        num_layers: int
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.positional_encoder = RotaryEmbedding(
            context_length,
            d_model // num_heads,
            rope_theta
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    d_ff,
                    num_heads,
                    self.positional_encoder
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size) # output embedding


    def forward(self, in_indices: Int[Tensor, "... sequence_length"]) -> Float[Tensor, "... sequence_length vocab_size"]:
        # 1. First take each token in  sequence_length and token Embedding
        # 2. Pass the token Embedding into TransformerBlock layers
        # 3. Do final norm
        # 4. Calculate output embedding of each token in vocab_size

        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)


        x_norm = self.ln_final(x) # (batch, sequence_length, d_model)
        return self.lm_head(x_norm)


