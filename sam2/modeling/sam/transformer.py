# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from:
# SAM2Export repository by Aimol-l
# Original file: https://github.com/Aimol-l/SAM2Export/blob/main/sam2/modeling/sam/transformer.py
# For more info, check issue: https://github.com/facebookresearch/sam2/issues/284

import contextlib
import math
import warnings
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# Make sure these imports exist in your repo:
from sam2.modeling.position_encoding import (
    apply_rotary_enc,
    apply_rotary_matenc,
    compute_axial_cis,
    get_rotation_matrices,
)
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import get_sdpa_settings

warnings.simplefilter(action="ignore", category=FutureWarning)

# --- FLASH ATTENTION FALLBACK LOGIC ---
OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
ALLOW_ALL_KERNELS = False

def sdp_kernel_context(dropout_p: float):
    """
    Decide whether we use FlashAttention / Mem-Eff / Math kernel, and
    gracefully fall back if something fails.
    """
    if ALLOW_ALL_KERNELS:
        # If we've already fallen back, just allow all kernels now
        return contextlib.nullcontext()

    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        enable_mem_efficient=OLD_GPU,
    )

# Optionally, if you want matrix-based RoPE
USE_MAT_ROTARY_ENC = True


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): shape (B, C, H, W)
          image_pe (torch.Tensor): shape (B, C, H, W), same as image_embedding
          point_embedding (torch.Tensor): shape (B, N_points, C)

        Returns:
          point_embedding_out, image_embedding_out
        """
        # Flatten the image to B x (HW) x C
        B, C, H, W = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)                # (B, HW, C)

        queries = point_embedding  # (B, N_pts, C)
        keys = image_embedding     # (B, HW,   C)

        # Run each transformer block
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,  # The "point" embedding as positional
                key_pe=image_pe,          # The "image" positional embedding
            )

        # Final attention from points -> image
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """
    A block with:
      1) Self-attn on the sparse queries
      2) Cross-attn (sparse->dense)
      3) MLP on queries
      4) Cross-attn (dense->sparse).
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLP(
            embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation
        )
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # (1) Self attention on queries
        if self.skip_first_layer_pe:
            # No positional encoding added if skip_first_layer_pe
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # (2) Cross attention: tokens -> image
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # (3) MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # (4) Cross attention: image -> tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    A basic multi-head attention layer, with optional downsample on Q/K/V.
    We also add a fallback logic for scaled_dot_product_attention if
    Flash Attention fails or is not available.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        kv_in_dim: int = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape  # (Batch, Tokens, Channels)
        x = x.reshape(b, n, num_heads, c // num_heads)
        # => (B, N, N_heads, C_per_head)
        return x.transpose(1, 2)  # => (B, N_heads, N, C_per_head)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)  # => (B, N, N_heads, C_per_head)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # 1) project Q, K, V
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 2) separate heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # 3) Possibly apply Flash / fallback
        dropout_p = self.dropout_p if self.training else 0.0
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception:
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            # fallback to normal
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # 4) recombine heads, final linear
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out


class RoPEAttention(Attention):
    """
    A specialized Attention that uses rotary position encoding
    (via either complex multiply or matrix-based real multiply).
    """
    def __init__(
        self,
        *args,
        rope_theta=10000.0,
        rope_k_repeat=False,
        feat_sizes=(64, 64),  # typical [width, height]
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.rope_k_repeat = rope_k_repeat

        # Build the standard "complex" approach:
        self.compute_cis = partial(
            compute_axial_cis,
            dim=self.internal_dim // self.num_heads,
            theta=rope_theta,
        )
        freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.freqs_cis = freqs_cis

        # If you want the matrix-based approach:
        global USE_MAT_ROTARY_ENC
        self.use_matrix = USE_MAT_ROTARY_ENC
        if self.use_matrix:
            # build a real rotation matrix once
            rotmats = get_rotation_matrices(
                dim=self.internal_dim // self.num_heads,
                end_x=feat_sizes[0],
                end_y=feat_sizes[1],
                theta=rope_theta,
            )
            self.rotmats = rotmats
            self.rope_theta = rope_theta

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        num_k_exclude_rope: int = 0
    ) -> Tensor:
        # 1) project Q, K, V
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # 2) separate heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # 3) apply rotary position encoding
        seq_len = q.shape[-2]
        w = h = math.isqrt(seq_len)  # or int(math.sqrt(seq_len))

        # Rebuild freq or rotation if needed
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != seq_len:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)

        if self.use_matrix:
            self.rotmats = self.rotmats.to(q.device)
            if self.rotmats.shape[0] != seq_len:
                self.rotmats = get_rotation_matrices(
                    dim=self.internal_dim // self.num_heads,
                    end_x=w,
                    end_y=h,
                    theta=self.rope_theta,
                ).to(q.device)

        # Possibly exclude some positions from rope, e.g. cross-attn
        num_k_rope = k.size(-2) - num_k_exclude_rope
        if self.use_matrix:
            # matrix-based approach
            q, k[:, :, :num_k_rope] = apply_rotary_matenc(
                q,
                k[:, :, :num_k_rope],
                rotmats=self.rotmats,
                repeat_freqs_k=self.rope_k_repeat,
            )
        else:
            # complex-based approach
            q, k[:, :, :num_k_rope] = apply_rotary_enc(
                q,
                k[:, :, :num_k_rope],
                freqs_cis=self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )

        # 4) scaled-dot-product attention w/ fallback
        dropout_p = self.dropout_p if self.training else 0.0
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception:
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out
