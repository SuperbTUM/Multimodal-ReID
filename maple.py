import copy
import math
import numpy as np
from typing import Tuple, Union
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CrossAttentionCoupling(nn.Module):
    """Cross-attention block for text-to-vision prompt coupling.

    Vision prompt tokens (Query) attend to encoded instruction tokens
    (Key/Value) so the vision branch can dynamically pull the exact
    visual-semantic cues it needs at every depth.
    """

    def __init__(self, vis_dim: int, text_dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = vis_dim // n_heads
        assert vis_dim % n_heads == 0, f"vis_dim ({vis_dim}) must be divisible by n_heads ({n_heads})"

        # Query projection: operates on vision prompt tokens (vis_dim)
        self.q_proj = nn.Linear(vis_dim, vis_dim)
        # Key/Value projections: operate on instruction tokens (text_dim)
        self.k_proj = nn.Linear(text_dim, vis_dim)
        self.v_proj = nn.Linear(text_dim, vis_dim)
        # Output projection
        self.out_proj = nn.Linear(vis_dim, vis_dim)

        self.ln_q = LayerNorm(vis_dim)
        self.ln_kv = LayerNorm(text_dim)

        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        # Q, K, V projections: normal init for expressivity
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.normal_(proj.weight, std=0.02)
            nn.init.zeros_(proj.bias)
        # Output projection: ZERO init (ControlNet / Flamingo technique).
        # At Step 0 out_proj produces zeros, so the residual connection
        # passes vision_queries through unchanged — preserving pre-trained
        # CLIP representations until the cross-attention slowly learns.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, vision_queries: torch.Tensor, instruction_tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            vision_queries:     [n_ctx_m, vis_dim] — current vision prompt tokens.
            instruction_tokens: [T, text_dim]      — encoded instruction token sequence.

        Returns:
            Updated vision prompts: [n_ctx_m, vis_dim]
        """
        # Layer-norm inputs
        q = self.ln_q(vision_queries)   # [n_ctx_m, vis_dim]
        kv = self.ln_kv(instruction_tokens)  # [T, text_dim]

        n_q = q.shape[0]
        n_kv = kv.shape[0]

        # Project
        Q = self.q_proj(q).view(n_q, self.n_heads, self.head_dim).transpose(0, 1)   # [H, n_q, d_h]
        K = self.k_proj(kv).view(n_kv, self.n_heads, self.head_dim).transpose(0, 1) # [H, n_kv, d_h]
        V = self.v_proj(kv).view(n_kv, self.n_heads, self.head_dim).transpose(0, 1) # [H, n_kv, d_h]

        # Scaled dot-product attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale  # [H, n_q, n_kv]
        attn = attn.float().softmax(dim=-1).type_as(Q)
        attn = self.dropout(attn)

        out = attn @ V  # [H, n_q, d_h]
        out = out.transpose(0, 1).contiguous().view(n_q, -1)  # [n_q, vis_dim]
        out = self.out_proj(out)

        # Residual connection
        return vision_queries + out


class VLPromptLearner(nn.Module):
    def __init__(self, n_cls, clip_model, dataset_name="market1501"):
        super().__init__()

        n_ctx = 4
        n_cls_ctx = 4
        if dataset_name in ("market1501", "dukemtmc", "msmt17"):
            ctx_init = "A photo of X X X X X person."
        else:
            ctx_init = "A photo of X X X X X vehicle."
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        tokenized_prompts = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        ctx_vectors = torch.empty(n_cls, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = ctx_init

        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_cls_ctx}")
        self.ctx = nn.Parameter(ctx_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1 + n_ctx, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + n_cls_ctx:, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prefix = prefix.expand(ctx.size(0), -1, -1)
        suffix = suffix.expand(ctx.size(0), -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, label):
        ctx = self.ctx[label]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


class VLPromptLearnerGPT4o(nn.Module):
    def __init__(self, n_cls, clip_model, prompts_path="prompts_market1501.txt"):
        super().__init__()

        prompts = []
        with open(prompts_path, "r") as f:
            while True:
                prompt = f.readline()
                if not prompt:
                    break
                label, desc = prompt.split(":")
                prompts.append(desc)
        f.close()

        assert len(prompts) == n_cls

        n_ctx = 4
        n_cls_ctx = 4

        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # use given words to initialize context vectors
        ctx_init = prompts  # .replace("_", " ")
        tokenized_prompts = clip.tokenize(ctx_init).cuda()  # [n_cls, 77, 512]
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        ctx_vectors = torch.empty(n_cls, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = ctx_init

        print(f"Independent V-L design")
        self.ctx = nn.Parameter(ctx_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1 + n_ctx, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:-n_cls_ctx, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        # prefix = prefix.expand(ctx.size(0), -1, -1)
        # suffix = suffix.expand(ctx.size(0), -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, label):
        ctx = self.ctx[label]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix, label)

        return prompts


class VLPromptLearnerVeri(nn.Module):

    n_cls_ctx = 4

    car_type_explanation = {
        "sedan": "{} sedan, a type of passenger car that typically features a lower profile, sleeker lines, a fixed roof, four doors, and a separate trunk compartment for cargo.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "suv": "{} SUV, a type of passenger car that typically features a taller body with a boxy shape, a high ground clearance, and a spacious interior capable of accommodating multiple passengers and cargo.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "van": "{} van, a spacious vehicle that features a boxy design, large cargo capacity, and multiple seating configurations.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "hatchback": "{} hatchback, a compact car that features a rear door opening upwards to access a cargo area.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "mpv": "{} MPV (Multi-Purpose Vehicle), a versatile automobile that features multiple seating configurations, ample interior space, and sliding doors.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "pickup": "{} pickup, a rugged vehicle that features an open cargo area at the rear, often equipped with towing capabilities and four-wheel drive.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "bus": "{} bus, a large vehicle that features multiple rows of seating, wide windows, and a distinctive boxy shape.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "truck": "{} truck, a robust vehicle that features a separate cabin and cargo area, often with a towing hitch, powerful engine, and sturdy chassis.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "estate": "{} estate, a versatile vehicle that features a spacious cargo area extending from the rear of the cabin, often with a sloping roofline and folding rear seats.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
        "": "{} background.".format(" ".join(["X" for _ in range(n_cls_ctx-1)])),
    }

    def __init__(self, num_class, clip_model, car_types):
        super().__init__()
        ctx_inits = []
        for car_type in car_types:
            # ctx_init = "A photo of X X X {}, a type of vehicle.".format(car_type)
            car_type_desc = car_type.split(" ")
            if isinstance(car_type_desc, list) and len(car_type_desc) == 2:
                sentence = " ".join([self.car_type_explanation[car_type_desc[1]][:(self.n_cls_ctx-1)*2-1], car_type_desc[0], self.car_type_explanation[car_type_desc[1]][(self.n_cls_ctx-1)*2:]])
                ctx_init = "A photo of X " + sentence
            else:
                ctx_init = "A photo of X " + self.car_type_explanation[car_type]
            ctx_init = ctx_init.replace("_", " ")
            ctx_inits.append(ctx_init)

        tokenized_prompts = torch.cat([clip.tokenize(ctx_init).cuda() for ctx_init in ctx_inits])

        dtype = clip_model.dtype
        token_embedding = clip_model.token_embedding
        ctx_dim = 512
        # use given words to initialize context vectors
        n_ctx = 3

        with torch.no_grad():
            embedding = token_embedding(tokenized_prompts).type(dtype)
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        cls_vectors = torch.empty(num_class, self.n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(cls_vectors, std=0.02)
        prompt_prefix = ctx_init

        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        self.ctx = nn.Parameter(cls_vectors)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :n_ctx + 1, :])
        self.register_buffer("token_suffix", embedding[:, n_ctx + 1 + self.n_cls_ctx:, :])
        self.num_class = num_class

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        else:
            prefix = prefix.expand(ctx.size(0), -1, -1)
            suffix = suffix.expand(ctx.size(0), -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, label):
        ctx = self.ctx[label]
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix, label)

        return prompts


class VLPromptLearnerSRC(nn.Module):
    def __init__(self, n_cls, clip_model, dataset_name="market1501"):
        super().__init__()

        n_ctx = 4
        n_cls_ctx = 4
        ctx_dim = 512
        if dataset_name in ("market1501", "dukemtmc", "msmt17"):
            ctx_init = "A photo of X X X X X person."
        else:
            ctx_init = "A photo of X X X X X vehicle."
        dtype = clip_model.dtype

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = n_ctx
        tokenized_prompts = clip.tokenize(ctx_init).cuda()

        ctx_vectors = torch.empty(n_cls, n_cls_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)

        # Also create frozen CLIP
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1 + n_ctx, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + n_cls_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prefix = prefix.expand(ctx.size(0), -1, -1)
        suffix = suffix.expand(ctx.size(0), -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, label):
        ctx = self.ctx[label]
        b = label.shape[0]

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        else:
            ctx = ctx.expand(b, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        return prompts


import random
class VLPromptLearnerCSC(nn.Module):
    def __init__(self, n_cls, clip_model, dataset_name="market1501", n_ctx_s=4, n_ctx_m=4, prompt_depth=12, unified_context=True):
        super().__init__()
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        n_ctx = 4
        # We use ctx_s for identity-specific context and ctx_m for shared ReID context.
        self.n_placeholders = n_ctx_s + n_ctx_m
        placeholders = " ".join(["X"] * self.n_placeholders)

        is_vehicle = dataset_name not in ("market1501", "dukemtmc", "msmt17")
        if not is_vehicle:
            self.INSTRUCTION_POOL = [
                f"A photo of a {placeholders} person. Find the same person. Ignore clothing and illumination changes.",
                f"Capture the image of a {placeholders} pedestrian. Match the same individual under various surveillance cameras.",
                f"Look at this {placeholders} person. Track their identity across distinct multi-camera networks.",
                f"A snapshot of a {placeholders} individual. Find the matching target while disregarding clothes variations.",
                # Semantic Swapping
                f"Find this {placeholders} person, ignoring transient clothing colors.",
                f"Search for this {placeholders} pedestrian focusing only on invariant physical traits.",
                f"Retrieve the identity matching this {placeholders} subject regardless of their outfit.",
                # Instruction Dropout (Unconditional Fallback)
                f"A photo of a {placeholders} person."
            ]
        else:
            self.INSTRUCTION_POOL = [
                f"A photo of a {placeholders} vehicle. Find the identical car by analyzing grille geometry and unique markers.",
                f"An automobile captured on camera. Track this same {placeholders} vehicle across distinct traffic surveillance views.",
                f"Look at this {placeholders} vehicle. Identify the matching car based on body shapes and window layouts.",
                f"A snapshot of a {placeholders} car. Find this identical target across different traffic camera networks.",
                # Semantic Swapping
                f"Find this identical {placeholders} vehicle, ignoring lighting and camera differences.",
                f"Search for this {placeholders} car focusing only on invariant structural traits.",
                f"Retrieve the vehicle matching this {placeholders} target regardless of viewpoint.",
                # Instruction Dropout (Unconditional Fallback)
                f"A photo of a {placeholders} vehicle."
            ]

        for i, instruct in enumerate(self.INSTRUCTION_POOL):
            # Dynamically calculate the token length of the prefix string
            prefix_str = instruct.split(placeholders)[0]
            prefix_tokens = clip.tokenize(prefix_str)[0]
            # In CLIP, the EOS token is the max token ID (49407), so argmax finds its index.
            # The index of EOS perfectly matches the length of the prefix including the SOS token.
            prefix_len = prefix_tokens.argmax().item()

            tokens = clip.tokenize(instruct).cuda()
            with torch.no_grad():
                emb = clip_model.token_embedding(tokens).type(dtype)
            
            prefix_slice = emb[:, :prefix_len, :]
            suffix_slice = emb[:, prefix_len + self.n_placeholders:, :]

            # Run full instruction through the frozen Text Transformer to get
            # contextualized hidden states. Raw token_embedding outputs lack
            # inter-word context (e.g. "red" has not attended to "backpack").
            # The resulting contextualized suffix tokens serve as Keys/Values
            # in the Cross-Attention layers of the vision branch.
            with torch.no_grad():
                x_full = emb + clip_model.positional_embedding.type(dtype)
                x_full = x_full.permute(1, 0, 2)  # NLD -> LND
                x_ctx = clip_model.transformer([x_full, [], 0])
                if isinstance(x_ctx, (list, tuple)):
                    x_ctx = x_ctx[0]
                x_ctx = x_ctx.permute(1, 0, 2)  # LND -> NLD
                x_ctx = clip_model.ln_final(x_ctx).type(dtype)

            # Extract contextualized suffix (instruction) tokens for Cross-Attention K/V
            ctx_suffix = x_ctx[:, prefix_len + self.n_placeholders:, :]  # [1, T, ctx_dim]
            instruct_emb_slice = ctx_suffix.mean(dim=1)  # [1, ctx_dim]
            instruct_tokens_full = ctx_suffix  # [1, T, ctx_dim]

            self.register_buffer(f"token_prefix_{i}", prefix_slice)
            self.register_buffer(f"token_suffix_{i}", suffix_slice)
            self.register_buffer(f"instruction_emb_{i}", instruct_emb_slice)
            self.register_buffer(f"instruction_tokens_{i}", instruct_tokens_full)
            self.register_buffer(f"tokenized_prompts_{i}", tokens)

        self.ctx_s = nn.Parameter(torch.empty(1 if unified_context else n_cls, n_ctx_s, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_s, std=0.02)
        
        self.ctx_m = nn.Parameter(torch.empty(1, n_ctx_m, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx_m, std=0.02)

        if hasattr(clip_model.visual, "class_embedding") and clip_model.visual.class_embedding is not None:
            vis_dim = clip_model.visual.class_embedding.shape[0]
        elif hasattr(clip_model.visual, "positional_embedding") and clip_model.visual.positional_embedding is not None:
            vis_dim = clip_model.visual.positional_embedding.shape[-1]
        else:
            vis_dim = clip_model.visual.conv1.out_channels

        # ── Independent Projection Layers (Depth-Specific) ──
        # Instead of a single shared linear layer, we use independent layers for each depth.
        self.proj = nn.ModuleList([nn.Linear(ctx_dim, vis_dim).to(dtype) for _ in range(prompt_depth)])
        for p in self.proj:
            nn.init.normal_(p.weight, std=0.02)
            nn.init.zeros_(p.bias)

        # ── Meta-Network for Layer-0 Vision Context (Instruction Injection) ──
        self.meta_net_layer0 = nn.Sequential(
            nn.Linear(ctx_dim, ctx_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ctx_dim, n_ctx_m * vis_dim),
        ).to(dtype)
        self.ln_layer0 = LayerNorm(vis_dim).to(dtype)
        self.layer0_gate = nn.Parameter(torch.zeros(1, dtype=dtype))

        for m in self.meta_net_layer0.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.meta_net_layer0[-1].weight)
        nn.init.zeros_(self.meta_net_layer0[-1].bias)

        # ── Cross-Attention for Deep Layers (Instruction Injection) ──
        n_heads_xattn = max(1, vis_dim // 64)
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionCoupling(
                vis_dim=vis_dim, text_dim=ctx_dim,
                n_heads=n_heads_xattn, dropout=0.0
            ).to(dtype) for _ in range(prompt_depth - 1)
        ])

        self.vis_dim = vis_dim
        self.n_cls = n_cls
        self.n_ctx_s = n_ctx_s
        self.n_ctx_m = n_ctx_m
        self.prompt_depth = prompt_depth
        self.unified_context = unified_context

        # Deep text prompts removed; vision prompts generated entirely via projection.

    @property
    def tokenized_prompts(self):
        if hasattr(self, 'current_tokenized_prompts'):
            return self.current_tokenized_prompts
        return self.tokenized_prompts_0

    def _generate_vision_prompts(self, idx):
        instruction_emb = getattr(self, f"instruction_emb_{idx}")    # [1, ctx_dim]
        instruction_tokens = getattr(self, f"instruction_tokens_{idx}")  # [1, T, ctx_dim]
        inst_tok = instruction_tokens.squeeze(0)  # [T, ctx_dim]

        m_c = self.ctx_m.squeeze(0) # [n_ctx_m, ctx_dim]
        
        # Layer 0
        base_vision_ctx_layer0 = self.proj[0](m_c) # [n_ctx_m, vis_dim]
        layer0_delta = self.meta_net_layer0(instruction_emb)  # [1, n_ctx_m * vis_dim]
        layer0_delta = layer0_delta.view(self.n_ctx_m, self.vis_dim)  # [n_ctx_m, vis_dim]
        layer0_delta = self.ln_layer0(layer0_delta)
        gate = torch.tanh(self.layer0_gate)
        shared_ctx_vision = base_vision_ctx_layer0 + gate * layer0_delta  # [n_ctx_m, vis_dim]

        # Deeper Layers
        deeper_vision_prompts = []
        for i in range(self.prompt_depth - 1):
            query_seed = self.proj[i+1](m_c)  # [n_ctx_m, vis_dim]
            vision_prompt = self.cross_attn_layers[i](query_seed, inst_tok)  # [n_ctx_m, vis_dim]
            deeper_vision_prompts.append(vision_prompt)

        return shared_ctx_vision, deeper_vision_prompts

    def forward(self, label, is_stage2=False, ensemble=False, force_idx=None):
        batch_size = label.shape[0] if label is not None else 1

        if ensemble:
            all_shared = []
            all_deeper = []
            for idx in range(len(self.INSTRUCTION_POOL)):
                shared_v, deeper_v = self._generate_vision_prompts(idx)
                all_shared.append(shared_v)
                all_deeper.append(deeper_v)
            shared_ctx_vision = torch.stack(all_shared, dim=0).mean(dim=0)
            depth = len(all_deeper[0])
            deeper_vision_prompts = [
                torch.stack([all_deeper[k][d] for k in range(len(self.INSTRUCTION_POOL))], dim=0).mean(dim=0)
                for d in range(depth)
            ]
            idx = 0
        else:
            if force_idx is not None:
                idx = force_idx
            elif self.training:
                idx = random.randint(0, len(self.INSTRUCTION_POOL) - 1)
            else:
                idx = 0
            shared_ctx_vision, deeper_vision_prompts = self._generate_vision_prompts(idx)

        self.current_idx = idx
        self.current_tokenized_prompts = getattr(self, f"tokenized_prompts_{idx}")
        current_prefix = getattr(self, f"token_prefix_{idx}")
        current_suffix = getattr(self, f"token_suffix_{idx}")

        deeper_text_prompts = None

        if self.unified_context:
            s_c = self.ctx_s.expand(batch_size, -1, -1)
        else:
            if label is None:
                s_c = self.ctx_s[0].unsqueeze(0).expand(batch_size, -1, -1)
            else:
                s_c = self.ctx_s[label]

        m_c_batch = self.ctx_m.expand(batch_size, -1, -1)
        prefix = current_prefix.expand(batch_size, -1, -1)
        suffix = current_suffix.expand(batch_size, -1, -1)
        prompts = torch.cat([prefix, s_c, m_c_batch, suffix], dim=1)

        return prompts, deeper_text_prompts, deeper_vision_prompts, shared_ctx_vision


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text=None):
        if isinstance(prompts, (list, tuple)):
            prompts, compound_prompts_deeper_text, _, _ = prompts
        
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text if compound_prompts_deeper_text is not None else [], 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == 'IVLP' or current_trainer == 'VPT':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock_IVLP(width, heads, attn_mask, True,
                                                                         text_layer, i,
                                                                         design_details) if prompts_needed > i
                                             else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False,
                                                                              text_layer, i, design_details)
                                             for i in range(layers)])
        elif current_trainer == 'MaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
        else:
            # Corresponds to default CoOp or CoCoOp
            assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]  # hyperparameter
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                # Code snippet for per layer visual prompts
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_shallow = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_shallow.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_MaPLe(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details.get('maple_length', 4)
        self.n_ctx_s = design_details.get('n_ctx_s', 0)
        self.n_ctx = design_details.get('n_ctx', 4) if self.n_ctx_s > 0 else 0
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added
        # as the shallow version
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        
        injecting_deep_prompts = False

        if not self.first_layer:
            if len(compound_prompts_deeper) > 0:
                # This means that deeper compound prompts are turned on
                if not (counter > len(compound_prompts_deeper) - 1):
                    injecting_deep_prompts = True
                    if not self.text_layer:
                        # Vision side: prompts after CLS
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        x = torch.cat([prefix, visual_context, suffix], dim=0)
                    else:
                        # Text side: SOS + n_ctx + n_ctx_s + ctx_m + suffix
                        prefix_len = 1 + self.n_ctx + self.n_ctx_s
                        prefix = x[:prefix_len, :, :]
                        suffix = x[prefix_len + self.compound_prompt_nctx:, :, :]
                        textual_context = compound_prompts_deeper[counter]
                        textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                    
                    # Once done, update the counter
                    counter += 1

        # Isolate the [CLS] token during the text-vision attention phase.
        # This prevents the text from overwhelming the identity-carrying [CLS] token,
        # forcing the text to act strictly as a spatial filter on the image patches.
        if not self.text_layer and (self.first_layer or injecting_deep_prompts):
            cls_token = x[:1, :, :]
            patch_tokens = x[1:, :, :]
            patch_tokens = patch_tokens + self.attention(self.ln_1(patch_tokens))
            x = torch.cat([cls_token, patch_tokens], dim=0)
        else:
            x = x + self.attention(self.ln_1(x))

        x = x + self.mlp(self.ln_2(x))
        return [x, compound_prompts_deeper, counter]  # return again as a list, so that nn.seq can work


class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details, stride_size):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)
        if design_details["vision_depth"] == 0:
            self.VPT_shallow = False
        else:
            self.VPT_shallow = True
        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((h_resolution*w_resolution) + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = design_details["vision_depth"]
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_till_layer_visual,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x11 = self.transformer.resblocks[:11](x)
        x12 = self.transformer.resblocks[11](x11)
        x11 = x11.permute(1, 0, 2)  # LND -> NLD
        x12 = x12.permute(1, 0, 2)

        x12 = self.ln_post(x12)

        if self.proj is not None:
            x_proj = x12 @ self.proj

        return x11, x12, x_proj


class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 design_details, stride_size: int):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((h_resolution*w_resolution) + 1, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, shared_ctx, compound_deeper_prompts):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x[:, 0:1, :], visual_ctx, x[:, 1:, :]], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # Again combine the inputs, so nn.sequential can work
        x11 = self.transformer.resblocks[:11]([x, compound_deeper_prompts, 0])  # third argument is counter
        x12 = self.transformer.resblocks[11](x11)

        x11 = x11[0]
        x11 = x11.permute(1, 0, 2)  # LND -> NLD
        x12 = x12[0]
        x12 = x12.permute(1, 0, 2)

        x12 = self.ln_post(x12)

        if self.proj is not None:
            x_proj = x12 @ self.proj

        return x11, x12, x_proj


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 h_resolution: int,
                 w_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details,
                 stride_size: int = 16
                 ):
        super().__init__()

        self.context_length = context_length
        trainer = design_details['trainer']

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=h_resolution*w_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if trainer == "MaPLe":
                self.visual = VisionTransformer_MaPLe(
                    h_resolution=h_resolution,
                    w_resolution=w_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details,
                    stride_size=stride_size
                )
            else:
                self.visual = VisionTransformer(
                    h_resolution=h_resolution,
                    w_resolution=w_resolution,
                    patch_size=vision_patch_size,
                    width=vision_width,
                    layers=vision_layers,
                    heads=vision_heads,
                    output_dim=embed_dim,
                    design_details=design_details,
                    stride_size=stride_size
                )
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        prompt_till_layer_text = design_details['language_depth']
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def resize_pos_embed(posemb, posemb_new, height, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[0]

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, height, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode='bicubic')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze(0)], dim=0)
    return posemb


def load_pretrained_maple_weights(model, weight_path, learners=None):
    print(f"Loading specialized MaPLe weights from {weight_path}")
    checkpoint = torch.load(weight_path, map_location="cuda")
    state_dict = checkpoint.get("state_dict", checkpoint)

    # 1. Map Learner Parameters
    target_learners = []
    if learners is not None:
        if isinstance(learners, (list, tuple)):
            target_learners.extend(learners)
        else:
            target_learners.append(learners)
    elif hasattr(model, 'prompt_learner'):
        target_learners.append(model.prompt_learner)

    for learner in target_learners:
        if isinstance(learner, VLPromptLearnerCSC):
            if "prompt_learner.ctx_s" in state_dict:
                learner.ctx_s.data.copy_(state_dict["prompt_learner.ctx_s"])

            # Old coupling_layers (nn.Linear projections) no longer exist.
            # meta_net_layer0 and cross_attn_layers use new architectures
            # and will be trained from scratch. Skip old proj/projection weights.
            if "prompt_learner.proj.weight" in state_dict:
                print("  [INFO] Skipping old proj weights (replaced by meta_net_layer0)")
            for i in range(learner.prompt_depth - 1):
                key_w = f"prompt_learner.compound_prompt_projections.{i}.weight"
                if key_w in state_dict:
                    print(f"  [INFO] Skipping old compound_prompt_projections.{i} (replaced by cross_attn_layers)")

            if "prompt_learner.token_prefix" in state_dict:
                learner.token_prefix.data.copy_(state_dict["prompt_learner.token_prefix"])
            if "prompt_learner.token_suffix" in state_dict:
                learner.token_suffix.data.copy_(state_dict["prompt_learner.token_suffix"])

    # 2. Map Encoders
    # ... rest of logic ...
    if hasattr(model, 'image_encoder'):
        vis_dict = {k.replace("image_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.")}
        model.image_encoder.load_state_dict(vis_dict, strict=False)

    # 3. Map Text Encoder
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        txt_dict = {k.replace("text_encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.")}
        model.text_encoder.load_state_dict(txt_dict, strict=False)

    # 4. Logit Scale
    if "logit_scale" in state_dict:
        model.logit_scale.data.copy_(state_dict["logit_scale"])
    
    print("CSC-MaPLe weights loaded successfully (identity-specific prompts preserved)")


def build_model(state_dict: dict, h_resolution, w_resolution, design_details, stride_size=16, **kwargs):
    design_details.update(kwargs)
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    h_resolution = h_resolution // stride_size
    w_resolution = w_resolution // stride_size

    model = CLIP(
        embed_dim,
        h_resolution, w_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details,
        stride_size
    )

    if vit:
        state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"],
                                                                     model.visual.positional_embedding, h_resolution,
                                                                     w_resolution)
    else:  # RN50
        state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(
            state_dict["visual.attnpool.positional_embedding"], model.visual.attnpool.positional_embedding,
            h_resolution, w_resolution)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()
