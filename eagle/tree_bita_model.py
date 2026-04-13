# coding=utf-8
"""
tree_bita_model.py
==================
BiTA-adapted EAGLE-3 draft model for single-pass tree drafting.

Implements a **2x2 Mini-Tree Topology**:

    Root (last accepted token)
     ├── M_1a (depth 1, Branch A) → predicts t+1
     ├── M_1b (depth 1, Branch B) → predicts t+1
     │     │
     M_2a (depth 2, Branch A)     M_2b (depth 2, Branch B)
       → sees M_1a, predicts t+2    → sees M_1b, predicts t+2

    Branch A path: Root → M_1a → M_2a
    Branch B path: Root → M_1b → M_2b

Attention mask rules:
  - Real tokens: standard causal. CANNOT see [P] or [M].
  - M_1a, M_1b: see Real + [P]. CANNOT see each other.
  - M_2a: sees Real + [P] + M_1a. CANNOT see M_1b or M_2b.
  - M_2b: sees Real + [P] + M_1b. CANNOT see M_1a or M_2a.
"""

import os
import math
import copy
import json
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# ─── EAGLE-3 backbone imports ───
try:
    from .model.cnets import Model as EAGLE3Model
    from .model.cnets import LlamaRMSNorm
    from .model.configs import EConfig
except ImportError:
    from model.cnets import Model as EAGLE3Model
    from model.cnets import LlamaRMSNorm
    from model.configs import EConfig


# ═══════════════════════════════════════════════════════════════════════════════
# 2x2 Mini-Tree Topology
# ═══════════════════════════════════════════════════════════════════════════════
# 4 mask tokens total:
#   Index 0 = M_1a (depth 1, Branch A)
#   Index 1 = M_1b (depth 1, Branch B)
#   Index 2 = M_2a (depth 2, Branch A) — parent is M_1a (index 0)
#   Index 3 = M_2b (depth 2, Branch B) — parent is M_1b (index 1)

NUM_MASK_TOKENS = 4
# Slot name constants for readability
M_1A, M_1B, M_2A, M_2B = 0, 1, 2, 3

# Parent map: child_idx → parent_idx (-1 = no [M] parent, root child)
MINI_TREE_PARENT_MAP = {
    M_1A: -1,   # M_1a has no [M] parent
    M_1B: -1,   # M_1b has no [M] parent
    M_2A: M_1A, # M_2a's parent is M_1a
    M_2B: M_1B, # M_2b's parent is M_1b
}

# Depth map: slot_idx → depth (1-based)
MINI_TREE_DEPTH_MAP = {
    M_1A: 1,
    M_1B: 1,
    M_2A: 2,
    M_2B: 2,
}

# Branch assignments for inference extraction
BRANCH_A = [M_1A, M_2A]  # path: Root → M_1a → M_2a
BRANCH_B = [M_1B, M_2B]  # path: Root → M_1b → M_2b


def build_2x2_tree_attention_mask(
    num_prompt: int,
    num_real: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build the 2x2 Mini-Tree Attention Mask.

    Layout: [P_1..P_p | R_1..R_n | M_1a, M_1b, M_2a, M_2b]
    Total length = p + n + 4

    Visibility rules:
      - [P] tokens: bidirectional among themselves.
      - Real tokens: standard causal mask. CANNOT see [P] or [M].
      - M_1a (idx p+n+0): sees [P], Real. NOT M_1b, M_2a, M_2b.
      - M_1b (idx p+n+1): sees [P], Real. NOT M_1a, M_2a, M_2b.
      - M_2a (idx p+n+2): sees [P], Real, M_1a. NOT M_1b, M_2b.
      - M_2b (idx p+n+3): sees [P], Real, M_1b. NOT M_1a, M_2a.

    Returns:
        mask: (1, 1, total_len, total_len) — 0 = attend, -inf = block
    """
    p = num_prompt
    n = num_real
    k = NUM_MASK_TOKENS  # always 4
    total = p + n + k

    mask = torch.full((total, total), float("-inf"), dtype=dtype, device=device)

    # ── [P] tokens: bidirectional among themselves ──
    mask[:p, :p] = 0.0

    # ── Real tokens: causal among themselves only ──
    for i in range(n):
        ri = p + i
        mask[ri, p: ri + 1] = 0.0  # sees preceding reals (inclusive)

    # ── Mask tokens ──
    m_start = p + n  # starting index of [M] block

    for mi in range(k):
        mi_abs = m_start + mi

        # All [M] tokens can see [P]
        mask[mi_abs, :p] = 0.0

        # All [M] tokens can see all Real tokens
        mask[mi_abs, p: p + n] = 0.0

        # Self-attention
        mask[mi_abs, mi_abs] = 0.0

    # ── Branch-specific parent visibility ──
    # M_2a (index 2) sees M_1a (index 0)
    mask[m_start + M_2A, m_start + M_1A] = 0.0

    # M_2b (index 3) sees M_1b (index 1)
    mask[m_start + M_2B, m_start + M_1B] = 0.0

    # Everything else between [M] tokens stays -inf (blocked):
    #   M_1a ✗ M_1b (siblings)
    #   M_2a ✗ M_2b (siblings)
    #   M_2a ✗ M_1b (cross-branch)
    #   M_2b ✗ M_1a (cross-branch)
    #   M_1a ✗ M_2a, M_2b (parent doesn't look at children)
    #   M_1b ✗ M_2a, M_2b (parent doesn't look at children)

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, total, total)


def build_2x2_position_ids(
    num_prompt: int,
    num_real: int,
    last_real_pos: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build position IDs for [P | Real | M_1a, M_1b, M_2a, M_2b].

    - [P]: positions 0..p-1
    - Real: last_real_pos - n + 1 .. last_real_pos
    - M_1a, M_1b: last_real_pos + 1  (both predict t+1)
    - M_2a, M_2b: last_real_pos + 2  (both predict t+2)
    """
    p = num_prompt
    n = num_real
    total = p + n + NUM_MASK_TOKENS
    pos = torch.zeros(total, dtype=torch.long, device=device)

    # [P]
    for i in range(p):
        pos[i] = i

    # Real
    real_start = max(0, last_real_pos - n + 1)
    for i in range(n):
        pos[p + i] = real_start + i

    # [M] tokens
    m_start = p + n
    pos[m_start + M_1A] = last_real_pos + 1   # depth 1
    pos[m_start + M_1B] = last_real_pos + 1   # depth 1
    pos[m_start + M_2A] = last_real_pos + 2   # depth 2
    pos[m_start + M_2B] = last_real_pos + 2   # depth 2

    return pos.unsqueeze(0)  # (1, total)


class TreeBiTAAdapter(nn.Module):
    """
    Wraps a frozen EAGLE-3 draft model backbone with BiTA-style
    learnable [P] prompt and [M] mask embeddings for single-pass
    2x2 Mini-Tree drafting.

    2x2 Mini-Tree:
        Root (sample_token)
         ├── M_1a → M_2a   (Branch A)
         └── M_1b → M_2b   (Branch B)

    The EAGLE-3 backbone (LlamaDecoderLayeremb) expects two channels:
        - input_emb:     token embeddings (hidden_size)
        - hidden_states: target model hidden states projected to hidden_size

    For [M] positions:
        - input_emb = mask_embeddings(slot_idx)
        - hidden_states = last_accepted_hidden + mask_hidden (learnable)

    For [P] positions:
        - input_emb = prompt_embeddings(idx)
        - hidden_states = prompt_hidden (learnable)

    For Real positions:
        - input_emb = embed_tokens(token_id)  [frozen]
        - hidden_states = fc(target_hidden_states)
    """

    def __init__(
        self,
        eagle3_config: EConfig,
        num_prompt_tokens: int = 8,
        top_k: int = 4,
        total_tokens: int = 63,
        depth: int = 2,
        threshold: float = 1.0,
    ):
        super().__init__()
        self.config = eagle3_config
        self.hidden_size = eagle3_config.hidden_size

        # Fixed 2x2 mini-tree topology
        self.num_mask_slots = NUM_MASK_TOKENS  # always 4
        self.parent_map = MINI_TREE_PARENT_MAP
        self.depth_map = MINI_TREE_DEPTH_MAP
        self.num_prompt_tokens = num_prompt_tokens
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth_val = depth
        self.threshold = math.log(threshold) if threshold > 0 else 0.0

        # ─── Frozen EAGLE-3 backbone components ───
        self.embed_tokens = nn.Embedding(
            eagle3_config.vocab_size, eagle3_config.hidden_size,
            eagle3_config.pad_token_id
        )
        self.lm_head = nn.Linear(
            eagle3_config.hidden_size,
            eagle3_config.draft_vocab_size,
            bias=False
        )
        self.norm = LlamaRMSNorm(eagle3_config.hidden_size, eps=eagle3_config.rms_norm_eps)

        # The single decoder layer
        from model.cnets import LlamaDecoderLayeremb
        self.midlayer = LlamaDecoderLayeremb(eagle3_config)

        # fc: projects concatenated target hidden states → hidden_size
        if hasattr(eagle3_config, "target_hidden_size"):
            fc_in = eagle3_config.target_hidden_size * 3
        else:
            fc_in = eagle3_config.hidden_size * 3
        self.fc = nn.Linear(fc_in, self.hidden_size, bias=False)

        # Draft vocab mapping buffers
        d2t = torch.zeros(eagle3_config.draft_vocab_size, dtype=torch.long)
        t2d = torch.zeros(eagle3_config.vocab_size, dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # ─── NEW: Learnable BiTA embeddings (ONLY trainable parameters) ───
        # [P] prompt embeddings
        self.prompt_embeddings = nn.Embedding(num_prompt_tokens, self.hidden_size)
        self.prompt_hidden = nn.Embedding(num_prompt_tokens, self.hidden_size)

        # [M] mask embeddings: exactly 4 (M_1a, M_1b, M_2a, M_2b)
        self.mask_embeddings = nn.Embedding(NUM_MASK_TOKENS, self.hidden_size)
        self.mask_hidden = nn.Embedding(NUM_MASK_TOKENS, self.hidden_size)

        # Initialize
        nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
        nn.init.normal_(self.prompt_hidden.weight, std=0.02)
        nn.init.normal_(self.mask_embeddings.weight, std=0.02)
        nn.init.normal_(self.mask_hidden.weight, std=0.02)

    def freeze_backbone(self):
        """Freeze all EAGLE-3 backbone params, keep only BiTA embeddings trainable."""
        for name, param in self.named_parameters():
            if any(x in name for x in [
                "prompt_embeddings", "prompt_hidden",
                "mask_embeddings", "mask_hidden"
            ]):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def load_eagle3_weights(self, state_dict: dict):
        """Load pre-trained EAGLE-3 weights into backbone components."""
        backbone_keys = {}
        for k, v in state_dict.items():
            if any(x in k for x in ["prompt_", "mask_embeddings", "mask_hidden"]):
                continue
            backbone_keys[k] = v
        missing, unexpected = self.load_state_dict(backbone_keys, strict=False)
        print(f"[TreeBiTAAdapter] Loaded backbone. "
              f"Missing (BiTA params): {len(missing)}, Unexpected: {len(unexpected)}")
        self.freeze_backbone()

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward_single_pass(
        self,
        hidden_states_context: torch.Tensor,
        input_ids_context: torch.Tensor,
        last_hidden_for_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-pass forward through the adapted EAGLE-3 backbone
        with the 2x2 Mini-Tree mask.

        Args:
            hidden_states_context: (1, n_ctx, hidden_size) projected target hidden states
            input_ids_context: (1, n_ctx) token IDs for context
            last_hidden_for_mask: (1, 1, hidden_size) last accepted hidden state
            position_ids: (1, total_len) or None

        Returns:
            mask_logits: (1, 4, draft_vocab_size) — logits at [M_1a, M_1b, M_2a, M_2b]
            all_hidden: (1, total_len, hidden_size) — full hidden states
        """
        device = hidden_states_context.device
        dtype = hidden_states_context.dtype
        bsz = hidden_states_context.shape[0]
        n_ctx = input_ids_context.shape[1]
        n_prompt = self.num_prompt_tokens
        n_mask = NUM_MASK_TOKENS  # 4
        total_len = n_prompt + n_ctx + n_mask

        # ─── Build input_emb channel ───
        prompt_idx = torch.arange(n_prompt, device=device)
        prompt_emb = self.prompt_embeddings(prompt_idx).unsqueeze(0).to(dtype)

        with torch.no_grad():
            real_emb = self.embed_tokens(input_ids_context).to(dtype)

        mask_idx = torch.arange(n_mask, device=device)
        mask_emb = self.mask_embeddings(mask_idx).unsqueeze(0).to(dtype)

        input_emb = torch.cat([prompt_emb, real_emb, mask_emb], dim=1)

        # ─── Build hidden_states channel ───
        prompt_hid = self.prompt_hidden(prompt_idx).unsqueeze(0).to(dtype)
        real_hid = hidden_states_context
        mask_hid_base = last_hidden_for_mask.expand(bsz, n_mask, -1).to(dtype)
        mask_hid_learn = self.mask_hidden(mask_idx).unsqueeze(0).to(dtype)
        mask_hid = mask_hid_base + mask_hid_learn

        hidden_states = torch.cat([prompt_hid, real_hid, mask_hid], dim=1)

        # ─── Build 2x2 tree attention mask ───
        tree_attn_mask = build_2x2_tree_attention_mask(
            num_prompt=n_prompt,
            num_real=n_ctx,
            dtype=torch.float32,
            device=device,
        )

        # ─── Build position IDs ───
        if position_ids is None:
            last_real_pos = n_ctx - 1
            position_ids = build_2x2_position_ids(
                num_prompt=n_prompt,
                num_real=n_ctx,
                last_real_pos=last_real_pos,
                device=device,
            )

        # ─── Forward through frozen EAGLE-3 decoder layer ───
        layer_outputs = self.midlayer(
            input_emb=input_emb,
            hidden_states=hidden_states,
            attention_mask=tree_attn_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
        all_hidden = layer_outputs[0]  # (1, total, hidden_size)

        # ─── Extract [M] logits ───
        mask_hidden_out = all_hidden[:, n_prompt + n_ctx:, :]  # (1, 4, h)
        mask_hidden_normed = self.norm(mask_hidden_out)
        mask_logits = self.lm_head(mask_hidden_normed)  # (1, 4, draft_vocab)

        return mask_logits, all_hidden

    def _map_draft_to_target_token(self, draft_token_id: torch.Tensor) -> torch.Tensor:
        """Map a draft vocab token ID to target vocab token ID."""
        if self.config.vocab_size != self.config.draft_vocab_size:
            return self.d2t[draft_token_id]
        return draft_token_id

    @torch.no_grad()
    def single_pass_draft(
        self,
        hidden_states_context: torch.Tensor,
        input_ids_context: torch.Tensor,
        last_hidden_for_mask: torch.Tensor,
        sample_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate 2x2 Mini-Tree draft via single forward pass.

        Produces 2 branches of depth 2:
            Branch A: sample_token → Top1(M_1a) → Top1(M_2a)
            Branch B: sample_token → Top1(M_1b) → Top1(M_2b)

        Returns:
            draft_tokens:      (1, 5) = [sample, m1a_tok, m1b_tok, m2a_tok, m2b_tok]
            retrieve_indices:  (2, 3)  = [[0, 1, 3], [0, 2, 4]]
                               paths for Branch A and B through draft_tokens
            tree_mask:         (1, 1, 5, 5) attention mask for target verification
            tree_position_ids: (5,) position IDs for verification
        """
        device = hidden_states_context.device

        # ─── Single forward pass ───
        mask_logits, _ = self.forward_single_pass(
            hidden_states_context, input_ids_context, last_hidden_for_mask
        )
        # mask_logits: (1, 4, draft_vocab_size)
        # Order: [M_1a, M_1b, M_2a, M_2b]

        # ─── Extract Top-1 per slot ───
        pred_draft_ids = mask_logits[0].argmax(dim=-1)  # (4,)

        # Map to target vocab
        tok_m1a = self._map_draft_to_target_token(pred_draft_ids[M_1A])
        tok_m1b = self._map_draft_to_target_token(pred_draft_ids[M_1B])
        tok_m2a = self._map_draft_to_target_token(pred_draft_ids[M_2A])
        tok_m2b = self._map_draft_to_target_token(pred_draft_ids[M_2B])

        # ─── Assemble draft_tokens ───
        # Layout: [sample_token, m1a, m1b, m2a, m2b]
        # Index:       0          1    2    3    4
        draft_tokens = torch.stack([
            sample_token.view(-1)[0],
            tok_m1a, tok_m1b, tok_m2a, tok_m2b
        ]).unsqueeze(0)  # (1, 5)

        # ─── Build tree_mask for target model verification ───
        # 5x5 mask:
        #           sample  m1a  m1b  m2a  m2b
        # sample  [   1     0    0    0    0  ]
        # m1a     [   1     1    0    0    0  ]
        # m1b     [   1     0    1    0    0  ]
        # m2a     [   1     1    0    1    0  ]   ← sees sample + m1a + self
        # m2b     [   1     0    1    0    1  ]   ← sees sample + m1b + self
        tree_mask = torch.zeros(5, 5, dtype=torch.float32, device=device)
        # Self-attention
        tree_mask[0, 0] = 1.0  # sample sees self
        tree_mask[1, 1] = 1.0  # m1a sees self
        tree_mask[2, 2] = 1.0  # m1b sees self
        tree_mask[3, 3] = 1.0  # m2a sees self
        tree_mask[4, 4] = 1.0  # m2b sees self
        # Everyone sees sample (root)
        tree_mask[1, 0] = 1.0  # m1a → sample
        tree_mask[2, 0] = 1.0  # m1b → sample
        tree_mask[3, 0] = 1.0  # m2a → sample
        tree_mask[4, 0] = 1.0  # m2b → sample
        # Branch A: m2a sees m1a
        tree_mask[3, 1] = 1.0
        # Branch B: m2b sees m1b
        tree_mask[4, 2] = 1.0

        tree_mask = tree_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 5, 5)

        # ─── Build tree_position_ids ───
        # sample=0, m1a=1, m1b=1, m2a=2, m2b=2
        tree_position_ids = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long, device=device)

        # ─── Build retrieve_indices ───
        # Branch A path through draft_tokens: [0(sample), 1(m1a), 3(m2a)]
        # Branch B path through draft_tokens: [0(sample), 2(m1b), 4(m2b)]
        retrieve_indices = torch.tensor(
            [[0, 1, 3],   # Branch A
             [0, 2, 4]],  # Branch B
            dtype=torch.long, device=device
        )

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def load_tree_bita_adapter(
    eagle3_config_path: str,
    eagle3_weights_path: str,
    num_prompt_tokens: int = 8,
    top_k: int = 4,
    device: str = "cuda",
) -> TreeBiTAAdapter:
    """Load a TreeBiTAAdapter with pretrained EAGLE-3 weights."""
    config = EConfig.from_pretrained(eagle3_config_path)
    adapter = TreeBiTAAdapter(
        eagle3_config=config,
        num_prompt_tokens=num_prompt_tokens,
        top_k=top_k,
    )

    if eagle3_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(eagle3_weights_path)
    else:
        state_dict = torch.load(eagle3_weights_path, map_location="cpu")

    adapter.load_eagle3_weights(state_dict)
    adapter = adapter.to(device)

    print(f"[TreeBiTAAdapter] Total params: {adapter.count_total_params():,}")
    print(f"[TreeBiTAAdapter] Trainable params: {adapter.count_trainable_params():,}")
    return adapter


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p, n = 4, 8
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    total = p + n + 4
    print(f"Mask shape: {mask.shape} (expected (1,1,{total},{total}))")
    m = p + n  # start of [M] block

    # Real tokens isolated
    assert (mask[0,0, p:p+n, :p] == float('-inf')).all(), "Real sees [P]!"
    assert (mask[0,0, p:p+n, m:] == float('-inf')).all(), "Real sees [M]!"
    print("✓ Real tokens isolated from [P] and [M]")

    # [M] sees [P] + Real
    for mi in range(4):
        assert (mask[0,0, m+mi, :p] == 0).all(), f"M{mi} can't see [P]"
        assert (mask[0,0, m+mi, p:p+n] == 0).all(), f"M{mi} can't see Real"
    print("✓ All [M] see [P] and Real")

    # Siblings blocked
    assert mask[0,0, m+M_1A, m+M_1B] == float('-inf'), "M_1a sees M_1b!"
    assert mask[0,0, m+M_1B, m+M_1A] == float('-inf'), "M_1b sees M_1a!"
    assert mask[0,0, m+M_2A, m+M_2B] == float('-inf'), "M_2a sees M_2b!"
    assert mask[0,0, m+M_2B, m+M_2A] == float('-inf'), "M_2b sees M_2a!"
    print("✓ Siblings correctly blocked (M_1a✗M_1b, M_2a✗M_2b)")

    # Cross-branch blocked
    assert mask[0,0, m+M_2A, m+M_1B] == float('-inf'), "M_2a sees M_1b!"
    assert mask[0,0, m+M_2B, m+M_1A] == float('-inf'), "M_2b sees M_1a!"
    print("✓ Cross-branch blocked (M_2a✗M_1b, M_2b✗M_1a)")

    # Branch parent visible
    assert mask[0,0, m+M_2A, m+M_1A] == 0, "M_2a can't see M_1a!"
    assert mask[0,0, m+M_2B, m+M_1B] == 0, "M_2b can't see M_1b!"
    print("✓ Branch parents visible (M_2a→M_1a, M_2b→M_1b)")

    # Parent doesn't see child
    assert mask[0,0, m+M_1A, m+M_2A] == float('-inf'), "M_1a sees M_2a!"
    assert mask[0,0, m+M_1B, m+M_2B] == float('-inf'), "M_1b sees M_2b!"
    print("✓ Parents don't see children")

    # Position IDs
    pos = build_2x2_position_ids(p, n, last_real_pos=n-1)
    assert pos[0, p+n+M_1A] == n, "M_1a wrong pos"
    assert pos[0, p+n+M_1B] == n, "M_1b wrong pos"
    assert pos[0, p+n+M_2A] == n+1, "M_2a wrong pos"
    assert pos[0, p+n+M_2B] == n+1, "M_2b wrong pos"
    print("✓ Position IDs correct (depth 1 → t+1, depth 2 → t+2)")

    print("\n═══ ALL 2x2 MINI-TREE TESTS PASSED ═══")
