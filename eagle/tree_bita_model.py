# coding=utf-8
"""
tree_bita_model.py
==================
BiTA-adapted EAGLE-3 draft model for single-pass tree drafting.

Supports two topologies (selectable via `topology` argument):

1. **"2x2"** (Mini-Tree) — 2 independent branches of depth 2:
       Root → M_1a → M_2a  (Branch A)
       Root → M_1b → M_2b  (Branch B)

2. **"serial"** (Chain) — 1 sequential chain of depth 4:
       Root → M_1 → M_2 → M_3 → M_4

Both share the same frozen EAGLE-3 backbone with 4 learnable [M] embeddings.
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
# Topology constants
# ═══════════════════════════════════════════════════════════════════════════════

NUM_MASK_TOKENS = 4  # both topologies use exactly 4 mask tokens

# ── 2x2 Mini-Tree topology ──
M_1A, M_1B, M_2A, M_2B = 0, 1, 2, 3

MINI_TREE_PARENT_MAP = {
    M_1A: -1, M_1B: -1,
    M_2A: M_1A, M_2B: M_1B,
}
MINI_TREE_DEPTH_MAP = {
    M_1A: 1, M_1B: 1, M_2A: 2, M_2B: 2,
}
BRANCH_A = [M_1A, M_2A]
BRANCH_B = [M_1B, M_2B]

# ── Serial (chain) topology ──
S_1, S_2, S_3, S_4 = 0, 1, 2, 3

SERIAL_PARENT_MAP = {
    S_1: -1,   # M_1 has no [M] parent
    S_2: S_1,  # M_2's parent is M_1
    S_3: S_2,  # M_3's parent is M_2
    S_4: S_3,  # M_4's parent is M_3
}
SERIAL_DEPTH_MAP = {
    S_1: 1, S_2: 2, S_3: 3, S_4: 4,
}

VALID_TOPOLOGIES = {"2x2", "serial"}


# ═══════════════════════════════════════════════════════════════════════════════
# Mask builders
# ═══════════════════════════════════════════════════════════════════════════════

def _build_base_mask(
    num_prompt: int,
    num_real: int,
    num_mask: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Build the shared mask skeleton: [P] bidir, Real causal, [M] sees [P]+Real+self."""
    p, n, k = num_prompt, num_real, num_mask
    total = p + n + k
    mask = torch.full((total, total), float("-inf"), dtype=dtype, device=device)

    # [P]: bidirectional
    mask[:p, :p] = 0.0

    # Real: causal
    for i in range(n):
        mask[p + i, p: p + i + 1] = 0.0

    # [M]: sees [P] + all Real + self
    m_start = p + n
    for mi in range(k):
        mi_abs = m_start + mi
        mask[mi_abs, :p] = 0.0          # sees [P]
        mask[mi_abs, p: p + n] = 0.0    # sees all Real
        mask[mi_abs, mi_abs] = 0.0      # sees self

    return mask, m_start


def build_2x2_tree_attention_mask(
    num_prompt: int,
    num_real: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    2x2 Mini-Tree mask. M_2a→M_1a, M_2b→M_1b. All other [M]↔[M] blocked.
    Returns: (1, 1, total, total)
    """
    mask, m = _build_base_mask(num_prompt, num_real, NUM_MASK_TOKENS, dtype, device)
    mask[m + M_2A, m + M_1A] = 0.0  # Branch A
    mask[m + M_2B, m + M_1B] = 0.0  # Branch B
    return mask.unsqueeze(0).unsqueeze(0)


def build_serial_attention_mask(
    num_prompt: int,
    num_real: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Serial (chain) mask. M_i sees all M_j where j < i (causal chain).
        M_1: sees [P], Real
        M_2: sees [P], Real, M_1
        M_3: sees [P], Real, M_1, M_2
        M_4: sees [P], Real, M_1, M_2, M_3
    Returns: (1, 1, total, total)
    """
    mask, m = _build_base_mask(num_prompt, num_real, NUM_MASK_TOKENS, dtype, device)
    # Causal chain among [M]
    for i in range(NUM_MASK_TOKENS):
        for j in range(i):
            mask[m + i, m + j] = 0.0  # M_i sees M_j (j < i)
    return mask.unsqueeze(0).unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Position ID builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_2x2_position_ids(
    num_prompt: int, num_real: int, last_real_pos: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Position IDs. M_1a,M_1b→t+1; M_2a,M_2b→t+2."""
    p, n = num_prompt, num_real
    pos = torch.zeros(p + n + NUM_MASK_TOKENS, dtype=torch.long, device=device)
    for i in range(p):
        pos[i] = i
    rs = max(0, last_real_pos - n + 1)
    for i in range(n):
        pos[p + i] = rs + i
    m = p + n
    pos[m + M_1A] = last_real_pos + 1
    pos[m + M_1B] = last_real_pos + 1
    pos[m + M_2A] = last_real_pos + 2
    pos[m + M_2B] = last_real_pos + 2
    return pos.unsqueeze(0)


def build_serial_position_ids(
    num_prompt: int, num_real: int, last_real_pos: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Position IDs. M_1→t+1, M_2→t+2, M_3→t+3, M_4→t+4."""
    p, n = num_prompt, num_real
    pos = torch.zeros(p + n + NUM_MASK_TOKENS, dtype=torch.long, device=device)
    for i in range(p):
        pos[i] = i
    rs = max(0, last_real_pos - n + 1)
    for i in range(n):
        pos[p + i] = rs + i
    m = p + n
    for si in range(NUM_MASK_TOKENS):
        pos[m + si] = last_real_pos + si + 1  # t+1, t+2, t+3, t+4
    return pos.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Topology dispatcher
# ═══════════════════════════════════════════════════════════════════════════════

def get_topology_config(topology: str):
    """Return (parent_map, depth_map, mask_builder, pos_builder) for a topology."""
    if topology == "2x2":
        return (MINI_TREE_PARENT_MAP, MINI_TREE_DEPTH_MAP,
                build_2x2_tree_attention_mask, build_2x2_position_ids)
    elif topology == "serial":
        return (SERIAL_PARENT_MAP, SERIAL_DEPTH_MAP,
                build_serial_attention_mask, build_serial_position_ids)
    else:
        raise ValueError(f"Unknown topology '{topology}'. Choose from: {VALID_TOPOLOGIES}")


# ═══════════════════════════════════════════════════════════════════════════════
# TreeBiTAAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TreeBiTAAdapter(nn.Module):
    """
    Wraps a frozen EAGLE-3 draft model backbone with BiTA-style
    learnable [P] prompt and [M] mask embeddings for single-pass drafting.

    Supports two topologies:
      - "2x2": Branch A (M_1a→M_2a), Branch B (M_1b→M_2b)
      - "serial": M_1→M_2→M_3→M_4 (causal chain)

    Selected via the `topology` constructor argument.
    """

    def __init__(
        self,
        eagle3_config: EConfig,
        topology: str = "2x2",
        num_prompt_tokens: int = 8,
        top_k: int = 4,
        total_tokens: int = 63,
        depth: int = 2,
        threshold: float = 1.0,
    ):
        super().__init__()
        assert topology in VALID_TOPOLOGIES, \
            f"topology must be one of {VALID_TOPOLOGIES}, got '{topology}'"

        self.config = eagle3_config
        self.hidden_size = eagle3_config.hidden_size
        self.topology = topology

        # Topology-specific config
        parent_map, depth_map, mask_fn, pos_fn = get_topology_config(topology)
        self.parent_map = parent_map
        self.depth_map = depth_map
        self._build_mask = mask_fn
        self._build_pos = pos_fn

        self.num_mask_slots = NUM_MASK_TOKENS  # always 4
        self.num_prompt_tokens = num_prompt_tokens
        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth_val = depth
        self.threshold = math.log(threshold) if threshold > 0 else 0.0

        # ─── Frozen EAGLE-3 backbone ───
        self.embed_tokens = nn.Embedding(
            eagle3_config.vocab_size, eagle3_config.hidden_size,
            eagle3_config.pad_token_id
        )
        self.lm_head = nn.Linear(
            eagle3_config.hidden_size, eagle3_config.draft_vocab_size, bias=False
        )
        self.norm = LlamaRMSNorm(eagle3_config.hidden_size, eps=eagle3_config.rms_norm_eps)

        from model.cnets import LlamaDecoderLayeremb
        self.midlayer = LlamaDecoderLayeremb(eagle3_config)

        if hasattr(eagle3_config, "target_hidden_size"):
            fc_in = eagle3_config.target_hidden_size * 3
        else:
            fc_in = eagle3_config.hidden_size * 3
        self.fc = nn.Linear(fc_in, self.hidden_size, bias=False)

        d2t = torch.zeros(eagle3_config.draft_vocab_size, dtype=torch.long)
        t2d = torch.zeros(eagle3_config.vocab_size, dtype=torch.bool)
        self.register_buffer("d2t", d2t)
        self.register_buffer("t2d", t2d)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # ─── Learnable BiTA embeddings (ONLY trainable) ───
        self.prompt_embeddings = nn.Embedding(num_prompt_tokens, self.hidden_size)
        self.prompt_hidden = nn.Embedding(num_prompt_tokens, self.hidden_size)
        self.mask_embeddings = nn.Embedding(NUM_MASK_TOKENS, self.hidden_size)
        self.mask_hidden = nn.Embedding(NUM_MASK_TOKENS, self.hidden_size)

        nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
        nn.init.normal_(self.prompt_hidden.weight, std=0.02)
        nn.init.normal_(self.mask_embeddings.weight, std=0.02)
        nn.init.normal_(self.mask_hidden.weight, std=0.02)

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            param.requires_grad = any(x in name for x in [
                "prompt_embeddings", "prompt_hidden",
                "mask_embeddings", "mask_hidden"
            ])

    def load_eagle3_weights(self, state_dict: dict):
        backbone_keys = {
            k: v for k, v in state_dict.items()
            if not any(x in k for x in ["prompt_", "mask_embeddings", "mask_hidden"])
        }
        missing, unexpected = self.load_state_dict(backbone_keys, strict=False)
        print(f"[TreeBiTAAdapter({self.topology})] Loaded backbone. "
              f"Missing (BiTA): {len(missing)}, Unexpected: {len(unexpected)}")
        self.freeze_backbone()

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ─── Core forward ───

    def forward_single_pass(
        self,
        hidden_states_context: torch.Tensor,
        input_ids_context: torch.Tensor,
        last_hidden_for_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-pass forward. Works with BOTH topologies — the mask/position
        builders are dispatched automatically based on self.topology.

        Returns:
            mask_logits: (1, 4, draft_vocab_size)
            all_hidden:  (1, total_len, hidden_size)
        """
        device = hidden_states_context.device
        dtype = hidden_states_context.dtype
        bsz = hidden_states_context.shape[0]
        n_ctx = input_ids_context.shape[1]
        n_prompt = self.num_prompt_tokens
        n_mask = NUM_MASK_TOKENS

        # Input embeddings
        prompt_idx = torch.arange(n_prompt, device=device)
        prompt_emb = self.prompt_embeddings(prompt_idx).unsqueeze(0).to(dtype)
        with torch.no_grad():
            real_emb = self.embed_tokens(input_ids_context).to(dtype)
        mask_idx = torch.arange(n_mask, device=device)
        mask_emb = self.mask_embeddings(mask_idx).unsqueeze(0).to(dtype)
        input_emb = torch.cat([prompt_emb, real_emb, mask_emb], dim=1)

        # Hidden states
        prompt_hid = self.prompt_hidden(prompt_idx).unsqueeze(0).to(dtype)
        real_hid = hidden_states_context
        mask_hid = (last_hidden_for_mask.expand(bsz, n_mask, -1).to(dtype)
                    + self.mask_hidden(mask_idx).unsqueeze(0).to(dtype))
        hidden_states = torch.cat([prompt_hid, real_hid, mask_hid], dim=1)

        # Attention mask (topology-specific)
        tree_attn_mask = self._build_mask(
            num_prompt=n_prompt, num_real=n_ctx,
            dtype=torch.float32, device=device,
        )

        # Position IDs (topology-specific)
        if position_ids is None:
            position_ids = self._build_pos(
                num_prompt=n_prompt, num_real=n_ctx,
                last_real_pos=n_ctx - 1, device=device,
            )

        # Forward
        layer_outputs = self.midlayer(
            input_emb=input_emb, hidden_states=hidden_states,
            attention_mask=tree_attn_mask, position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=False,
        )
        all_hidden = layer_outputs[0]

        mask_out = all_hidden[:, n_prompt + n_ctx:, :]
        mask_logits = self.lm_head(self.norm(mask_out))

        return mask_logits, all_hidden

    # ─── Draft helpers ───

    def _map_token(self, draft_id: torch.Tensor) -> torch.Tensor:
        if self.config.vocab_size != self.config.draft_vocab_size:
            return self.d2t[draft_id]
        return draft_id

    # ─── 2x2 drafting ───

    @torch.no_grad()
    def _draft_2x2(self, mask_logits, sample_token, device):
        """
        2x2 tree → 5 tokens, 2 branches.
        draft = [sample, m1a, m1b, m2a, m2b]
        retrieve = [[0,1,3], [0,2,4]]
        """
        pred = mask_logits[0].argmax(dim=-1)
        toks = [self._map_token(pred[i]) for i in range(4)]

        draft_tokens = torch.stack([
            sample_token.view(-1)[0], toks[0], toks[1], toks[2], toks[3]
        ]).unsqueeze(0)

        # Verification mask
        tm = torch.zeros(5, 5, dtype=torch.float32, device=device)
        for i in range(5):
            tm[i, i] = 1.0
        for i in [1, 2, 3, 4]:
            tm[i, 0] = 1.0
        tm[3, 1] = 1.0  # m2a → m1a
        tm[4, 2] = 1.0  # m2b → m1b

        tree_pos = torch.tensor([0, 1, 1, 2, 2], dtype=torch.long, device=device)
        retrieve = torch.tensor([[0, 1, 3], [0, 2, 4]], dtype=torch.long, device=device)

        return draft_tokens, retrieve, tm.unsqueeze(0).unsqueeze(0), tree_pos

    # ─── Serial drafting ───

    @torch.no_grad()
    def _draft_serial(self, mask_logits, sample_token, device):
        """
        Serial chain → 5 tokens, 1 path.
        draft = [sample, m1, m2, m3, m4]
        retrieve = [[0, 1, 2, 3, 4]]
        """
        pred = mask_logits[0].argmax(dim=-1)
        toks = [self._map_token(pred[i]) for i in range(4)]

        draft_tokens = torch.stack([
            sample_token.view(-1)[0], toks[0], toks[1], toks[2], toks[3]
        ]).unsqueeze(0)

        # Verification mask: causal chain
        #        sample  m1   m2   m3   m4
        # sample [  1     0    0    0    0  ]
        # m1     [  1     1    0    0    0  ]
        # m2     [  1     1    1    0    0  ]
        # m3     [  1     1    1    1    0  ]
        # m4     [  1     1    1    1    1  ]
        tm = torch.zeros(5, 5, dtype=torch.float32, device=device)
        for i in range(5):
            for j in range(i + 1):
                tm[i, j] = 1.0

        tree_pos = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device)
        retrieve = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long, device=device)

        return draft_tokens, retrieve, tm.unsqueeze(0).unsqueeze(0), tree_pos

    # ─── Unified drafting entry point ───

    @torch.no_grad()
    def single_pass_draft(
        self,
        hidden_states_context: torch.Tensor,
        input_ids_context: torch.Tensor,
        last_hidden_for_mask: torch.Tensor,
        sample_token: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single forward pass → draft tree (topology-aware).

        Returns:
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids
        """
        device = hidden_states_context.device

        mask_logits, _ = self.forward_single_pass(
            hidden_states_context, input_ids_context, last_hidden_for_mask
        )

        if self.topology == "2x2":
            return self._draft_2x2(mask_logits, sample_token, device)
        else:  # serial
            return self._draft_serial(mask_logits, sample_token, device)


# ═══════════════════════════════════════════════════════════════════════════════
# Loading utility
# ═══════════════════════════════════════════════════════════════════════════════

def load_tree_bita_adapter(
    eagle3_config_path: str,
    eagle3_weights_path: str,
    topology: str = "2x2",
    num_prompt_tokens: int = 8,
    top_k: int = 4,
    device: str = "cuda",
) -> TreeBiTAAdapter:
    config = EConfig.from_pretrained(eagle3_config_path)
    adapter = TreeBiTAAdapter(
        eagle3_config=config, topology=topology,
        num_prompt_tokens=num_prompt_tokens, top_k=top_k,
    )
    if eagle3_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(eagle3_weights_path)
    else:
        state_dict = torch.load(eagle3_weights_path, map_location="cpu")
    adapter.load_eagle3_weights(state_dict)
    adapter = adapter.to(device)
    print(f"[TreeBiTAAdapter] Total: {adapter.count_total_params():,}, "
          f"Trainable: {adapter.count_trainable_params():,}")
    return adapter


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test (both topologies)
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    p, n = 4, 8

    print("═══ 2x2 Mini-Tree Tests ═══")
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    m = p + n
    assert (mask[0,0, p:m, :p] == float('-inf')).all()
    assert (mask[0,0, p:m, m:] == float('-inf')).all()
    assert mask[0,0, m+M_2A, m+M_1A] == 0 and mask[0,0, m+M_2A, m+M_1B] == float('-inf')
    assert mask[0,0, m+M_2B, m+M_1B] == 0 and mask[0,0, m+M_2B, m+M_1A] == float('-inf')
    assert mask[0,0, m+M_1A, m+M_1B] == float('-inf')
    print("✓ 2x2 mask correct")

    print("\n═══ Serial Chain Tests ═══")
    mask_s = build_serial_attention_mask(num_prompt=p, num_real=n)
    # Real isolated
    assert (mask_s[0,0, p:m, :p] == float('-inf')).all()
    assert (mask_s[0,0, p:m, m:] == float('-inf')).all()
    print("✓ Real isolated")
    # M_2 sees M_1
    assert mask_s[0,0, m+S_2, m+S_1] == 0
    # M_3 sees M_1 and M_2
    assert mask_s[0,0, m+S_3, m+S_1] == 0
    assert mask_s[0,0, m+S_3, m+S_2] == 0
    # M_4 sees M_1, M_2, M_3
    assert mask_s[0,0, m+S_4, m+S_1] == 0
    assert mask_s[0,0, m+S_4, m+S_2] == 0
    assert mask_s[0,0, m+S_4, m+S_3] == 0
    print("✓ Serial causal chain correct")
    # M_1 does NOT see M_2, M_3, M_4
    assert mask_s[0,0, m+S_1, m+S_2] == float('-inf')
    assert mask_s[0,0, m+S_1, m+S_3] == float('-inf')
    assert mask_s[0,0, m+S_1, m+S_4] == float('-inf')
    # M_2 does NOT see M_3, M_4
    assert mask_s[0,0, m+S_2, m+S_3] == float('-inf')
    assert mask_s[0,0, m+S_2, m+S_4] == float('-inf')
    print("✓ Serial no-future-look correct")

    # Position IDs
    pos_s = build_serial_position_ids(p, n, last_real_pos=n-1)
    for si in range(4):
        assert pos_s[0, m+si] == n + si, f"Serial M_{si+1} wrong pos"
    print("✓ Serial positions: t+1, t+2, t+3, t+4")

    print("\n═══ ALL TESTS PASSED ═══")
