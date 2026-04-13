"""
test_bita.py — Tests for both 2x2 Mini-Tree and Serial Chain topologies.

Tests cover:
  - Mask shape and dimensions
  - Real token isolation (can't see [P] or [M])
  - [M] sees [P] and Real
  - Topology-specific [M]↔[M] visibility
  - Position IDs
  - Causal mask among Real tokens
  - Parameter freeze/unfreeze
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tree_bita_model import (
    build_2x2_tree_attention_mask, build_serial_attention_mask,
    build_2x2_position_ids, build_serial_position_ids,
    NUM_MASK_TOKENS, M_1A, M_1B, M_2A, M_2B,
    S_1, S_2, S_3, S_4,
    BRANCH_A, BRANCH_B,
    MINI_TREE_PARENT_MAP, MINI_TREE_DEPTH_MAP,
    SERIAL_PARENT_MAP, SERIAL_DEPTH_MAP,
)

P, N = 8, 16  # shared test params
INF = float('-inf')


# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════

def _check_real_isolation(mask, p, n, label):
    m = p + n
    assert (mask[0,0, p:m, :p] == INF).all(), f"[{label}] Real sees [P]!"
    assert (mask[0,0, p:m, m:] == INF).all(), f"[{label}] Real sees [M]!"

def _check_m_sees_p_and_real(mask, p, n, label):
    m = p + n
    for mi in range(NUM_MASK_TOKENS):
        assert (mask[0,0, m+mi, :p] == 0).all(), f"[{label}] M{mi} can't see [P]"
        assert (mask[0,0, m+mi, p:p+n] == 0).all(), f"[{label}] M{mi} can't see Real"

def _check_real_causal(mask, p, n, label):
    for i in range(n):
        for j in range(n):
            val = mask[0,0, p+i, p+j].item()
            if j <= i:
                assert val == 0.0, f"[{label}] Real[{i}] blocked from past Real[{j}]"
            else:
                assert val == INF, f"[{label}] Real[{i}] sees future Real[{j}]"


# ═══════════════════════════════════════════════════════════════════
# 2x2 Mini-Tree Tests
# ═══════════════════════════════════════════════════════════════════

def test_2x2_mask_shape():
    mask = build_2x2_tree_attention_mask(P, N)
    total = P + N + 4
    assert mask.shape == (1, 1, total, total)
    print("[2x2-1] Mask shape ✓")

def test_2x2_real_isolation():
    mask = build_2x2_tree_attention_mask(P, N)
    _check_real_isolation(mask, P, N, "2x2")
    print("[2x2-2] Real isolation ✓")

def test_2x2_m_sees_context():
    mask = build_2x2_tree_attention_mask(P, N)
    _check_m_sees_p_and_real(mask, P, N, "2x2")
    print("[2x2-3] [M] sees [P]+Real ✓")

def test_2x2_siblings_blocked():
    mask = build_2x2_tree_attention_mask(P, N)
    m = P + N
    assert mask[0,0, m+M_1A, m+M_1B] == INF
    assert mask[0,0, m+M_1B, m+M_1A] == INF
    assert mask[0,0, m+M_2A, m+M_2B] == INF
    assert mask[0,0, m+M_2B, m+M_2A] == INF
    print("[2x2-4] Siblings blocked ✓")

def test_2x2_cross_branch_blocked():
    mask = build_2x2_tree_attention_mask(P, N)
    m = P + N
    assert mask[0,0, m+M_2A, m+M_1B] == INF
    assert mask[0,0, m+M_2B, m+M_1A] == INF
    print("[2x2-5] Cross-branch blocked ✓")

def test_2x2_branch_parent():
    mask = build_2x2_tree_attention_mask(P, N)
    m = P + N
    assert mask[0,0, m+M_2A, m+M_1A] == 0
    assert mask[0,0, m+M_2B, m+M_1B] == 0
    print("[2x2-6] Branch parent visible ✓")

def test_2x2_parent_no_child():
    mask = build_2x2_tree_attention_mask(P, N)
    m = P + N
    assert mask[0,0, m+M_1A, m+M_2A] == INF
    assert mask[0,0, m+M_1B, m+M_2B] == INF
    print("[2x2-7] Parent ✗ child ✓")

def test_2x2_positions():
    pos = build_2x2_position_ids(P, N, last_real_pos=N-1)
    m = P + N
    assert pos[0, m+M_1A] == N and pos[0, m+M_1B] == N
    assert pos[0, m+M_2A] == N+1 and pos[0, m+M_2B] == N+1
    print(f"[2x2-8] Positions: depth1→{N}, depth2→{N+1} ✓")

def test_2x2_causal():
    mask = build_2x2_tree_attention_mask(P, N)
    _check_real_causal(mask, P, N, "2x2")
    print("[2x2-9] Real causal mask ✓")


# ═══════════════════════════════════════════════════════════════════
# Serial Chain Tests
# ═══════════════════════════════════════════════════════════════════

def test_serial_mask_shape():
    mask = build_serial_attention_mask(P, N)
    total = P + N + 4
    assert mask.shape == (1, 1, total, total)
    print("[serial-1] Mask shape ✓")

def test_serial_real_isolation():
    mask = build_serial_attention_mask(P, N)
    _check_real_isolation(mask, P, N, "serial")
    print("[serial-2] Real isolation ✓")

def test_serial_m_sees_context():
    mask = build_serial_attention_mask(P, N)
    _check_m_sees_p_and_real(mask, P, N, "serial")
    print("[serial-3] [M] sees [P]+Real ✓")

def test_serial_causal_chain():
    mask = build_serial_attention_mask(P, N)
    m = P + N
    # M_i sees all M_j where j < i
    for i in range(NUM_MASK_TOKENS):
        for j in range(NUM_MASK_TOKENS):
            val = mask[0,0, m+i, m+j].item()
            if j <= i:
                assert val == 0.0, f"M_{i+1} can't see M_{j+1} (should be causal)"
            else:
                assert val == INF, f"M_{i+1} sees future M_{j+1}"
    print("[serial-4] Causal chain correct (M_i sees M_j for j≤i) ✓")

def test_serial_positions():
    pos = build_serial_position_ids(P, N, last_real_pos=N-1)
    m = P + N
    for si in range(4):
        assert pos[0, m+si] == N + si, f"M_{si+1} pos wrong"
    print(f"[serial-5] Positions: t+1, t+2, t+3, t+4 ✓")

def test_serial_causal():
    mask = build_serial_attention_mask(P, N)
    _check_real_causal(mask, P, N, "serial")
    print("[serial-6] Real causal mask ✓")


# ═══════════════════════════════════════════════════════════════════
# Topology adapter tests
# ═══════════════════════════════════════════════════════════════════

def test_adapter_topology_param():
    from model.configs import EConfig
    from tree_bita_model import TreeBiTAAdapter

    config = EConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        hidden_act="silu", max_position_embeddings=512,
        rms_norm_eps=1e-5, draft_vocab_size=500,
    )

    # 2x2
    a1 = TreeBiTAAdapter(config, topology="2x2", num_prompt_tokens=4)
    assert a1.topology == "2x2"
    assert a1.depth_map == MINI_TREE_DEPTH_MAP
    print("[topo-1] 2x2 adapter created ✓")

    # serial
    a2 = TreeBiTAAdapter(config, topology="serial", num_prompt_tokens=4)
    assert a2.topology == "serial"
    assert a2.depth_map == SERIAL_DEPTH_MAP
    print("[topo-2] serial adapter created ✓")

    # freeze check
    a1.freeze_backbone()
    trainable = a1.count_trainable_params()
    expected = (4 * 64) * 2 + (4 * 64) * 2
    assert trainable == expected, f"Expected {expected}, got {trainable}"
    print(f"[topo-3] Trainable params: {trainable} ✓")


def test_topology_constants():
    assert MINI_TREE_DEPTH_MAP == {0:1, 1:1, 2:2, 3:2}
    assert SERIAL_DEPTH_MAP == {0:1, 1:2, 2:3, 3:4}
    assert BRANCH_A == [0, 2]
    assert BRANCH_B == [1, 3]
    print("[topo-4] All topology constants correct ✓")


# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("BiTA-EAGLE3 Tests — 2x2 + Serial Topologies")
    print("=" * 60 + "\n")

    # 2x2
    print("── 2x2 Mini-Tree ──")
    test_2x2_mask_shape()
    test_2x2_real_isolation()
    test_2x2_m_sees_context()
    test_2x2_siblings_blocked()
    test_2x2_cross_branch_blocked()
    test_2x2_branch_parent()
    test_2x2_parent_no_child()
    test_2x2_positions()
    test_2x2_causal()

    # Serial
    print("\n── Serial Chain ──")
    test_serial_mask_shape()
    test_serial_real_isolation()
    test_serial_m_sees_context()
    test_serial_causal_chain()
    test_serial_positions()
    test_serial_causal()

    # Topology adapter
    print("\n── Topology Adapter ──")
    test_adapter_topology_param()
    test_topology_constants()

    print("\n" + "=" * 60)
    print("ALL 19 TESTS PASSED ✓")
    print("=" * 60)
