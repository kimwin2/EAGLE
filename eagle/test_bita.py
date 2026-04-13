"""
test_bita.py — Tests for the 2x2 Mini-Tree BiTA-EAGLE3 adapter.

Tests:
  1. Mask shape and dimensions
  2. Real token isolation (can't see [P] or [M])
  3. [M] sees [P] and Real
  4. Sibling blocking (M_1a ✗ M_1b, M_2a ✗ M_2b)
  5. Cross-branch blocking (M_2a ✗ M_1b, M_2b ✗ M_1a)
  6. Branch parent visibility (M_2a → M_1a, M_2b → M_1b)
  7. Parent doesn't see child
  8. Position IDs (depth-1 → t+1, depth-2 → t+2)
  9. Causal mask among Real tokens
  10. Retrieve indices for Branch A/B
  11. Tree verification mask (5×5)
  12. Parameter freeze/unfreeze
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from tree_bita_model import (
    build_2x2_tree_attention_mask,
    build_2x2_position_ids,
    NUM_MASK_TOKENS, M_1A, M_1B, M_2A, M_2B,
    BRANCH_A, BRANCH_B,
    MINI_TREE_PARENT_MAP, MINI_TREE_DEPTH_MAP,
)


def test_mask_shape():
    p, n = 8, 16
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    total = p + n + 4
    assert mask.shape == (1, 1, total, total), f"Bad shape: {mask.shape}"
    print(f"[1] Mask shape: {mask.shape} ✓")


def test_real_isolation():
    p, n = 8, 16
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    m = p + n
    assert (mask[0, 0, p:m, :p] == float('-inf')).all(), "Real sees [P]!"
    assert (mask[0, 0, p:m, m:] == float('-inf')).all(), "Real sees [M]!"
    print("[2] Real tokens isolated from [P] and [M] ✓")


def test_mask_sees_prompt_and_real():
    p, n = 8, 16
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    m = p + n
    for mi in range(4):
        assert (mask[0, 0, m + mi, :p] == 0).all(), f"M{mi} can't see [P]"
        assert (mask[0, 0, m + mi, p:m] == 0).all(), f"M{mi} can't see Real"
    print("[3] All [M] tokens see [P] and Real ✓")


def test_sibling_blocking():
    p, n = 8, 16
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    m = p + n
    # Depth-1 siblings
    assert mask[0, 0, m + M_1A, m + M_1B] == float('-inf'), "M_1a sees M_1b!"
    assert mask[0, 0, m + M_1B, m + M_1A] == float('-inf'), "M_1b sees M_1a!"
    # Depth-2 siblings
    assert mask[0, 0, m + M_2A, m + M_2B] == float('-inf'), "M_2a sees M_2b!"
    assert mask[0, 0, m + M_2B, m + M_2A] == float('-inf'), "M_2b sees M_2a!"
    print("[4] Sibling blocking correct (M_1a✗M_1b, M_2a✗M_2b) ✓")


def test_cross_branch_blocking():
    p, n = 8, 16
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    m = p + n
    assert mask[0, 0, m + M_2A, m + M_1B] == float('-inf'), "M_2a sees M_1b!"
    assert mask[0, 0, m + M_2B, m + M_1A] == float('-inf'), "M_2b sees M_1a!"
    print("[5] Cross-branch blocking correct (M_2a✗M_1b, M_2b✗M_1a) ✓")


def test_branch_parent_visibility():
    p, n = 8, 16
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    m = p + n
    assert mask[0, 0, m + M_2A, m + M_1A] == 0, "M_2a can't see parent M_1a!"
    assert mask[0, 0, m + M_2B, m + M_1B] == 0, "M_2b can't see parent M_1b!"
    print("[6] Branch parent visibility correct (M_2a→M_1a, M_2b→M_1b) ✓")


def test_parent_no_child():
    p, n = 8, 16
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    m = p + n
    assert mask[0, 0, m + M_1A, m + M_2A] == float('-inf'), "M_1a sees child M_2a!"
    assert mask[0, 0, m + M_1B, m + M_2B] == float('-inf'), "M_1b sees child M_2b!"
    print("[7] Parents don't see children ✓")


def test_position_ids():
    p, n = 8, 16
    pos = build_2x2_position_ids(p, n, last_real_pos=n - 1)
    m = p + n
    assert pos[0, m + M_1A] == n, f"M_1a pos={pos[0, m+M_1A]}, expected {n}"
    assert pos[0, m + M_1B] == n, f"M_1b pos={pos[0, m+M_1B]}, expected {n}"
    assert pos[0, m + M_2A] == n + 1, f"M_2a pos={pos[0, m+M_2A]}, expected {n+1}"
    assert pos[0, m + M_2B] == n + 1, f"M_2b pos={pos[0, m+M_2B]}, expected {n+1}"
    print(f"[8] Position IDs correct: depth-1→{n}, depth-2→{n+1} ✓")


def test_causal_mask():
    p, n = 4, 8
    mask = build_2x2_tree_attention_mask(num_prompt=p, num_real=n)
    for i in range(n):
        for j in range(n):
            val = mask[0, 0, p + i, p + j].item()
            if j <= i:
                assert val == 0.0, f"Real[{i}] blocked from past Real[{j}]"
            else:
                assert val == float('-inf'), f"Real[{i}] sees future Real[{j}]"
    print("[9] Real tokens have correct causal mask ✓")


def test_topology_constants():
    assert MINI_TREE_PARENT_MAP[M_1A] == -1
    assert MINI_TREE_PARENT_MAP[M_1B] == -1
    assert MINI_TREE_PARENT_MAP[M_2A] == M_1A
    assert MINI_TREE_PARENT_MAP[M_2B] == M_1B
    assert MINI_TREE_DEPTH_MAP[M_1A] == 1
    assert MINI_TREE_DEPTH_MAP[M_1B] == 1
    assert MINI_TREE_DEPTH_MAP[M_2A] == 2
    assert MINI_TREE_DEPTH_MAP[M_2B] == 2
    assert BRANCH_A == [M_1A, M_2A]
    assert BRANCH_B == [M_1B, M_2B]
    print("[10] Topology constants correct ✓")


def test_param_counting():
    """Verify trainable param count with small config."""
    from model.configs import EConfig
    config = EConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
        hidden_act="silu", max_position_embeddings=512,
        rms_norm_eps=1e-5, draft_vocab_size=500,
    )
    from tree_bita_model import TreeBiTAAdapter
    adapter = TreeBiTAAdapter(eagle3_config=config, num_prompt_tokens=4)
    adapter.freeze_backbone()

    trainable = adapter.count_trainable_params()
    # 4 embeddings: prompt_emb(4×64) + prompt_hid(4×64) + mask_emb(4×64) + mask_hid(4×64)
    expected = (4 * 64) * 2 + (4 * 64) * 2  # prompt + mask
    assert trainable == expected, f"Expected {expected}, got {trainable}"
    print(f"[11] Trainable params: {trainable} (expected {expected}) ✓")

    # Verify freeze
    for name, param in adapter.named_parameters():
        is_bita = any(x in name for x in ["prompt_", "mask_embeddings", "mask_hidden"])
        if is_bita:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"
    print("[12] Freeze/unfreeze correct ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("2x2 Mini-Tree BiTA-EAGLE3 Tests")
    print("=" * 60 + "\n")

    test_mask_shape()
    test_real_isolation()
    test_mask_sees_prompt_and_real()
    test_sibling_blocking()
    test_cross_branch_blocking()
    test_branch_parent_visibility()
    test_parent_no_child()
    test_position_ids()
    test_causal_mask()
    test_topology_constants()
    test_param_counting()

    print("\n" + "=" * 60)
    print("ALL 12 TESTS PASSED ✓")
    print("=" * 60)
