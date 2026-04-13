# coding=utf-8
"""
inference.py
============
Modified generation pipeline for BiTA-adapted EAGLE-3.

Replaces the sequential K-step drafting loop in EAGLE-3 with a single-pass
BiTA-style tree drafting using learnable [P] and [M] embeddings.

Usage:
    python inference.py \
        --base_model_path /path/to/llama-3-8b-instruct \
        --eagle3_model_path /path/to/eagle3/checkpoint \
        --bita_weights_path /path/to/bita_embeddings.pt \
        --prompt "What is the meaning of life?"
"""

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig

# ─── Local imports ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_bita_model import (
    TreeBiTAAdapter, NUM_MASK_TOKENS, VALID_TOPOLOGIES,
)
from model.configs import EConfig
from model.kv_cache import initialize_past_key_values
from model.utils import (
    prepare_logits_processor,
    tree_decoding,
    evaluate_posterior,
    reset_tree_mode,
)


class BiTAEaModel(nn.Module):
    """
    End-to-end model combining:
    - Frozen target LLM (for prefill + tree verification)
    - BiTA-adapted EAGLE-3 draft model (for single-pass tree drafting)

    This replaces EaModel from ea_model.py with BiTA-style single-pass drafting.
    """

    def __init__(
        self,
        base_model: nn.Module,
        base_model_name_or_path: str,
        bita_adapter: TreeBiTAAdapter,
    ):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast=False)
        self.bita_adapter = bita_adapter
        # For compatibility with existing tree verification
        self.use_eagle3 = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
    ):
        """Forward pass through base model (for prefill and verification)."""
        with torch.inference_mode():
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            hidden_states = outputs[0]

        if output_orig:
            return outputs, orig, hidden_states
        else:
            return outputs, hidden_states

    @torch.no_grad()
    def bita_draft(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        logits_processor=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        2x2 Mini-Tree single-pass drafting.

        Produces two branches of depth 2 in a SINGLE forward pass:
            Branch A: sample_token → Top1(M_1a) → Top1(M_2a)
            Branch B: sample_token → Top1(M_1b) → Top1(M_2b)

        Returns:
            draft_tokens:      (1, 5) = [sample, m1a, m1b, m2a, m2b]
            retrieve_indices:  (2, 3) = [[0,1,3], [0,2,4]] for Branch A/B
            tree_mask:         (1, 1, 5, 5) verification mask
            tree_position_ids: (5,) = [0, 1, 1, 2, 2]
        """
        adapter = self.bita_adapter
        sample_token = input_ids[:, -1:]  # (1, 1)

        # Remove BOS token for EAGLE-3 convention
        context_ids = input_ids[:, 1:]

        # Project hidden states through EAGLE-3's fc
        if hidden_states.shape[-1] != adapter.hidden_size:
            hidden_proj = adapter.fc(hidden_states)
        else:
            hidden_proj = hidden_states

        last_hidden = hidden_proj[:, -1:, :]  # (1, 1, h)

        # ─── Single forward pass → 2x2 mini-tree ───
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = \
            adapter.single_pass_draft(
                hidden_states_context=hidden_proj,
                input_ids_context=context_ids,
                last_hidden_for_mask=last_hidden,
                sample_token=sample_token,
            )

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids

    @torch.no_grad()
    def eagenerate(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        max_new_tokens: int = 512,
        max_length: int = 2048,
        log: bool = False,
        is_llama3: bool = False,
    ):
        """
        Generate text using BiTA-style single-pass speculative decoding.

        This is the main generation entry point, replacing EaModel.eagenerate.
        """
        if is_llama3:
            stop_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding_token = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()

        # ── Initialize KV cache for target model ──
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model, max_length=max_length)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(self)

        # ── Prefill: run target model on full prompt ──
        outputs, orig, hidden_states_prefill = self(
            input_ids, past_key_values=past_key_values, output_orig=True
        )

        # Get first token
        if logits_processor is not None:
            logits = orig[:, -1]
            logits = logits_processor(None, logits)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            token = torch.multinomial(probabilities, 1)
        else:
            token = torch.argmax(orig[:, -1])
            token = token[None, None]

        input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)

        # Get concatenated hidden states for EAGLE-3
        ea_device = self.bita_adapter.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states = torch.cat(outputs["hidden_states"], dim=-1)

        # ── BiTA single-pass drafting ──
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.bita_draft(
            hidden_states, input_ids, logits_processor
        )

        new_token = 0
        effective_max_length = max_length - self.bita_adapter.total_tokens - 10

        # ── Main generation loop ──
        for idx in range(effective_max_length):
            self.base_model.model.tree_mask = tree_mask

            draft_tokens = draft_tokens.to(input_ids.device)

            # ── Target model verification ──
            logits, hidden_state_new, outputs = tree_decoding(
                self,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            draft_tokens = torch.cat((draft_tokens, padding_token), dim=1)
            candidates = draft_tokens[0, retrieve_indices]

            # ── Evaluate candidates ──
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # ── Update state ──
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, \
                new_token, hidden_states, token = self._update_inference_inputs(
                    input_ids, candidates, best_candidate, accept_length,
                    retrieve_indices, logits_processor, new_token,
                    past_key_values_data, current_length_data,
                    hidden_state_new, sample_p, outputs,
                )

            # ── Check stopping conditions ──
            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if self.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > effective_max_length:
                break

        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx

    @torch.no_grad()
    def _update_inference_inputs(
        self,
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values_data_list,
        current_length_data,
        hidden_state_new,
        sample_p,
        outputs,
    ):
        """Update inputs after acceptance/rejection, then do BiTA single-pass drafting."""
        prev_input_len = input_ids.shape[1]

        # Map best candidate indices
        select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
        )

        # Append accepted tokens
        input_ids = torch.cat(
            [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)],
            dim=-1,
        )

        # Update KV cache
        for past_key_values_data in past_key_values_data_list:
            tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
            dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)

        current_length_data.fill_(prev_input_len + tgt.shape[-2])

        # Get accepted hidden states
        retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
        accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]

        # Sample next token
        prob = sample_p
        if logits_processor is not None:
            token = torch.multinomial(prob, 1)
            token = token[None]
        else:
            token = torch.argmax(prob)
            token = token[None, None]

        # Update hidden states (concatenate from target model outputs)
        ea_device = self.bita_adapter.lm_head.weight.device
        if outputs["hidden_states"][0].device != ea_device:
            outputs["hidden_states"] = [x.to(ea_device) for x in outputs["hidden_states"]]
        hidden_states_concat = torch.cat(outputs["hidden_states"], dim=-1)

        # ── BiTA single-pass drafting for next step ──
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids = self.bita_draft(
            accept_hidden_state_new,
            torch.cat((input_ids, token.to(input_ids.device)), dim=1),
            logits_processor,
        )

        new_token += accept_length + 1

        return (
            input_ids, draft_tokens, retrieve_indices, tree_mask,
            tree_position_ids, new_token, None, token,
        )

    @torch.no_grad()
    def ea_generate(self, input_ids, **kwargs):
        """Streaming version of eagenerate."""
        # Delegate to eagenerate for now
        result = self.eagenerate(input_ids, **kwargs)
        yield result


# ═══════════════════════════════════════════════════════════════════════════════
# Model loading utilities
# ═══════════════════════════════════════════════════════════════════════════════

def load_bita_model(
    base_model_path: str,
    eagle3_model_path: str,
    bita_weights_path: Optional[str] = None,
    topology: str = "2x2",
    num_prompt_tokens: int = 8,
    total_token: int = 6,
    depth: int = 2,
    top_k: int = 4,
    threshold: float = 1.0,
    **kwargs,
) -> BiTAEaModel:
    """
    Load the complete BiTA-EAGLE3 model with 2x2 Mini-Tree adapter.

    Args:
        base_model_path: Path to target LLM
        eagle3_model_path: Path to EAGLE-3 checkpoint directory
        bita_weights_path: Path to trained BiTA embeddings (.pt file)
        num_prompt_tokens: Number of [P] prompt tokens
        total_token: Total tree tokens for drafting (5 for 2x2 tree)
        depth: Maximum tree depth (2 for mini-tree)
        top_k: Top-k candidates per position
        threshold: Score threshold
    """
    # ── Load target model ──
    Type = AutoConfig.from_pretrained(base_model_path).architectures[0]
    if Type == 'LlamaForCausalLM':
        from model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
        base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)
    elif Type == 'Qwen2ForCausalLM':
        from model.modeling_qwen2_kv import Qwen2ForCausalLM as KVQwen2ForCausalLM
        base_model = KVQwen2ForCausalLM.from_pretrained(base_model_path, **kwargs)
    else:
        from model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
        base_model = KVLlamaForCausalLM.from_pretrained(base_model_path, **kwargs)

    # ── Load EAGLE-3 config and build 2x2 Mini-Tree adapter ──
    configpath = os.path.join(eagle3_model_path, "config.json")
    if not os.path.exists(configpath):
        from huggingface_hub import hf_hub_download
        configpath = hf_hub_download(eagle3_model_path, "config.json")

    eagle3_config = EConfig.from_pretrained(configpath)

    adapter = TreeBiTAAdapter(
        eagle3_config=eagle3_config,
        topology=topology,
        num_prompt_tokens=num_prompt_tokens,
        top_k=top_k,
        total_tokens=total_token,
        depth=depth,
        threshold=threshold,
    )

    # ── Load EAGLE-3 backbone weights ──
    try:
        load_model_path = os.path.join(eagle3_model_path, "pytorch_model.bin")
        if not os.path.exists(load_model_path):
            from huggingface_hub import hf_hub_download
            load_model_path = hf_hub_download(eagle3_model_path, "pytorch_model.bin")
        eagle3_state = torch.load(load_model_path, map_location=base_model.device)
    except Exception:
        from safetensors.torch import load_file
        load_model_path = os.path.join(eagle3_model_path, "model.safetensors")
        if not os.path.exists(load_model_path):
            from huggingface_hub import hf_hub_download
            load_model_path = hf_hub_download(eagle3_model_path, "model.safetensors")
        eagle3_state = load_file(load_model_path)

    adapter.load_eagle3_weights(eagle3_state)

    # ── Load BiTA embeddings if available ──
    if bita_weights_path and os.path.exists(bita_weights_path):
        bita_state = torch.load(bita_weights_path, map_location="cpu")
        adapter.load_state_dict(bita_state, strict=False)
        print(f"[BiTA] Loaded trained embeddings from {bita_weights_path}")

    # ── Move to device ──
    device = base_model.model.layers[-1].self_attn.q_proj.weight.device
    adapter = adapter.to(base_model.dtype).to(device)

    # ── Create BiTAEaModel ──
    model = BiTAEaModel(base_model, base_model_path, adapter)
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='BiTA-EAGLE3 Inference')
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Path to target LLM')
    parser.add_argument('--eagle3_model_path', type=str, required=True,
                        help='Path to EAGLE-3 checkpoint directory')
    parser.add_argument('--bita_weights_path', type=str, default=None,
                        help='Path to trained BiTA embeddings')
    parser.add_argument('--prompt', type=str, default="What is speculative decoding?",
                        help='Input prompt')
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--num_prompt_tokens', type=int, default=8)
    parser.add_argument('--is_llama3', action='store_true')
    parser.add_argument('--topology', type=str, default='2x2', choices=['2x2', 'serial'],
                        help='Mask topology: "2x2" (mini-tree) or "serial" (causal chain)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run latency benchmark')
    args = parser.parse_args()

    # ── Load model ──
    print("Loading BiTA-EAGLE3 model...")
    model = load_bita_model(
        base_model_path=args.base_model_path,
        eagle3_model_path=args.eagle3_model_path,
        bita_weights_path=args.bita_weights_path,
        topology=args.topology,
        num_prompt_tokens=args.num_prompt_tokens,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = model.tokenizer

    # ── Generate ──
    if args.is_llama3:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": args.prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt_text = args.prompt

    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(
        model.base_model.device
    )

    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)

    if args.benchmark:
        # Warmup
        for _ in range(2):
            _ = model.eagenerate(
                input_ids, max_new_tokens=32, max_length=args.max_length,
                temperature=args.temperature, top_p=args.top_p,
                is_llama3=args.is_llama3,
            )

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        num_runs = 5
        total_new_tokens = 0
        for _ in range(num_runs):
            output_ids, num_tokens, num_iters = model.eagenerate(
                input_ids, max_new_tokens=args.max_new_tokens,
                max_length=args.max_length, temperature=args.temperature,
                top_p=args.top_p, log=True, is_llama3=args.is_llama3,
            )
            total_new_tokens += num_tokens

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        print(f"\n{'='*60}")
        print(f"Benchmark Results:")
        print(f"  Avg tokens/run:  {total_new_tokens/num_runs:.1f}")
        print(f"  Avg time/run:    {elapsed/num_runs:.3f}s")
        print(f"  Throughput:      {total_new_tokens/elapsed:.1f} tokens/s")
        print(f"  Avg acceptance:  {total_new_tokens/(num_iters*num_runs):.2f} tokens/step")
        print(f"{'='*60}")

    else:
        output_ids, num_tokens, num_iters = model.eagenerate(
            input_ids, max_new_tokens=args.max_new_tokens,
            max_length=args.max_length, temperature=args.temperature,
            top_p=args.top_p, log=True, is_llama3=args.is_llama3,
        )

        output_text = tokenizer.decode(
            output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
        )
        print(f"Output: {output_text}")
        print(f"\n  New tokens: {num_tokens}")
        print(f"  Iterations: {num_iters}")
        print(f"  Avg acceptance: {num_tokens/max(num_iters,1):.2f} tokens/step")


if __name__ == "__main__":
    main()
