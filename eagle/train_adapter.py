# coding=utf-8
"""
train_adapter.py
================
Training script for BiTA-adapted EAGLE-3.

Only the learnable [P] prompt embeddings and [M] mask embeddings are trained.
All backbone parameters (EAGLE-3 draft model + target model) remain frozen.

Loss: CE + KL divergence at [M] positions, distilled from target model outputs.
"""

import argparse
import json
import os
import sys
import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
from accelerate.utils import set_seed
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup

set_seed(42)

# ─── Local imports ───
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tree_bita_model import (
    TreeBiTAAdapter, NUM_MASK_TOKENS, VALID_TOPOLOGIES,
)
from model.configs import EConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Data utilities (adapted from EAGLE-3's training pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataset(tokenizer, datapath: str, max_len: int = 2048):
    """Build a ShareGPT-format dataset with chat template applied."""

    ds = load_dataset('json', data_files=datapath)['train'].shuffle(seed=42)
    original_columns = ds.column_names
    num_proc = min(8, os.cpu_count() or 1)

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": [],
        }
        for i in range(len(examples['id'])):
            messages = [
                {"role": "system",
                 "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
            ]
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples['conversations'][i]
            if not source:
                continue
            if roles[source[0]["from"]] != "user":
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                messages.append({"role": role, "content": sentence["value"]})

            conversation = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation, return_tensors="pt", add_special_tokens=False,
            ).input_ids[0]

            if len(input_ids) > max_len:
                continue

            loss_mask = torch.ones_like(input_ids)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)
            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for ti, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)
                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
                if ti == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if ti != 0:
                    cur_len += 3
            loss_mask[cur_len:] = 0

            attention_mask = torch.ones_like(loss_mask)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds = ds.map(
        preprocess_function, batched=True, num_proc=num_proc,
        remove_columns=original_columns, load_from_cache_file=False,
    )
    ds.set_format(type="torch")
    return ds


class DataCollatorWithPadding:
    """Collate function that pads batch to max length."""

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        return torch.cat((intensors, padding_tensor), dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        return {
            "input_ids": torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features]),
            "attention_mask": torch.cat([self.paddingtensor2D(item['attention_mask'], max_length) for item in features]),
            "loss_mask": torch.cat([self.paddingtensor2D(item['loss_mask'], max_length) for item in features]),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def padding(tensor, left=True):
    """Shift tensor by 1 position (left or right)."""
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


class BiTATrainer:
    """
    Trainer for the BiTA adapter on EAGLE-3.

    Only trains [P] and [M] embeddings via knowledge distillation from
    the frozen target model.
    """

    def __init__(
        self,
        adapter: TreeBiTAAdapter,
        target_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        ce_weight: float = 1.0,
        kl_weight: float = 0.5,
        tree_depth: int = 2,
        device: str = "cuda",
    ):
        self.adapter = adapter
        self.target_model = target_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ce_weight = ce_weight
        self.kl_weight = kl_weight
        self.tree_depth = tree_depth
        self.device = device

        # Ensure target model is frozen and in eval mode
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def prepare_target_data(self, input_ids, attention_mask):
        """
        Run the frozen target model to get hidden states and logits.

        Returns:
            hidden_states: concatenated hidden states from first 3 layers
                          (matching EAGLE-3's training convention)
            target_logits: target model's logits (shifted for next-token)
        """
        outs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # EAGLE-3 uses first 3 hidden state layers concatenated
        h0 = outs.hidden_states[0]
        h1 = outs.hidden_states[1]
        h2 = outs.hidden_states[2]
        hidden_states = torch.cat((h0, h1, h2), dim=-1)

        target_logits = outs.logits
        return hidden_states, target_logits

    def compute_loss_at_positions(
        self,
        mask_logits: torch.Tensor,
        target_logits_at_positions: torch.Tensor,
        target_tokens_at_positions: torch.Tensor,
        position_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute CE + KL loss at [M] positions.

        Args:
            mask_logits: (batch, k, draft_vocab) predictions from [M] slots
            target_logits_at_positions: (batch, k, target_vocab) target logits
            target_tokens_at_positions: (batch, k) target argmax tokens
            position_mask: (batch, k) 1 where loss should be computed
        """
        # ── CE Loss ──
        # Map target tokens to draft vocab if needed
        if hasattr(self.adapter, 't2d') and self.adapter.config.vocab_size != self.adapter.config.draft_vocab_size:
            # Filter target logits to draft vocab
            target_logits_draft = target_logits_at_positions[..., self.adapter.t2d]
            # Map target token indices
            valid_mask = self.adapter.t2d[target_tokens_at_positions.long()]
        else:
            target_logits_draft = target_logits_at_positions
            valid_mask = torch.ones_like(position_mask)

        effective_mask = position_mask * valid_mask.float()

        # Target distribution (soft labels)
        target_probs = F.softmax(target_logits_draft.float(), dim=-1).detach()

        # Draft distribution (log probabilities)
        draft_logp = F.log_softmax(mask_logits.float(), dim=-1)

        # CE loss: -sum(target_p * log(draft_p))
        ce_loss = -torch.sum(target_probs * draft_logp, dim=-1)  # (batch, k)
        ce_loss = (ce_loss * effective_mask).sum() / (effective_mask.sum() + 1e-8)

        # ── KL Divergence Loss ──
        target_logp = F.log_softmax(target_logits_draft.float(), dim=-1)
        kl_loss = F.kl_div(draft_logp, target_probs, reduction='none').sum(-1)
        kl_loss = (kl_loss * effective_mask).sum() / (effective_mask.sum() + 1e-8)

        # ── Accuracy ──
        with torch.no_grad():
            pred_tokens = mask_logits.argmax(dim=-1)
            target_max = target_probs.argmax(dim=-1)
            correct = ((pred_tokens == target_max).float() * effective_mask).sum()
            accuracy = correct / (effective_mask.sum() + 1e-8)

        total_loss = self.ce_weight * ce_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
            "accuracy": accuracy,
        }

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        One training step (topology-aware).

        Ground truth mapping per slot is determined by adapter.depth_map:
          - 2x2:    slots [0,1,2,3] → depths [1,1,2,2] → positions [t+1,t+1,t+2,t+2]
          - serial:  slots [0,1,2,3] → depths [1,2,3,4] → positions [t+1,t+2,t+3,t+4]
        """
        self.adapter.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        loss_mask = batch["loss_mask"].to(self.device)

        batch_size, seq_len = input_ids.shape
        max_depth = max(self.adapter.depth_map.values())  # 2 for 2x2, 4 for serial

        # ── Step 1: Get target data ──
        hidden_states_all, target_logits = self.prepare_target_data(input_ids, attention_mask)

        with torch.no_grad():
            hidden_states_proj = self.adapter.fc(hidden_states_all)

        total_ce = 0.0
        total_kl = 0.0
        total_acc = 0.0
        num_valid = 0

        for b in range(batch_size):
            sample_len = attention_mask[b].sum().long().item()
            if sample_len < max_depth + 2:
                continue

            valid_positions = torch.where(
                loss_mask[b, :sample_len - max_depth] > 0
            )[0]
            if len(valid_positions) == 0:
                continue

            max_ctx = min(32, len(valid_positions))
            stride = max(1, len(valid_positions) // max_ctx)
            selected_positions = valid_positions[::stride][:max_ctx]

            for ctx_end in selected_positions:
                ctx_end_idx = ctx_end.item()
                if ctx_end_idx < 1:
                    continue

                # Check all future positions within bounds
                if ctx_end_idx + max_depth >= sample_len:
                    continue

                ctx_input_ids = input_ids[b:b+1, :ctx_end_idx + 1]
                ctx_hidden = hidden_states_proj[b:b+1, :ctx_end_idx + 1]
                last_hidden = ctx_hidden[:, -1:, :]

                mask_logits, _ = self.adapter.forward_single_pass(
                    hidden_states_context=ctx_hidden,
                    input_ids_context=ctx_input_ids,
                    last_hidden_for_mask=last_hidden,
                )

                # ── Build ground truth using depth_map ──
                # Each slot i predicts ctx_end + depth_map[i]
                target_positions = torch.tensor(
                    [ctx_end_idx + self.adapter.depth_map[i] for i in range(NUM_MASK_TOKENS)],
                    device=self.device,
                )
                position_valid = torch.ones(1, NUM_MASK_TOKENS, device=self.device)

                target_logits_m = target_logits[b, target_positions].unsqueeze(0)
                target_tokens_m = target_logits[b, target_positions].argmax(-1).unsqueeze(0)

                metrics = self.compute_loss_at_positions(
                    mask_logits, target_logits_m, target_tokens_m, position_valid
                )

                total_ce += metrics["ce_loss"].item()
                total_kl += metrics["kl_loss"].item()
                total_acc += metrics["accuracy"].item()
                num_valid += 1

                metrics["loss"].backward()

        if num_valid == 0:
            return {"loss": 0.0, "ce_loss": 0.0, "kl_loss": 0.0, "accuracy": 0.0}

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

        return {
            "loss": (total_ce * self.ce_weight + total_kl * self.kl_weight) / num_valid,
            "ce_loss": total_ce / num_valid,
            "kl_loss": total_kl / num_valid,
            "accuracy": total_acc / num_valid,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train BiTA adapter for EAGLE-3')
    parser.add_argument('--base_model_path', type=str, required=True,
                        help='Path to target LLM (e.g., LLaMA-3-8B-Instruct)')
    parser.add_argument('--eagle3_config_path', type=str, required=True,
                        help='Path to EAGLE-3 config.json')
    parser.add_argument('--eagle3_weights_path', type=str, required=True,
                        help='Path to EAGLE-3 pretrained weights (pytorch_model.bin or model.safetensors)')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (ShareGPT JSONL)')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Path to test data (ShareGPT JSONL)')
    parser.add_argument('--save_dir', type=str, default='./bita_adapter_ckpt',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=0.5)
    parser.add_argument('--num_prompt_tokens', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--gradient_accumulation', type=int, default=4)
    parser.add_argument('--topology', type=str, default='2x2', choices=['2x2', 'serial'],
                        help='Mask topology: "2x2" (mini-tree) or "serial" (causal chain)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, use_fast=False)

    # ── Load target model (frozen) ──
    print("[1/5] Loading target model...")
    from model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
    target_model = KVLlamaForCausalLM.from_pretrained(
        args.base_model_path, torch_dtype=torch.float16,
    ).to(device).eval()
    for p in target_model.parameters():
        p.requires_grad = False

    # ── Load EAGLE-3 adapter ──
    print(f"[2/5] Loading TreeBiTA adapter (topology={args.topology})...")
    config = EConfig.from_pretrained(args.eagle3_config_path)
    adapter = TreeBiTAAdapter(
        eagle3_config=config,
        topology=args.topology,
        num_prompt_tokens=args.num_prompt_tokens,
    )

    # Load pretrained EAGLE-3 weights
    if args.eagle3_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        eagle3_state = load_file(args.eagle3_weights_path)
    else:
        eagle3_state = torch.load(args.eagle3_weights_path, map_location="cpu")

    adapter.load_eagle3_weights(eagle3_state)
    adapter = adapter.to(torch.float16).to(device)

    print(f"  Total params:     {adapter.count_total_params():>12,}")
    print(f"  Trainable params: {adapter.count_trainable_params():>12,}")

    # ── Build datasets ──
    print("[3/5] Building datasets...")
    train_dataset = build_dataset(tokenizer, args.train_data, max_len=args.max_len)
    test_dataset = None
    if args.test_data:
        test_dataset = build_dataset(tokenizer, args.test_data, max_len=args.max_len)
    print(f"  Train samples: {len(train_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
        collate_fn=DataCollatorWithPadding(),
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=2, pin_memory=True,
            collate_fn=DataCollatorWithPadding(),
        )

    # ── Optimizer (only BiTA parameters) ──
    print("[4/5] Setting up optimizer...")
    trainable_params = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.num_epochs // args.gradient_accumulation
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps,
    )

    trainer = BiTATrainer(
        adapter=adapter,
        target_model=target_model,
        optimizer=optimizer,
        scheduler=scheduler,
        ce_weight=args.ce_weight,
        kl_weight=args.kl_weight,
        device=device,
    )

    # ── Training loop ──
    print("[5/5] Starting training...")
    os.makedirs(args.save_dir, exist_ok=True)

    best_test_loss = float("inf")

    for epoch in range(args.num_epochs):
        adapter.train()
        epoch_metrics = {"loss": [], "ce_loss": [], "kl_loss": [], "accuracy": []}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            metrics = trainer.train_step(batch)

            for k, v in metrics.items():
                epoch_metrics[k].append(v)

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        # Epoch summary
        avg_metrics = {k: sum(v) / max(len(v), 1) for k, v in epoch_metrics.items()}
        print(f"\n  Epoch {epoch+1} — Train Loss: {avg_metrics['loss']:.4f}, "
              f"CE: {avg_metrics['ce_loss']:.4f}, "
              f"KL: {avg_metrics['kl_loss']:.4f}, "
              f"Acc: {avg_metrics['accuracy']:.4f}")

        # ── Test evaluation ──
        if test_loader is not None:
            adapter.eval()
            test_metrics = {"loss": [], "accuracy": []}
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="  Testing"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    loss_mask = batch["loss_mask"].to(device)

                    hidden_states_all, target_logits = trainer.prepare_target_data(
                        input_ids, attention_mask
                    )
                    hidden_states_proj = adapter.fc(hidden_states_all)

                    # Simple evaluation: use full context, predict ahead
                    b = 0
                    sample_len = attention_mask[b].sum().long().item()
                    mid_pos = sample_len // 2
                    if mid_pos < 2:
                        continue

                    ctx_ids = input_ids[b:b+1, :mid_pos]
                    ctx_hid = hidden_states_proj[b:b+1, :mid_pos]
                    last_hid = ctx_hid[:, -1:, :]

                    mask_logits, _ = adapter.forward_single_pass(
                        ctx_hid, ctx_ids, last_hid
                    )

                    # Evaluate accuracy at each [M] position
                    num_mask = adapter.num_mask_slots
                    correct = 0
                    total = 0
                    for mi in range(num_mask):
                        depth = adapter.depth_map[mi]
                        tp = mid_pos + depth - 1
                        if tp < sample_len:
                            pred = mask_logits[0, mi].argmax().item()
                            tgt = target_logits[b, tp].argmax().item()
                            # Map if needed
                            if adapter.config.vocab_size != adapter.config.draft_vocab_size:
                                if adapter.t2d[tgt]:
                                    correct += int(pred == tgt)
                            else:
                                correct += int(pred == tgt)
                            total += 1

                    test_metrics["accuracy"].append(correct / max(total, 1))

            avg_test_acc = sum(test_metrics["accuracy"]) / max(len(test_metrics["accuracy"]), 1)
            print(f"  Test Accuracy: {avg_test_acc:.4f}")

        # ── Save checkpoint ──
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}")
        os.makedirs(ckpt_path, exist_ok=True)

        # Save only trainable parameters
        trainable_state = {
            k: v for k, v in adapter.state_dict().items()
            if any(x in k for x in ["prompt_embeddings", "prompt_hidden",
                                      "mask_embeddings", "mask_hidden"])
        }
        torch.save(trainable_state, os.path.join(ckpt_path, "bita_embeddings.pt"))
        print(f"  Saved BiTA embeddings to {ckpt_path}")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
