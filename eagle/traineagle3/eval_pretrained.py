"""
Standalone evaluation script to measure position-wise (0~6) accuracy
of a pretrained EAGLE3 draft model (without quantization).

This reuses the same Model class and evaluation logic from main.py/cnets.py
but only runs the test-set evaluation loop (no training).

Usage (via DeepSpeed):
    python -m deepspeed.launcher.runner --num_gpus 4 eval_pretrained.py \
        --deepspeed_config ds_config.json \
        --basepath ../../Llama-3.1-8B-Instruct \
        --testpath ../../sharegpt_test.jsonl \
        --draftpath ../../EAGLE3-LLaMA3.1-Instruct-8B \
        --num_hidden_layers 1 \
        --quant_method none
"""
import argparse
import deepspeed
import json
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Evaluate pretrained EAGLE3 draft model accuracy')
parser.add_argument('--basepath', type=str, default='../../Llama-3.1-8B-Instruct',
                    help='Path to the base LLM (e.g. Llama-3.1-8B-Instruct)')
parser.add_argument('--trainpath', type=str, default='../../sharegpt_train.jsonl',
                    help='Path to train dataset (jsonl, used for draft vocab building if cache.pt missing)')
parser.add_argument('--testpath', type=str, default='../../sharegpt_test.jsonl',
                    help='Path to test dataset (jsonl)')
parser.add_argument('--draftpath', type=str, required=True,
                    help='Path to pretrained EAGLE3 draft model weights')
parser.add_argument('--num_hidden_layers', type=int, default=1,
                    help='Number of hidden layers for the draft model')
parser.add_argument('--quant_method', type=str, default='none',
                    choices=['littlebit', 'onebit', 'none'],
                    help='Quantization method (use "none" for pretrained eval)')
parser.add_argument('--eff_bit', type=float, default=0.1,
                    help='Effective bit for LittleBit quantization (only used if quant_method=littlebit)')
parser.add_argument('--output', type=str, default=None,
                    help='Path to save evaluation results as JSON')
parser.add_argument("--local_rank", type=int, default=-1)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

# ── Load DeepSpeed config ─────────────────────────────────────────────────────
deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)


class TrainConfig(dict):
    def __getattr__(self, key):
        return self.get(key)


train_config = TrainConfig({
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 1,  # not used for eval, but required by Model
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "config.json",
    "gradient_checkpoint": False,
    "gradient_checkpointing": False,
    "disable_littlebit": True if args.quant_method == "none" else False,
    "draftpath": args.draftpath,
    "eff_bit": args.eff_bit,
    "quant_method": args.quant_method,
})

# ── Imports that depend on the working directory ──────────────────────────────
from transformers import AutoTokenizer
from cnets import padding, Model
from configs import EConfig
from typing import Any, Dict, List

# ── Dataset building (reused from main.py) ────────────────────────────────────
from datasets import load_dataset


def build_dataset_rank(tokenizer, datapath):
    ds = load_dataset('json', data_files=datapath)
    ds = ds['train']
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 8

    def preprocess_function(examples):
        new_examples = {
            "attention_mask": [],
            "input_ids": [],
            "loss_mask": []
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
                messages.append(
                    {"role": role, "content": sentence["value"]}
                )
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            if len(input_ids) > train_config["max_len"]:
                continue
            loss_mask = torch.ones_like(input_ids)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            total_len = len(input_ids)
            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)
                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
                if i == 0:
                    loss_mask[cur_len: cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3: cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=True
    )
    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:
    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['input_ids'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_attention_mask = torch.cat(
            [self.paddingtensor2D(item['attention_mask'], max_length) for item in features])
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item['loss_mask'], max_length) for item in features])
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


# ── Main evaluation ───────────────────────────────────────────────────────────
def main():
    tokenizer = AutoTokenizer.from_pretrained(args.basepath)
    testdataset = build_dataset_rank(tokenizer, args.testpath)

    config = EConfig.from_pretrained(train_config["config_path"])
    if args.num_hidden_layers is not None:
        config.num_hidden_layers = args.num_hidden_layers

    # Build model with quant_method=none → no quantization applied
    model = Model(config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True)

    # Build draft vocab mapping (uses cache.pt if available, otherwise builds from trainpath)
    model.scandata(args.trainpath, args.basepath)

    # Initialize DeepSpeed engine (evaluation-only, no optimizer needed)
    args.deepspeed_config = None
    model_engine, _, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    global_rank = deepspeed.comm.get_rank()
    rank = deepspeed.comm.get_local_rank()
    world_size = deepspeed.comm.get_world_size()

    sampler = DistributedSampler(testdataset, num_replicas=world_size, rank=global_rank, shuffle=False)
    test_loader = DataLoader(
        testdataset,
        batch_size=train_config["bs"],
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=DataCollatorWithPadding()
    )

    # ── Evaluation loop ───────────────────────────────────────────────────────
    model.eval()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    if global_rank == 0:
        print("=" * 60)
        print("Evaluating pretrained EAGLE3 draft model (no quantization)")
        print(f"  Draft model path : {args.draftpath}")
        print(f"  Base model path  : {args.basepath}")
        print(f"  Test data path   : {args.testpath}")
        print(f"  Hidden layers    : {args.num_hidden_layers}")
        print(f"  Quant method     : {args.quant_method}")
        print(f"  Num test samples : {len(testdataset)}")
        print("=" * 60)

    for batch_idx, data in enumerate(tqdm(test_loader, disable=(global_rank != 0), desc="Evaluating")):
        with torch.no_grad():
            plosses, vlosses, acces = model_engine(
                input_ids=data["input_ids"].to(rank),
                attention_mask=data["attention_mask"].to(rank),
                loss_mask=data["loss_mask"],
            )
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))]

    # ── Aggregate and print results ───────────────────────────────────────────
    results = {"accuracy": {}, "ploss": {}}

    if global_rank == 0:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (Pretrained EAGLE3, no quantization)")
        print("=" * 60)

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        results["accuracy"][f"pos_{i}"] = acc_i
        if global_rank == 0:
            print(f"  Position {i} | Acc: {acc_i:.4f}")

    if global_rank == 0:
        print("-" * 60)

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        results["ploss"][f"pos_{i}"] = loss_i
        if global_rank == 0:
            print(f"  Position {i} | pLoss: {loss_i:.4f}")

    if global_rank == 0:
        print("=" * 60)

    # ── Save results to JSON ──────────────────────────────────────────────────
    if global_rank == 0:
        output_path = args.output or "eval_pretrained_results.json"
        results["config"] = {
            "draftpath": args.draftpath,
            "basepath": args.basepath,
            "testpath": args.testpath,
            "num_hidden_layers": args.num_hidden_layers,
            "quant_method": args.quant_method,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
