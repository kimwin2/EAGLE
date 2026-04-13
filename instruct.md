# BiTA-EAGLE3 학습 가이드 (2x2 Mini-Tree / Serial Chain)

## 목차
1. [사전 준비](#1-사전-준비)
2. [환경 설정](#2-환경-설정)
3. [데이터 준비](#3-데이터-준비)
4. [학습 실행](#4-학습-실행)
5. [추론 테스트](#5-추론-테스트)
6. [트러블슈팅](#6-트러블슈팅)

---

## 1. 사전 준비

### 필요한 모델 체크포인트

| 항목 | 설명 | 예시 경로 |
|------|------|-----------|
| **Target LLM** | 타겟 모델 (HuggingFace 형식) | `meta-llama/Meta-Llama-3-8B-Instruct` |
| **EAGLE-3 드래프트 모델** | 사전학습된 EAGLE-3 체크포인트 | `yuhuili/EAGLE3-LLaMA3-Instruct-8B` |
| **학습 데이터** | ShareGPT 포맷 JSON | `ShareGPT_V4.5_filtered.json` |

### 필요 GPU

- **최소**: 1x A100 80GB (또는 RTX 4090 24GB — 짧은 max_len 필요)
- **권장**: 1x A100 80GB (target 모델 fp16 + EAGLE-3 backbone + BiTA embeddings)
- BiTA 학습 파라미터는 매우 적음 (~0.01%), 하지만 target 모델 forward pass가 필요하므로 VRAM이 중요

---

## 2. 환경 설정

### 서버에서 레포 클론 및 환경 구성

```bash
# 1. 레포 클론
git clone https://github.com/kimwin2/EAGLE.git
cd EAGLE

# 2. conda 환경 생성
conda create -n bita-eagle python=3.10 -y
conda activate bita-eagle

# 3. PyTorch 설치 (CUDA 버전에 맞게)
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121

# 4. 나머지 의존성
pip install transformers>=4.53.1 accelerate==0.26.0 datasets sentencepiece protobuf wandb
pip install safetensors huggingface_hub tqdm

# 5. (선택) flash-attention 설치 (속도 향상)
pip install flash-attn --no-build-isolation
```

### 유닛 테스트 실행

```bash
cd EAGLE
python eagle/test_bita.py
```

**기대 출력:**
```
============================================================
BiTA-EAGLE3 Tests — 2x2 + Serial Topologies
============================================================

── 2x2 Mini-Tree ──
[2x2-1] Mask shape ✓
...
── Serial Chain ──
[serial-1] Mask shape ✓
...
ALL 19 TESTS PASSED ✓
```

---

## 3. 데이터 준비

### ShareGPT 포맷 데이터

학습 데이터는 EAGLE-3와 동일한 **ShareGPT 포맷** JSON을 사용합니다.

```json
[
  {
    "id": "sample_001",
    "conversations": [
      {"from": "human", "value": "What is speculative decoding?"},
      {"from": "gpt", "value": "Speculative decoding is a technique..."}
    ]
  },
  ...
]
```

### 데이터 다운로드

```bash
# ShareGPT 데이터 다운로드 (EAGLE에서 사용하는 것과 동일)
# 방법 1: HuggingFace에서 직접 다운로드
wget https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json

# 방법 2: 이미 서버에 있다면 경로만 확인
ls /path/to/your/ShareGPT_data.json
```

### Train/Test 분할 (선택)

```bash
python -c "
import json, random
random.seed(42)
with open('ShareGPT_V4.3_unfiltered_cleaned_split.json') as f:
    data = json.load(f)
random.shuffle(data)
split = int(len(data) * 0.95)
with open('train.json', 'w') as f:
    json.dump(data[:split], f)
with open('test.json', 'w') as f:
    json.dump(data[split:], f)
print(f'Train: {split}, Test: {len(data)-split}')
"
```

---

## 4. 학습 실행

### 기본 학습 명령어 (2x2 Mini-Tree)

```bash
cd EAGLE

python eagle/train_adapter.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_config_path eagle/traineagle3/config.json \
    --eagle3_weights_path /path/to/eagle3/pytorch_model.bin \
    --train_data /path/to/train.json \
    --test_data /path/to/test.json \
    --save_dir ./bita_ckpt_2x2 \
    --topology 2x2 \
    --num_epochs 20 \
    --batch_size 1 \
    --lr 1e-3 \
    --num_prompt_tokens 8 \
    --max_len 2048 \
    --warmup_steps 200 \
    --ce_weight 1.0 \
    --kl_weight 0.5
```

### Serial Chain 학습 명령어

```bash
python eagle/train_adapter.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_config_path eagle/traineagle3/config.json \
    --eagle3_weights_path /path/to/eagle3/pytorch_model.bin \
    --train_data /path/to/train.json \
    --save_dir ./bita_ckpt_serial \
    --topology serial \
    --num_epochs 20
```

> 💡 **토폴로지 차이:**  
> - `--topology 2x2`: 2 branches × depth 2 (M_1a→M_2a, M_1b→M_2b). 캐스캐이딩 실패에 강함.  
> - `--topology serial`: 1 chain × depth 4 (M_1→M_2→M_3→M_4). 더 깊은 추측 가능.

### 매개변수 설명

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--base_model_path` | (필수) | HuggingFace 형식 타겟 LLM 경로 |
| `--eagle3_config_path` | (필수) | EAGLE-3 드래프트 모델 config.json |
| `--eagle3_weights_path` | (필수) | EAGLE-3 가중치 (.bin 또는 .safetensors) |
| `--train_data` | (필수) | ShareGPT 형식 학습 데이터 |
| `--test_data` | None | 평가용 데이터 (선택) |
| `--save_dir` | `./bita_adapter_ckpt` | 체크포인트 저장 경로 |
| `--num_epochs` | 20 | 에포크 수 |
| `--batch_size` | 1 | 배치 크기 |
| `--lr` | 1e-3 | 학습률 (BiTA 임베딩은 높은 lr 가능) |
| `--num_prompt_tokens` | 8 | [P] 프롬프트 토큰 수 |
| `--max_len` | 2048 | 최대 시퀀스 길이 |
| `--ce_weight` | 1.0 | Cross-Entropy 손실 가중치 |
| `--kl_weight` | 0.5 | KL Divergence 손실 가중치 |
| `--topology` | `2x2` | 마스크 토폴로지: `2x2` (미니트리) 또는 `serial` (직렬 체인) |

### EAGLE-3 체크포인트 경로 찾기

HuggingFace에서 EAGLE-3 체크포인트를 다운로드한 경우:

```bash
# 방법 1: HuggingFace 캐시에서 찾기
python -c "
from huggingface_hub import snapshot_download
path = snapshot_download('yuhuili/EAGLE3-LLaMA3-Instruct-8B')
print(f'Downloaded to: {path}')
"

# 방법 2: 수동 다운로드
huggingface-cli download yuhuili/EAGLE3-LLaMA3-Instruct-8B --local-dir ./eagle3_ckpt

# config.json과 pytorch_model.bin (또는 model.safetensors) 확인
ls ./eagle3_ckpt/
```

### nohup / tmux로 백그라운드 실행

```bash
# tmux 사용 (권장)
tmux new -s bita_train

cd EAGLE
python eagle/train_adapter.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_config_path ./eagle3_ckpt/config.json \
    --eagle3_weights_path ./eagle3_ckpt/pytorch_model.bin \
    --train_data ./train.json \
    --save_dir ./bita_ckpt \
    --num_epochs 20

# Ctrl+B, D 로 detach
# tmux attach -t bita_train 로 재접속
```

```bash
# 또는 nohup 사용
nohup python eagle/train_adapter.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_config_path ./eagle3_ckpt/config.json \
    --eagle3_weights_path ./eagle3_ckpt/pytorch_model.bin \
    --train_data ./train.json \
    --save_dir ./bita_ckpt \
    --num_epochs 20 \
    > train.log 2>&1 &

# 로그 확인
tail -f train.log
```

### 학습 출력 예시

```
[1/5] Loading target model...
[2/5] Loading TreeBiTA adapter...
[TreeBiTAAdapter] Loaded backbone. Missing (BiTA params): 4, Unexpected: 0
  Total params:       202,899,456
  Trainable params:         49,152  ← 매우 작음!
[3/5] Building datasets...
  Train samples: 89000
[4/5] Setting up optimizer...
[5/5] Starting training...

Epoch 1/20: 100%|██████| 89000/89000 [2:15:30] loss=3.2451, acc=0.3210, lr=1.00e-03
  Epoch 1 — Train Loss: 3.2451, CE: 2.8120, KL: 0.4331, Acc: 0.3210
  Saved BiTA embeddings to ./bita_ckpt/epoch_1

Epoch 2/20: 100%|██████| 89000/89000 [2:14:10] loss=2.1530, acc=0.4520, lr=9.50e-04
  ...
```

### 체크포인트 구조

```
bita_ckpt/
├── epoch_1/
│   └── bita_embeddings.pt    ← ~200KB (매우 작음!)
├── epoch_2/
│   └── bita_embeddings.pt
├── ...
└── epoch_20/
    └── bita_embeddings.pt
```

> 💡 저장되는 건 BiTA 임베딩 4개뿐 (`prompt_embeddings`, `prompt_hidden`, `mask_embeddings`, `mask_hidden`)이라 체크포인트 크기가 매우 작습니다.

---

## 5. 추론 테스트

### 학습된 BiTA 어댑터로 생성

```bash
python eagle/inference.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_model_path ./eagle3_ckpt \
    --bita_weights_path ./bita_ckpt/epoch_20/bita_embeddings.pt \
    --prompt "Explain the concept of speculative decoding in LLM inference." \
    --topology 2x2 \
    --is_llama3 \
    --max_new_tokens 256
```

### 벤치마크 모드 (속도 비교)

```bash
python eagle/inference.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_model_path ./eagle3_ckpt \
    --bita_weights_path ./bita_ckpt/epoch_20/bita_embeddings.pt \
    --is_llama3 \
    --benchmark \
    --max_new_tokens 256
```

**기대 출력:**
```
============================================================
Benchmark Results:
  Avg tokens/run:  245.2
  Avg time/run:    3.150s
  Throughput:      77.8 tokens/s
  Avg acceptance:  1.85 tokens/step
============================================================
```

---

## 6. 트러블슈팅

### CUDA OOM (Out of Memory)

```bash
# max_len 줄이기
--max_len 1024

# 또는 gradient checkpointing 사용 (target 모델)
# train_adapter.py에서 target_model.gradient_checkpointing_enable() 추가 가능
```

### EAGLE-3 config에 `target_hidden_size` 없음

기존 EAGLE-3 config.json에 `target_hidden_size`가 없는 경우, `fc` 레이어가 `hidden_size * 3`으로 초기화됩니다. 타겟 모델의 hidden size가 EAGLE-3의 hidden size와 다른 경우 config.json에 추가:

```json
{
  "target_hidden_size": 4096,
  ...
}
```

### 가중치 로딩 시 Missing keys 경고

```
Missing (BiTA params): 4, Unexpected: 0
```

이건 **정상**입니다. 4개의 missing key는 BiTA 임베딩 (`prompt_embeddings`, `prompt_hidden`, `mask_embeddings`, `mask_hidden`)으로, 새로 학습할 파라미터입니다.

### `d2t` / `t2d` 버퍼 초기화

Draft vocab ≠ Target vocab인 경우 (예: EAGLE-3의 `draft_vocab_size=32000`, target의 `vocab_size=128256`), `d2t`와 `t2d` 매핑 버퍼가 올바르게 초기화되어야 합니다. 이 매핑은 EAGLE-3 체크포인트에 포함되어 있어야 합니다.

### 학습이 안 되는 경우 체크리스트

```bash
# 1. trainable param이 0이 아닌지 확인
python -c "
import torch, sys; sys.path.insert(0,'eagle')
from tree_bita_model import TreeBiTAAdapter
from model.configs import EConfig
config = EConfig.from_pretrained('eagle/traineagle3/config.json')
adapter = TreeBiTAAdapter(eagle3_config=config)
adapter.freeze_backbone()
print(f'Trainable: {adapter.count_trainable_params():,}')
# 기대값: 약 49,152 (8*4096*2 + 4*4096*2 with hidden_size=4096, p=8)
"

# 2. CUDA 사용 가능한지
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 3. Target 모델 로딩 테스트
python -c "
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', torch_dtype='float16')
print('OK', sum(p.numel() for p in m.parameters()))
"
```

---

## 빠른 시작 (한 줄 요약)

```bash
# 1. 클론 & 환경
git clone https://github.com/kimwin2/EAGLE.git && cd EAGLE
pip install torch transformers accelerate datasets safetensors huggingface_hub tqdm sentencepiece

# 2. 테스트
python eagle/test_bita.py

# 3. 학습 (2x2)
python eagle/train_adapter.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_config_path eagle/traineagle3/config.json \
    --eagle3_weights_path /path/to/eagle3/pytorch_model.bin \
    --train_data /path/to/sharegpt.json \
    --save_dir ./bita_ckpt --topology 2x2 --num_epochs 20

# 3-alt. 학습 (serial)
python eagle/train_adapter.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_config_path eagle/traineagle3/config.json \
    --eagle3_weights_path /path/to/eagle3/pytorch_model.bin \
    --train_data /path/to/sharegpt.json \
    --save_dir ./bita_ckpt_serial --topology serial --num_epochs 20

# 4. 추론 (2x2)
python eagle/inference.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_model_path /path/to/eagle3 \
    --bita_weights_path ./bita_ckpt/epoch_20/bita_embeddings.pt \
    --topology 2x2 --is_llama3 --benchmark

# 4-alt. 추론 (serial)
python eagle/inference.py \
    --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
    --eagle3_model_path /path/to/eagle3 \
    --bita_weights_path ./bita_ckpt_serial/epoch_20/bita_embeddings.pt \
    --topology serial --is_llama3 --benchmark
```
