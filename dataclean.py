from datasets import load_dataset
import json

print("데이터 다운로드 중...")
dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")

print("Train / Test 분리 중...")
split_dataset = dataset.train_test_split(test_size=0.01, seed=42)

def clean_and_save_jsonl(ds, filename):
    valid_count = 0
    with open(filename, "w", encoding="utf-8") as f:
        for row in ds:
            try:
                # 1. ID를 무조건 문자열로 강제 변환
                row_id = str(row.get("id", ""))
                
                # 2. 대화 기록 검증 및 문자열 강제 변환
                convs = row.get("conversations", [])
                if not isinstance(convs, list):
                    continue # 리스트가 아니면 불량 데이터로 간주하고 버림
                
                clean_convs = []
                for c in convs:
                    clean_convs.append({
                        "from": str(c.get("from", "")),
                        "value": str(c.get("value", "")) # 숫자가 있어도 강제로 문자로 만듦
                    })
                
                # 3. 깨끗해진 데이터만 저장
                clean_row = {"id": row_id, "conversations": clean_convs}
                f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
                valid_count += 1
                
            except Exception:
                continue # 혹시 모를 다른 에러가 있는 행도 스킵

    print(f"[SUCCESS] {filename} 저장 완료! (정상 데이터: {valid_count}개)")

# 파일 덮어쓰기 실행
clean_and_save_jsonl(split_dataset["train"], "sharegpt_train.jsonl")
clean_and_save_jsonl(split_dataset["test"], "sharegpt_test.jsonl")
