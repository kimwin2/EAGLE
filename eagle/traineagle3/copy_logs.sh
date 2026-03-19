#!/bin/bash
# 사용법: FILES 배열에 복사할 파일 경로를 넣고 실행
# 체크포인트 폴더명(checkpoints_qat_2gpu 등)으로 C:\log\ 아래에 폴더를 만들고 복사

DST="/c/log"

FILES=(
  "/group-volume/ym1012.kim/homepc/EAGLE/eagle/traineagle3/checkpoints_qat_2gpu/runs/events.out.tfevents.1773905810.run117178-lr5e5-5ep-2gpu-1layer-littlebi.605.0"
  "/group-volume/ym1012.kim/homepc/EAGLE/eagle/traineagle3/checkpoints/runs/events.out.tfevents.xxx.example2"
  "/group-volume/ym1012.kim/homepc/EAGLE/eagle/traineagle3/checkpoints_qat_03/runs/events.out.tfevents.xxx.example3"
)

for f in "${FILES[@]}"; do
  # runs/ 의 부모 폴더명을 꺼냄 (예: checkpoints_qat_2gpu)
  folder=$(basename "$(dirname "$(dirname "$f")")")
  mkdir -p "$DST/$folder"
  cp -v "$f" "$DST/$folder/"
done

echo "Done!"
