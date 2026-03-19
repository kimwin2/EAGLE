#!/bin/bash
# 서버에서 실행: 로그 파일을 로컬 Windows PC로 scp 전송
# 사용법: bash copy_logs.sh
#
# 아래 두 변수를 본인 환경에 맞게 수정하세요
WINDOWS_USER="ymkim"                    # Windows 사용자명
WINDOWS_IP="your_windows_ip"            # Windows PC IP (예: 192.168.0.10)
DST_BASE="C:/log"                       # Windows 저장 경로

FILES=(
  "/group-volume/ym1012.kim/homepc/EAGLE/eagle/traineagle3/checkpoints_qat_2gpu/runs/events.out.tfevents.1773905810.run117178-lr5e5-5ep-2gpu-1layer-littlebi.605.0"
  "/group-volume/ym1012.kim/homepc/EAGLE/eagle/traineagle3/checkpoints/runs/events.out.tfevents.xxx.example2"
  "/group-volume/ym1012.kim/homepc/EAGLE/eagle/traineagle3/checkpoints_qat_03/runs/events.out.tfevents.xxx.example3"
)

for f in "${FILES[@]}"; do
  folder=$(basename "$(dirname "$(dirname "$f")")")
  echo "[INFO] Copying to ${WINDOWS_USER}@${WINDOWS_IP}:${DST_BASE}/${folder}/"
  ssh ${WINDOWS_USER}@${WINDOWS_IP} "mkdir -p '${DST_BASE}/${folder}'" 2>/dev/null
  scp "$f" "${WINDOWS_USER}@${WINDOWS_IP}:${DST_BASE}/${folder}/"
done

echo "Done!"
