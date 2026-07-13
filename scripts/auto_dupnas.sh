#!/usr/bin/env bash
set -euo pipefail

OPTION="${1:-${OPTION:-}}"

case "${OPTION}" in
  stage1|stage2|stage3|stage4|full-stage)
    ;;
  *)
    echo "Error: Invalid OPTION: '${OPTION}'"
    echo "Valid options: stage1, stage2, stage3, stage4, full-stage"
    exit 1
    ;;
esac

echo "Selected option: ${OPTION}"

PERSIST_TRAIN_LOG="/4TB/aeuser/DupNAS-AE/DupNAS/NASBase/train_log"
#PERSIST_CKPT_LOG="/4TB/aeuser/DupNAS-AE/DupNAS/NASBase/checkpoints"

LOCAL_TRAIN_LOG="NASBase/train_log"
LOCAL_CKPT_LOG="NASBase/checkpoints"

copy_required_file() {
  local src="$1"
  local dst_dir="$2"

  if [[ ! -f "$src" ]]; then
    echo "ERROR: Required prerequisite file not found:"
    echo "  $src"
    exit 1
  fi

  mkdir -p "$dst_dir"
  cp "$src" "$dst_dir/"

  echo "Copied prerequisite:"
  echo "  $src"
  echo "  -> $dst_dir/"
}

restore_stage_files() {
  case "${OPTION}" in
    stage1)
      # No prerequisite files needed
      ;;

    stage2)
      copy_required_file \
        "${PERSIST_TRAIN_LOG}/${SUFFIX}_ssoptlog.json" \
        "${LOCAL_TRAIN_LOG}"
      ;;

    stage3)
      copy_required_file \
        "${PERSIST_TRAIN_LOG}/${SUFFIX}_ssoptlog.json" \
        "${LOCAL_TRAIN_LOG}"

      copy_required_file \
        "${PERSIST_TRAIN_LOG}/${SUFFIX}_trsupnetresults.json" \
        "${LOCAL_CKPT_LOG}"

      copy_required_file \
        "${PERSIST_TRAIN_LOG}/${SUFFIX}_supernet_shuffle_best.pth" \
        "${LOCAL_CKPT_LOG}"
      ;;

    stage4)
      copy_required_file \
        "${PERSIST_TRAIN_LOG}/${SUFFIX}_ssoptlog.json" \
        "${LOCAL_TRAIN_LOG}"

      copy_required_file \
        "${PERSIST_TRAIN_LOG}/${SUFFIX}_trsupnetresults.json" \
        "${LOCAL_CKPT_LOG}"

      copy_required_file \
        "${PERSIST_TRAIN_LOG}/${SUFFIX}_supernet_shuffle_best.pth" \
        "${LOCAL_CKPT_LOG}"

      copy_required_file \
        "${PERSIST_TRAIN_LOG}/${SUFFIX}_evosearchlog.json" \
        "${LOCAL_TRAIN_LOG}"
      ;;
  esac
}
# =========================
# User configuration
# =========================
KAGGLE_API_TOKEN=KGAT_dc7f37c8f3c2822a09adafe42d086f97

ARC="shuffle"
MODE="dupnas"
VMSIZE="128"
SUFFIX="aetest"

# =========================
# Configure Kaggle access
# =========================
export KAGGLE_API_TOKEN

mkdir -p ~/.kaggle
echo "${KAGGLE_API_TOKEN}" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

# =========================
# Prepare DupNAS
# =========================
cd DupNAS/

cp "settings/settings-${ARC}.py" settings.py

LOG_PREFIX="${ARC}-im100-${MODE}-vm${VMSIZE}-${SUFFIX}"

# =========================
# Stage functions
# =========================
run_stage1() {
  echo "=============================="
  echo "Starting Stage 1"
  echo "=============================="

  python3.9 -m NASBase.run_nas \
    --stages 1 \
    --arc "${ARC}" \
    --dataset IMAGE100 \
    --mode "${MODE}" \
    --vmsize "${VMSIZE}" \
    --suffix "${SUFFIX}" \
    --no-rlogger \
    2>&1 | tee "${LOG_PREFIX}-s1.txt"

  echo "Stage 1 finished successfully."
}

run_stage2() {
  echo "=============================="
  echo "Starting Stage 2"
  echo "=============================="

  python3.9 -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29601 \
    -m NASBase.run_nas \
    --stages 2 \
    --arc "${ARC}" \
    --dataset IMAGE100 \
    --mode "${MODE}" \
    --vmsize "${VMSIZE}" \
    --suffix "${SUFFIX}" \
    --no-rlogger \
    --dist ddp \
    --amp fp16 \
    2>&1 | tee "${LOG_PREFIX}-s2.txt"

  # # single gpu 
  # python3.9 -m NASBase.run_nas \
  #   --stages 2 \
  #   --arc "${ARC}" \
  #   --dataset IMAGE100 \
  #   --mode "${MODE}" \
  #   --vmsize "${VMSIZE}" \
  #   --suffix "${SUFFIX}" \
  #   --no-rlogger \
  #   2>&1 | tee "${LOG_PREFIX}-s2.txt"

  echo "Stage 2 finished successfully."
}

run_stage3() {
  echo "=============================="
  echo "Starting Stage 3"
  echo "=============================="

  python3.9 -m NASBase.run_nas \
    --stages 3 \
    --arc "${ARC}" \
    --dataset IMAGE100 \
    --mode "${MODE}" \
    --vmsize "${VMSIZE}" \
    --suffix "${SUFFIX}" \
    --no-rlogger \
    2>&1 | tee "${LOG_PREFIX}-s3.txt"

  echo "Stage 3 finished successfully."
}

run_stage4() {
  echo "=============================="
  echo "Starting Stage 4"
  echo "=============================="

  python3.9 -m torch.distributed.run \
    --nnodes=1 \
    --nproc_per_node=4 \
    --max_restarts=0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29611 \
    -m NASBase.run_nas \
    --stages 4 \
    --arc "${ARC}" \
    --dataset IMAGE100 \
    --mode "${MODE}" \
    --vmsize "${VMSIZE}" \
    --suffix "${SUFFIX}" \
    --no-rlogger \
    --dist ddp \
    --amp fp16 \
    2>&1 | tee "${LOG_PREFIX}-s4.txt"

  echo "Stage 4 finished successfully."
}

# =========================
# Run selected option
# =========================


case "${OPTION}" in
  stage1)
    run_stage1
    ;;

  stage2)
    restore_stage_files
    run_stage2
    ;;

  stage3)
    restore_stage_files
    run_stage3
    ;;

  stage4)
    restore_stage_files
    run_stage4
    ;;

  full-stage)
    run_stage1
    run_stage2
    run_stage3
    run_stage4

    echo "=============================="
    echo "All stages finished successfully."
    echo "=============================="
    ;;
esac

