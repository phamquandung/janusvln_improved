#!/bin/bash -e
#SBATCH --job-name=janusvln-ghost-full
#SBATCH --output=logs/ghost_full_%j.log
#SBATCH --error=logs/ghost_full_%j.err
#SBATCH --nodelist=worker-2
#SBATCH --gpus=4
#SBATCH --cpus-per-task=60
#SBATCH --mem-per-cpu=8192
#
# !!! VERIFY: container must have the `janusvln` conda env WITH janusvln_improved's deps.
#SBATCH --container-image=/mnt/data/vmo-ai-task/dungpq6/ubuntu22-cuda128-conda-janusvln-spatialstack.sqsh
#SBATCH --container-mounts=/mnt/data/:/mnt/data/,/home/dungpq6/Project:/home/dungpq6/Project

# Full JanusVLN-recipe training FROM the Qwen2.5-VL base (NOT a fine-tune of JanusVLN_Base):
# train the LLM + geometry projector on R2R+RxR, with GHOST causal StreamVGGT (frozen) as the
# geometry encoder and JanusVLN's flat-add fusion (VSFI is a separate, not-yet-built change).
set -euo pipefail

source /home/dungpq6/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-janusvln}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/dungpq6/Project/janusvln_improved}"
cd "${PROJECT_ROOT}"
mkdir -p logs

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
echo "[check] qwen_vl import source:"
python -c "import qwen_vl; print('  qwen_vl ->', qwen_vl.__file__)" || true

export GHOST_SRC="${GHOST_SRC:-/home/dungpq6/Project/GHOST/src}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 10000))}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-$(nvidia-smi --list-gpus | wc -l)}}"

# --- start from the Qwen2.5-VL BASE (full reproduction, not a JanusVLN fine-tune) ----------
MODEL_PATH="${QWEN_BASE:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/Qwen2.5-VL-7B-Instruct}"
STREAMVGGT_MODEL_PATH="${STREAMVGGT_CKPT:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/StreamVGGT/checkpoints.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/vmo-ai-task/dungpq6/model-checkpoint/JanusVLN_GHOST_full_r2r_rxr}"
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache}"
DATASETS="${DATASETS:-train_r2r_rxr}"          # combined R2R+RxR annotation
DEEPSPEED_CFG="${DEEPSPEED_CFG:-scripts/zero2.json}"
# Lever 2 Tier A: GHOST importance-based geometry-memory eviction (head-free: saliency +
# temporal recency), bounded budget, active in train AND eval. Set GEOM_MEM_POLICY=full for
# the no-eviction +Geo baseline. Budget is per-run tunable (sweep it — that's the Tier A study).
GEOM_MEM_POLICY="${GEOM_MEM_POLICY:-full}"
GEOM_KV_BUDGET="${GEOM_KV_BUDGET:-1200000}"
# Lever 3 (fusion): "flat_add" (JanusVLN S+lam*G) or "gated" (VSFI-style zero-init gate).
FUSION_METHOD="${FUSION_METHOD:-gated}"
mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}"

# --- W&B (offline) -------------------------------------------------------------------------
export REPORT_TO="${REPORT_TO:-wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export WANDB_PROJECT="${WANDB_PROJECT:-janusvln-ghost}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-ghost_full_${SLURM_JOB_ID:-manual}}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${CACHE_DIR}/wandb}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
[[ "${REPORT_TO}" == "wandb" ]] && mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

echo "PROJECT_ROOT=${PROJECT_ROOT}  NPROC_PER_NODE=${NPROC_PER_NODE}  MASTER_PORT=${MASTER_PORT}"
echo "MODEL_PATH=${MODEL_PATH}  (Qwen2.5-VL base — full training, not fine-tune)"
echo "STREAMVGGT_MODEL_PATH=${STREAMVGGT_MODEL_PATH}  GHOST_SRC=${GHOST_SRC}"
echo "OUTPUT_DIR=${OUTPUT_DIR}  DATASETS=${DATASETS}  DEEPSPEED_CFG=${DEEPSPEED_CFG}  REPORT_TO=${REPORT_TO}"
echo "GEOM_MEM_POLICY=${GEOM_MEM_POLICY}  GEOM_KV_BUDGET=${GEOM_KV_BUDGET}  FUSION_METHOD=${FUSION_METHOD}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    src/qwen_vl/train/train_qwen.py \
    --model_name_or_path "${MODEL_PATH}" \
    --geometry_encoder ghost \
    --streamvggt_model_path "${STREAMVGGT_MODEL_PATH}" \
    --geometry_memory_policy "${GEOM_MEM_POLICY}" \
    --geometry_kv_budget "${GEOM_KV_BUDGET}" \
    --fusion_method "${FUSION_METHOD}" \
    --tune_mm_llm True \
    --tune_mm_mlp True \
    --tune_mm_vision False \
    --dataset_use "${DATASETS}" \
    --output_dir "${OUTPUT_DIR}" \
    --cache_dir "${CACHE_DIR}" \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --mm_projector_lr 1e-5 \
    --vision_tower_lr 1e-6 \
    --optim adamw_torch \
    --model_max_length 163840 \
    --data_flatten False \
    --max_pixels $((576*28*28)) \
    --min_pixels $((16*28*28)) \
    --base_interval 2 \
    --video_max_frames 8 \
    --video_min_frames 4 \
    --video_max_frame_pixels $((1664*28*28)) \
    --video_min_frame_pixels $((256*28*28)) \
    --num_train_epochs 1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --deepspeed "${DEEPSPEED_CFG}" \
    --gradient_checkpointing \
    --dataloader_num_workers 8 \
    --group_by_modality_length true \
    --seed 42 \
    --report_to "${REPORT_TO}" \
    --reference_frame first
