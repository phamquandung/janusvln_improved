export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="./:${PYTHONPATH:-}"
# VGGT attention-sink window in FRAMES for the StreamVGGT/ghost path (read by
# StartRecentKVCache in the model). 8/48 = full sink (~36GB, use on the server);
# 4/24 (~35GB) fits a 47GB workstation. Only affects geometry_encoder in ghost/streamvggt.
export VGGT_KV_START="${VGGT_KV_START:-8}"
export VGGT_KV_RECENT="${VGGT_KV_RECENT:-48}"
# Geometry memory policy at EVAL: unset -> use the checkpoint's trained policy (full/sink).
# Set GEOMETRY_MEMORY_POLICY=importance to ablate GHOST importance-eviction at eval only
# (non-parametric; runs without retraining). GEOMETRY_KV_BUDGET = token budget for it.
export GEOMETRY_MEMORY_POLICY="${GEOMETRY_MEMORY_POLICY:-importance}"
export GEOMETRY_KV_BUDGET="${GEOMETRY_KV_BUDGET:-1200000}"
MASTER_PORT=$((RANDOM % 101 + 20000))
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

CHECKPOINT="${CHECKPOINT:-/mnt/samsung/Project/CoRL-ICRA/JanusVLN/model/JanusVLN_GHOST_full_r2r}"
SCENE_IDS="${SCENE_IDS:-EU6Fwq7SyZv}"
OUTPUT_PATH="${OUTPUT_PATH:-evaluation/scene/${SCENE_IDS}}"
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"
MAX_STEPS="${MAX_STEPS:-400}"

echo "CHECKPOINT: ${CHECKPOINT}"
echo "SCENE_IDS: ${SCENE_IDS}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "CONFIG: ${CONFIG}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "SAVE_VIDEO: ${SAVE_VIDEO}"
echo "VGGT_KV window (frames): start=${VGGT_KV_START} recent=${VGGT_KV_RECENT}"
echo "GEOMETRY_MEMORY_POLICY (override): ${GEOMETRY_MEMORY_POLICY:-<checkpoint default>}  budget=${GEOMETRY_KV_BUDGET:-<default>}"
echo "MAX_STEPS: ${MAX_STEPS}"

mkdir -p "${OUTPUT_PATH}"

extra_args=()
if [ "${SAVE_VIDEO}" = "1" ]; then
  extra_args+=(--save_video)
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port=$MASTER_PORT src/evaluation_scene.py \
  --model_path "${CHECKPOINT}" \
  --habitat_config_path "${CONFIG}" \
  --eval_split "${EVAL_SPLIT}" \
  --scene_ids "${SCENE_IDS}" \
  --max_steps "${MAX_STEPS}" \
  "${extra_args[@]}" \
  --output_path "${OUTPUT_PATH}"
