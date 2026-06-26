export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="./:${PYTHONPATH:-}"
MASTER_PORT=$((RANDOM % 101 + 20000))
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

CHECKPOINT="${CHECKPOINT:-/mnt/samsung/Project/CoRL-ICRA/JanusVLN_Infinite/model/JanusVLN_Base}"
SCENE_IDS="${SCENE_IDS:-EU6Fwq7SyZv}"
OUTPUT_PATH="${OUTPUT_PATH:-evaluation/scene/${SCENE_IDS}}"
CONFIG="${CONFIG:-config/vln_r2r.yaml}"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"
SAVE_VIDEO="${SAVE_VIDEO:-0}"

echo "CHECKPOINT: ${CHECKPOINT}"
echo "SCENE_IDS: ${SCENE_IDS}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "CONFIG: ${CONFIG}"
echo "EVAL_SPLIT: ${EVAL_SPLIT}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "SAVE_VIDEO: ${SAVE_VIDEO}"

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
  "${extra_args[@]}" \
  --output_path "${OUTPUT_PATH}"
