#!/bin/bash
# Lever-1 fine-tune: swap offline VGGT -> GHOST causal StreamVGGT (frozen),
# fine-tune FROM the released JanusVLN checkpoint (NOT a cold start) with ONLY the
# geometry projector (VGGTMerger, i.e. model.merger) unfrozen.
#
# Why this freezes everything but the projector: set_model() in train_qwen.py always
# sets model.merger.requires_grad=True and model.vggt.requires_grad=False; with all
# three tune flags below False, the LLM (model.* + lm_head), the Qwen visual tower and
# its merger are all frozen. Net trainable = the VGGTMerger projector only.

MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# Released JanusVLN checkpoint to fine-tune from (do NOT point at the Qwen base).
MODEL_PATH="${JANUSVLN_CKPT:-./model/JanusVLN-3B}"
# GHOST StreamVGGT weights (server: $model-checkpoint/StreamVGGT/model.pth). If empty,
# the encoder inherits the VGGT-backbone weights already in the checkpoint.
STREAMVGGT_MODEL_PATH="${STREAMVGGT_CKPT:-}"

OUTPUT_DIR="./JanusVLN_GHOST_projector_only"
CACHE_DIR="./cache"
mkdir -p $OUTPUT_DIR

DATASETS="train_r2r"

# GHOST lives in a sibling repo with absolute streamvggt.* imports.
export GHOST_SRC="${GHOST_SRC:-../GHOST/src}"
export NCCL_NVLS_ENABLE=0

torchrun --nproc_per_node=$NPROC_PER_NODE \
            --master_addr=$MASTER_ADDR \
            --master_port=$MASTER_PORT \
            src/qwen_vl/train/train_qwen.py \
            --model_name_or_path $MODEL_PATH \
            --geometry_encoder ghost \
            --streamvggt_model_path "$STREAMVGGT_MODEL_PATH" \
            --tune_mm_llm False \
            --tune_mm_vision False \
            --tune_mm_mlp False \
            --dataset_use $DATASETS \
            --output_dir $OUTPUT_DIR \
            --cache_dir $CACHE_DIR \
            --bf16 \
            --per_device_train_batch_size 1 \
            --gradient_accumulation_steps 8 \
            --learning_rate 1e-4 \
            --mm_projector_lr 1e-4 \
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
            --lr_scheduler_type "cosine" \
            --weight_decay 0.01 \
            --logging_steps 10 \
            --save_steps 1000 \
            --save_total_limit 1 \
            --deepspeed "scripts/zero3.json" \
            --gradient_checkpointing \
            --dataloader_num_workers 8 \
            --group_by_modality_length true \
            --seed 42 \
            --report_to "none" \
            --reference_frame first \
            > ${OUTPUT_DIR}/train.log 2>&1
