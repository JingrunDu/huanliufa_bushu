#!/bin/bash
# 训练 perspective 数据集
#
# 损失采用标准加权平均：Ll1 = sum(error * w) / sum(w)，分子分母都乘语义权重，
# 设备区误差占更大比重，同时总损失尺度归一到平均权重 1、不发生量级漂移。
# 致密化倾斜关闭（densify_equipment_factor=1.0），仅由损失加权引导优化。
# 语义权重：设备 3.0 / 背景 0.5。

# 切换到脚本所在目录，保证相对路径可用（无需修改绝对路径即可移植运行）
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DATA_PATH=${DATA_PATH:-"./data/perspective"}
OUTPUT_PATH=${OUTPUT_PATH:-"./output/perspective"}

mkdir -p ${OUTPUT_PATH}

OAR_JOB_ID=perspective python train.py \
    -s ${DATA_PATH} \
    -i images \
    -m ${OUTPUT_PATH} \
    -r 1 \
    --densification_interval 100 \
    --densify_until_iter 60000 \
    --densify_from_iter 500 \
    --densify_grad_threshold 0.0002 \
    --opacity_reset_interval 5000 \
    --optimizer_type default \
    --iterations 120000 \
    --lambda_dssim 0.4 \
    --mult 0.6 \
    --loss_thresh 0.05 \
    --grad_abs_thresh 0.0003 \
    --highfeature_lr 0.025 \
    --percent_dense 0.003 \
    --freq_weight_alpha 0.5 \
    --use_ms_ssim \
    --progressive_prune_interval 5000 \
    --progressive_prune_ratio 0.05 \
    --use_semantic_weight \
    --semantic_equipment_weight 3.0 \
    --semantic_background_weight 0.5 \
    --densify_equipment_factor 1.0 \
    --test_iterations 7000 15000 30000 50000 80000 120000 \
    --save_iterations 7000 30000 50000 80000 120000 \
    --data_device cpu \
    2>&1 | tee ${OUTPUT_PATH}/train.log
