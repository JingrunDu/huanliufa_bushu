# 最新训练结果

- 实验：huanliufa_20260922_051004
- 原始位置：oss://fmc/beifen/data/3dgs_bushu/huanliufa_20260922_051004/
- 训练图像：6650；迭代次数：665000；高斯数：3273550。
- 最终模型：`point_cloud/iteration_665000/point_cloud.ply`。
- `input.ply`：初始化点云，不是训练模型。
- `cameras.json`：相机信息；`cfg_args`：基本参数。
- `training_command.txt` / `training_config.sh`：实际训练命令及配置。
- `loss_log.csv`：每 100 次迭代的损失记录。
- `wandb_run.json`：实验运行信息（历史本地目录不保证仍可访问）。
- `render_comparison_100/`：100 个训练视角的 GT 与渲染对比，排除人物区域后平均逐图 PSNR 27.47 dB，并非独立测试集指标。
- `render_perturbed_100/`：偏航/俯仰 ±2°、平移为轨迹半径 0.5% 的扰动视角。
- `render_perturbed_large_100/`：偏航/俯仰 ±5°、平移为轨迹半径 1%，`montage_100.jpg` 为 100 张图的 10×10 拼图。

完整相机扰动保存在各目录 poses.json。扰动视角没有对应 GT，不计算 PSNR。COLMAP 坐标单位不等同于米。

PLY 保存高斯参数，不包含优化器状态，不是完整的断点续训 checkpoint。
