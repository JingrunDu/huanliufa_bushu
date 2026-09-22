# 环流法部署：FastGS 场景重建

包含 FastGS 训练代码、CUDA 扩展源码、星云提交脚本及本次实验结果。

## 目录

- `train.py`、`scene/`、`gaussian_renderer/`、`arguments/`、`utils/`：训练和渲染实现。
- `submodules/`：CUDA 扩展源码（保留上游许可证，未包含编译产物）。
- `deployment/huanliufa_nebula/`：环境恢复、星云训练提交、相机加载、W&B loss 记录及离线渲染脚本。
- `deployment/sibr_nebula/`：SIBR 远程桌面验证脚本。当前星云计算镜像缺少 NVIDIA Xorg 驱动，尚未打通。
- `result/`：`huanliufa_20260922_051004` 实验的最终高斯模型、参数、loss 和渲染结果。

## 获取大文件

点云和结果图片使用 Git LFS。克隆前安装 Git LFS，克隆后执行：

```bash
git lfs install
git lfs pull
```

## 训练

实际实验使用 OSS 环境包（Python 3.10、PyTorch 2.3.0+cu121），根目录 `environment.yml` 为原项目配置，与该环境包版本不同。

1. 在 `deployment/huanliufa_nebula/config.sh` 配置星云项目、队列及本地 OSS 凭据文件路径。
2. 数据需包含 COLMAP `sparse/0/`、`images/`、`mask_sam3/`、`semantic_weights/`。
3. 设置 W&B 登录凭据，或设 `WANDB_MODE=offline/disabled`。
4. 执行 `bash deployment/huanliufa_nebula/submit.sh`。

部署及渲染脚本保留实验机器上的绝对路径。迁移机器时需修改代码、数据、环境包与输出路径；提交脚本默认从 OSS 读取训练代码，不会自动将本仓库源码同步到 OSS。

数据及 14 GiB 环境包不入库。详细参数与流程见部署目录 README。

## 许可

保留原项目及各依赖许可证；部分源码带有上游研究/非商业用途条款，请按各文件及依赖许可证使用。
