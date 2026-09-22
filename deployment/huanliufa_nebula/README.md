# FastGS 星云提交（直接读取 OSS 挂载）

提交：`bash submit.sh`。查看命令而不提交：`bash submit.sh --dry-run`。

日常修改 `config.sh`，集中配置项目、队列、镜像、任务名、进度刷新间隔、
OSS 凭据文件位置及缓存开关；资源数量仍在 `cluster.json` 中修改。
训练参数和加载选项在 `training_config.sh` 修改，入口为 `train_6650.sh`。
`config.sh` 已保存两组参考配置：默认 H20 项目/队列，以及注释掉的
`aidata` / `aidata_tmp_4090`。切换时先注释当前组的两行，再取消备用组两行的注释，
只保留一组有效配置。备用配置来自参考脚本，不代表当前账号一定有访问权限。
命令行环境变量仍可临时覆盖配置文件。

默认沿用 textgs/render/run_check.sh 的 H20 队列、项目和 pytorch280 镜像，
申请单 worker、1 GPU、16 CPU、128000 MB 内存。队列权限和资源可用性由星云决定。
当前训练代码不是分布式实现，不应增加 worker_count。

前置条件：

- 已配置 ~/.nebulactl/config.ini。
- 脚本通过 oss_access_id、oss_access_key、oss_bucket、oss_endpoint 配置挂载，
  平台将 oss://fmc/ 固定挂载到 /data/oss_bucket_0。
  默认读取参考目录 textgs/render/oss_creds.env 中的 FMC_AK、FMC_SK、FMC_ENDPOINT。
  可用 OSS_CREDS_FILE 指定其他凭据文件，或预先设置这三个环境变量。
  凭据文件不复制到提交目录；dry-run 隐藏 AK/SK。
- 使用生产 endpoint，脚本自动去掉 http:// 或 https:// 前缀。
  默认 enable_oss_cache=true（平台缓存占用少量离线额度）、
  disable_oss_meta_cache=false、oss_appendable=true（支持 TensorBoard）。
  可设置 ENABLE_OSS_CACHE=false 关闭读取缓存；设置
  DISABLE_OSS_META_CACHE=true 可提高写一致性，但会降低读取性能。
- worker 将单个 OSS 环境 tar 包传输到本地，验证 SHA-256 后解压到
  /tmp/fmc/fastgs，恢复执行权限，
  使用其中的 Python 3.10。缺失的辅助 Python 包会通过 pip 安装，
  PyTorch 和 CUDA 扩展沿用备份。启动时验证 GPU 和扩展兼容性；
  尚未实际提交星云验证。环境包传输每 15 秒显示百分比、GiB、平均速度和 ETA，
  本地解压每 30 秒报告已用时间。同一 SHA-256 的本地环境直接复用。

可覆盖配置示例：

```bash
QUEUE=embodied_ai_h20_na130 \
NEBULA_PROJECT=embodied_ai_simulation_collaboration \
PROGRESS_SECONDS=30 \
bash submit.sh
```

训练代码、图像、遮罩直接从 OSS 读取，只有运行环境复制到节点本地。
提交目录中的入口与进度包装脚本由 nebulactl 打包。
输出直接写入 oss://fmc/beifen/data/3dgs_bushu/<job_name>/，包括 train.log、
loss_log.csv、cfg_args、cameras.json、input.ply、point_cloud/iteration_*/point_cloud.ply。
环境检查和初始化输出在星云 Logview；本入口不生成本地启动脚本的 launcher.log/PID。
OSS 挂载可能延迟刷新日志，以 Logview 的实时输出为准。

`run_training.py` 为原训练脚本的 tqdm 提供按行输出的进度条，每隔
PROGRESS_SECONDS 秒显示百分比、迭代数、训练已用时间、预计剩余时间、
迭代速度和损失，训练结束显示耗时。进度条从迭代初始化开始计时；
另行记录含数据初始化的训练进程耗时，以及包含环境准备的整个任务耗时。
预计剩余时间会随增密、剪枝、模型保存等操作变化。

训练公共参数参考 exp_6650/run_fastgs_a4_canonical.sh 和 run_final_guidedgs.sh：

| 配置 | 当前值 |
| --- | --- |
| 训练图像数 | 6650，全量，不开启 --eval |
| 总迭代数 | 6650 × 100 = 665000 |
| 增密间隔 | floor(6650 / 3) = 2216 |
| 增密开始 / 截止 | 6650 / 332500 |
| 不透明度重置间隔 | 99750 |
| 位置学习率衰减步数 | 665000，与本次训练长度一致 |
| densify_grad_threshold / grad_thresh | 0.0002 / 0.0002 |
| percent_dense / dense | 0.003 / 0.003，显式设置实际使用的 dense |
| grad_abs_thresh / highfeature_lr | 0.0003 / 0.025 |
| lambda_dssim / mult / loss_thresh | 0.4 / 0.6 / 0.05 |
| 部署版额外渐进不透明度剪枝 | 关闭，参考公共实验未使用 |
| 保存点 | 665000；原训练代码仍未启用评估或 checkpoint 保存 |

与参考论文实验不同：不使用 5985/665 划分，保留部署版本的语义权重
3.0/0.5、频率加权 0.5 和多尺度 SSIM；位置学习率调度显式延长至训练总长度。
优化器仍保持参考实现的 15000/20000 阶段边界和 16/32/64 更新间隔。
正常情况下每 6650 次迭代遍历一遍全部相机，665000 次即 100 遍；
原代码仍会跳过没有可见高斯的迭代，这类视角被选中不代表产生了梯度。
输出额外保存 training_config.sh 和实际 training_command.txt。

解决 Loading Training Cameras 无进度的问题：camera_loading.py 对原加载入口
做运行时替换，8 路并行读图和准备遮罩，CUDA Camera 在主线程创建，保持列表顺序。
每 PROGRESS_SECONDS 秒显示完成张数、耗时、ETA、速度、RSS 和等待文件。
按最终参考实验使用硬人物遮罩（MASK_SIGMA=0，最近邻缩放和 0.5 阈值），
避免旧实现对每张图做 sigma=7 的 43×43 CPU 卷积；如需复现旧软遮罩，
设置 MASK_SIGMA=7，会使用数学等价的可分离高斯卷积。
不把图片整份下载到本地，图像张量仍缓存于 CPU 内存。
这些变更只作用于新版启动入口，不会修改或热更新已运行的旧任务。

## W&B 最小接入

config.sh 中默认配置：

```bash
WANDB_ENTITY=jingrundu-tsinghua-university
WANDB_PROJECT=huanliufa_bushu
WANDB_MODE=online
WANDB_BASE_URL=https://api.wandb.ai
```

提交命令不变：`bash submit.sh`。训练开始前自动检查/安装 wandb==0.25.0，
run_training.py 创建与输出目录同名的 Run，正常结束或异常退出时完成 Run。
训练曲线只记录总损失 `train/loss`，横轴 `train/iteration`，每 100 次迭代记录一次。
直接复用已有 loss_log.csv 中的标量，不增加 GPU 同步、损失计算或评估渲染。
关闭 W&B 自动系统指标采样；保留任务信息、控制台日志和运行生命周期标记。
原有 CSV 保持不变，不上传模型 PLY。W&B 仍有少量后台记录/上传开销。
日志打印 W&B Run URL，并将运行信息写入 OSS 输出目录的 wandb_run.json。

认证优先读取提交端 WANDB_API_KEY，否则使用 ~/.netrc 中对应服务的登录凭据。
submit_with_auth.py 在权限为 700 的临时目录中创建提交包，凭据文件权限为 600；
通过任务代码包中的 .wandb_auth.json 传递给 worker，不写入 --env 参数。
提交完成后清理本地临时包。不要把该私有任务代码包分享给无权限的人员。
如果尚未登录，执行 `/tmp/fmc/fastgs/bin/python -m wandb login`。

在线初始化超时/失败时尝试离线记录，SDK 安装失败时关闭 W&B 并继续训练。
CSV 和 train.log 保留。SDK 文件先写到 /tmp/huanliufa_wandb，正常结束流程中
备份到 OSS 输出目录的 wandb/；被强制杀死或节点丢失时无法保证离线文件已备份。
离线目录之后可使用 `wandb sync <offline-run目录>` 补传。

独立验证（不训练、不申请 GPU）：

```bash
/tmp/fmc/fastgs/bin/python /home/dujingrun.djr/huanliufa_nebula/wandb_smoke.py
```

本机已验证创建 Run、上传 3 条测试记录、正常结束及服务端读回。
验证 Run：https://wandb.ai/jingrundu-tsinghua-university/huanliufa_bushu/runs/xwlrvfgv
星云 worker 网络连通性尚未通过新任务实际验证。

环境包：`oss://fmc/beifen/data/3dgs_bushu/env/fastgs_py310_v1.tar`，
校验清单为同路径追加 `.json`。可在 config.sh 修改 ENV_ARCHIVE_REL。
仅当 tar 上传并校验完大小后才发布清单，启动脚本要求两者均存在。
环境恢复需为 tar 和解压后的环境同时保留本地空间；恢复结束清理临时 tar。
升级时保留已有环境为 fastgs.previous_*，不直接删除已有环境。

一次性制包脚本为 tools/pack_environment.py，使用参考凭据文件中的 FMC_*。
脚本通过 OSS SDK 并行读取原备份，在 /tmp/fastgs_env_package 制包，再上传 OSS；
已有完整清单时拒绝覆盖。同样的 tar 恢复方式已接入
/home/dujingrun.djr/start_huanliufa_training.sh。
