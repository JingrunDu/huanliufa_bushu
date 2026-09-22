# 项目需拥有下面队列的使用权限；默认沿用训练项目。
NEBULA_PROJECT=${NEBULA_PROJECT:-embodied_ai_simulation_collaboration}
QUEUE=${QUEUE:-future_data_d_4090_2}
ALGO_NAME=${ALGO_NAME:-pytorch280}
# auto：选择 OSS 下已保存模型的最新训练目录。
MODEL_PATH=${MODEL_PATH:-auto}
# 可填写 OSS 上预编译的 SIBR_gaussianViewer_app；auto 在 Ubuntu / Alibaba Cloud Linux 节点尝试编译。
SIBR_BIN=${SIBR_BIN:-auto}
OSS_CREDS_FILE=${OSS_CREDS_FILE:-/data/oss_bucket_0/beifen/home/dujingrun.djr/code/textgs/render/oss_creds.env}
