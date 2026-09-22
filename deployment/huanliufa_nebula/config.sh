# H20（项目与队列成对切换）
NEBULA_PROJECT=${NEBULA_PROJECT:-embodied_ai_simulation_collaboration}
# QUEUE=${QUEUE:-embodied_ai_h20_na130}
QUEUE=${QUEUE:-embodied_ai_h20}



# 备用：4090
# NEBULA_PROJECT=${NEBULA_PROJECT:-aidata}
# QUEUE=${QUEUE:-aidata_tmp_4090}

# 运行配置（环境包下载并解压到 /tmp/fmc/fastgs）
# 训练迭代数、增密与加载参数见 training_config.sh。
ALGO_NAME=${ALGO_NAME:-pytorch280}
JOB_NAME=${JOB_NAME:-huanliufa_$(date +%Y%m%d_%H%M%S)}
PROGRESS_SECONDS=${PROGRESS_SECONDS:-30}  # 进度日志刷新间隔
CLUSTER_FILE=${CLUSTER_FILE:-./cluster.json}

# OSS
ENV_ARCHIVE_REL=${ENV_ARCHIVE_REL:-beifen/data/3dgs_bushu/env/fastgs_py310_v1.tar}
OSS_CREDS_FILE=${OSS_CREDS_FILE:-/data/oss_bucket_0/beifen/home/dujingrun.djr/code/textgs/render/oss_creds.env}
ENABLE_OSS_CACHE=${ENABLE_OSS_CACHE:-true}
DISABLE_OSS_META_CACHE=${DISABLE_OSS_META_CACHE:-false}
OSS_APPENDABLE=${OSS_APPENDABLE:-true}  # TensorBoard 需要

# W&B
WANDB_ENTITY=${WANDB_ENTITY:-jingrundu-tsinghua-university}
WANDB_PROJECT=${WANDB_PROJECT:-huanliufa_bushu}
WANDB_MODE=${WANDB_MODE:-online}  # online / offline / disabled
WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
