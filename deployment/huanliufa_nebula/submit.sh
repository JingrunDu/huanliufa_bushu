#!/usr/bin/env bash
set -euo pipefail
set +x  # 不输出包含密钥的 shell trace
cd "$(dirname "$0")"

source ./config.sh
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_BASE_URL
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo '无效的 WANDB_MODE' >&2; exit 1 ;; esac
OSS_MOUNT=/data/oss_bucket_0
if [[ -z ${FMC_AK:-} || -z ${FMC_SK:-} || -z ${FMC_ENDPOINT:-} ]]; then
  if [[ ! -r "$OSS_CREDS_FILE" ]]; then
    echo "无法读取 $OSS_CREDS_FILE；请设置 OSS_CREDS_FILE 或 FMC_AK、FMC_SK、FMC_ENDPOINT。" >&2
    exit 1
  fi
  source "$OSS_CREDS_FILE"
fi
: "${FMC_AK:?缺少 FMC_AK}"
: "${FMC_SK:?缺少 FMC_SK}"
: "${FMC_ENDPOINT:?缺少 FMC_ENDPOINT}"
# 平台要求生产 endpoint 域名，不带协议前缀。
MOUNT_ENDPOINT=${FMC_ENDPOINT#https://}
MOUNT_ENDPOINT=${MOUNT_ENDPOINT#http://}
MOUNT_ENDPOINT=${MOUNT_ENDPOINT%/}
if [[ -z "$MOUNT_ENDPOINT" || "$MOUNT_ENDPOINT" == *[/:,[:space:]]* || "$FMC_AK" == *,* || "$FMC_SK" == *,* ]]; then
  echo "请提供单个生产 endpoint 域名及对应凭据；本脚本只挂载 fmc。" >&2
  exit 1
fi
for flag in "$ENABLE_OSS_CACHE" "$DISABLE_OSS_META_CACHE" "$OSS_APPENDABLE"; do
  [[ "$flag" == true || "$flag" == false ]] || { echo '缓存开关必须为 true 或 false' >&2; exit 1; }
done
ENVS="PYTHONUNBUFFERED=1,OSS_MOUNT=$OSS_MOUNT,PROGRESS_SECONDS=$PROGRESS_SECONDS,ENV_ARCHIVE_REL=$ENV_ARCHIVE_REL,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,WANDB_MODE=$WANDB_MODE,WANDB_BASE_URL=$WANDB_BASE_URL"

ARGS=(nebulactl run mdl
  "--nebula_project=$NEBULA_PROJECT"
  "--queue=$QUEUE"
  --entry=train_node.py
  "--algo_name=$ALGO_NAME"
  --worker_count=1
  "--user_params=--run-name=$JOB_NAME"
  "--file.cluster_file=$CLUSTER_FILE"
  "--job_name=$JOB_NAME"
  '--ignore=__pycache__/*,.git/*,tools/*'
  "--env=$ENVS"
  "--oss_access_id=$FMC_AK"
  "--oss_access_key=$FMC_SK"
  --oss_bucket=fmc
  "--oss_endpoint=$MOUNT_ENDPOINT"
  "--enable_oss_cache=$ENABLE_OSS_CACHE"
  "--disable_oss_meta_cache=$DISABLE_OSS_META_CACHE"
  "--oss_appendable=$OSS_APPENDABLE"
  --max_failover_times=0)

echo "队列: $QUEUE; 项目: $NEBULA_PROJECT; 单 worker / 单 GPU"
echo "任务挂载: oss://fmc/ → $OSS_MOUNT；读取缓存: $ENABLE_OSS_CACHE"
echo "输出: oss://fmc/beifen/data/3dgs_bushu/$JOB_NAME/"
if [[ ${1:-} == --dry-run ]]; then
  for arg in "${ARGS[@]}"; do
    case "$arg" in
      --oss_access_id=*) printf '%s ' '--oss_access_id=<redacted>' ;;
      --oss_access_key=*) printf '%s ' '--oss_access_key=<redacted>' ;;
      *) printf '%q ' "$arg" ;;
    esac
  done
  printf '\n'
  exit 0
fi
command -v nebulactl >/dev/null
python3 ./submit_with_auth.py "${ARGS[@]}"
