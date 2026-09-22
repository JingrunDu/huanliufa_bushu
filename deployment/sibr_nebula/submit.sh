#!/usr/bin/env bash
set -euo pipefail
set +x
cd "$(dirname "$0")"
source ./config.sh
source "$OSS_CREDS_FILE"
: "${FMC_AK:?}" "${FMC_SK:?}" "${FMC_ENDPOINT:?}"
endpoint=${FMC_ENDPOINT#https://}; endpoint=${endpoint#http://}; endpoint=${endpoint%/}
job=sibr_$(date +%Y%m%d_%H%M%S)
for value in "$MODEL_PATH" "$SIBR_BIN"; do
 [[ "$value" != *[[:space:],]* ]] || { echo '配置路径不能含空格或逗号'; exit 1; }
done
args=(nebulactl run mdl "--nebula_project=$NEBULA_PROJECT" "--queue=$QUEUE"
 --entry=viewer_node.py "--algo_name=$ALGO_NAME" --worker_count=1
 "--user_params=--run-name=$job"
 --file.cluster_file=./cluster.json "--job_name=$job" --max_failover_times=0
 "--env=PYTHONUNBUFFERED=1,MODEL_PATH=$MODEL_PATH,SIBR_BIN=$SIBR_BIN,VIEWER_JOB=$job"
 "--oss_access_id=$FMC_AK" "--oss_access_key=$FMC_SK" --oss_bucket=fmc
 "--oss_endpoint=$endpoint" --enable_oss_cache=true '--ignore=__pycache__/*,.git/*')
echo "项目=$NEBULA_PROJECT 队列=$QUEUE 模型=$MODEL_PATH"
if [[ ${1:-} == --dry-run ]]; then
 for arg in "${args[@]}"; do
  case "$arg" in --oss_access_id=*|--oss_access_key=*) printf '%s ' "${arg%%=*}=<redacted>";; *) printf '%q ' "$arg";; esac
 done
 printf '\n'; exit 0
fi
# 只打包工作文件，避免把凭据文件放进代码包。
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp viewer_node.py worker.sh cluster.json "$stage/"
cd "$stage"
"${args[@]}"
