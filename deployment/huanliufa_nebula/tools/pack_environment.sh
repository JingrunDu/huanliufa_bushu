#!/usr/bin/env bash
set -euo pipefail
set +x
cd "$(dirname "$0")"
if [[ -z ${FMC_AK:-} || -z ${FMC_SK:-} || -z ${FMC_ENDPOINT:-} ]]; then
    set -a
    source "${OSS_CREDS_FILE:-/data/oss_bucket_0/beifen/home/dujingrun.djr/code/textgs/render/oss_creds.env}"
    set +a
fi
export FMC_AK FMC_SK FMC_ENDPOINT
exec python -u pack_environment.py "$@"
