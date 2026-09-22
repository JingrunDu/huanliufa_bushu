"""W&B 凭据解析；不输出或记录密钥。"""
import json
import netrc
import os
from pathlib import Path
from urllib.parse import urlparse


def api_key():
    if os.environ.get('WANDB_API_KEY'):
        return os.environ['WANDB_API_KEY']
    auth_file = os.environ.get('WANDB_AUTH_FILE')
    if auth_file:
        return json.loads(Path(auth_file).read_text())['api_key']
    try:
        host = urlparse(os.environ.get('WANDB_BASE_URL', 'https://api.wandb.ai')).hostname
        credentials = netrc.netrc().authenticators(host)
        return credentials[2] if credentials else None
    except (OSError, netrc.NetrcParseError):
        return None
