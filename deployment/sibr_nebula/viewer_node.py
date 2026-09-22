"""星云 Python 入口，保留 shell 原始退出码。"""
import os
import argparse
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument('--run-name', required=True)
args = parser.parse_args()
os.environ['VIEWER_JOB'] = args.run_name
os.execvp('bash', ['bash', str(Path(__file__).with_name('worker.sh'))])
