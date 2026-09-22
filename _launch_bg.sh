#!/bin/bash
export PATH=/tmp/fmc/multiworld/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
cd /home/dujingrun.djr/code/3dgs/huanliufa_bushu
bash train_perspective.sh
