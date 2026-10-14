#!/bin/bash
# Experiments run sequentially (CPU). Set DATA_DIR / CACHE_DIR before running.
cd "$(dirname "$0")/src"
python train.py --model resnet18 --size 64 --epochs 5 --tag resnet18_64
python train.py --model smallcnn --size 64 --epochs 5 --no-pretrained --lr 3e-3 --tag smallcnn_64
