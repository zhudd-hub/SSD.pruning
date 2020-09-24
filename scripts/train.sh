#!/bin/sh
cd ..
# export NGPUS=3
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --config-file configs/mbv2_ssd_prune.yaml SOLVER.WARMUP_FACTOR 0.03333 SOLVER.WARMUP_ITERS 1000
python train.py --config-file configs/mbv2_ssd_prune.yaml --finetune

