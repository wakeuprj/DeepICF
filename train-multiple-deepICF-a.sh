#!/usr/bin/env bash
# sh train-multiple-deepICF-a.sh 5 |& tee Stability-Models-DeepICF-a/logs-ray04
if [ ! -d "Stability-Models-DeepICF-a" ]; then
  mkdir Stability-Models-DeepICF-a
fi
for i in $(seq 0 $1)
do
  python3 DeepICF_a.py --path Data/ --dataset ml-1m --epochs 20 --beta 0.8 --weight_size 16 --activation 0 --algorithm 0 --verbose 1 --batch_choice user --embed_size 16 --layers [64,32,16] --regs [1e-06,1e-06,1e-06] --reg_W [10,10,10,10] --alpha 0 --train_loss 1 --num_neg 4 --lr 0.01 --batch_norm 1 --pretrain 0
done

