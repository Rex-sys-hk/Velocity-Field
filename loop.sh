#!/bin/bash
for i in {1..10000}
do
    echo "$i /10000"
    # python train.py --name DIPP --train_set ./processed --valid_set ./validate --use_planning --seed 3407 --num_workers 92 --pretrain_epochs 5 --train_epochs 20 --batch_size 184 --learning_rate 2e-4 --device cuda
    python train.py --name DIPP --train_set /mnt/processed --valid_set /mnt/validate --use_planning --seed 3407 --num_workers 90 --pretrain_epochs 5 --train_epochs 20 --batch_size 180 --learning_rate 2e-4 --device cuda
done