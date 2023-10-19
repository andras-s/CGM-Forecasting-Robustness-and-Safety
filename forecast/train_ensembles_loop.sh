#! /bin/sh

gpu=0

for horizon in 0.5 1 2
do
    for seed in 0 1 2 3 4
    do
        /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_ensemble.py $horizon $seed $gpu
    done
done