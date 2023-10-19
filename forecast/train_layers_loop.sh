#! /bin/sh

gpu=0


#training_type="E2E"
#similarity_measure="MMD"

#for seed in 0 1 2 3 4
#do
#    /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer.py 2 $seed "FT" "MMD" $gpu
#done



#for horizon in 0.5 1 2
#do
#    for seed in 0 1 2 3 4
#    do
#        /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer.py $horizon $seed "E2E" "MMD" $gpu
#    done
#done





/local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer.py 2 4 "FT" "MMD" $gpu

#measures="MMD CS"
#training_types="FT E2E"
#for horizon in 0.5 1 2
#do
#    for seed in 0 1 2 3 4
#    do
#        for training_type in $training_types
#        do
#            for similarity_measure in $measures
#            do
#                /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer.py $horizon $seed "$training_type" "$similarity_measure" $gpu
#            done
#        done
#    done
#done
