#! /bin/sh


### PEG experiments
#horizon=1
#seed=0
#gpu=1
#
#for fold in 0 1 2 3 4
#do
#  /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 14_measure_runtimes_peg.py "ConvT" "NLL" $fold $horizon $seed $gpu
#  #/local/home/ansass/anaconda3/envs/forecast/bin/python3.9 14_measure_runtimes_peg.py "ConvT" "NLLPEGSurface" $fold $horizon $seed $gpu
#  #/local/home/ansass/anaconda3/envs/forecast/bin/python3.9 14_measure_runtimes_peg.py "LSTM" "NLL" $fold $horizon $seed $gpu
#  #/local/home/ansass/anaconda3/envs/forecast/bin/python3.9 14_measure_runtimes_peg.py "LSTM" "NLLPEGSurface" $fold $horizon $seed $gpu
#done


### GDU experiments
split_attribute="treatment"
fold=2
horizon=1
gpu=0

#models="LSTM Ensemble"
#for model in $models
#do
#  for seed in 0 1 2 3 4
#  do
#    /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 14_measure_runtimes_gdu.py $split_attribute $fold $horizon "$model" "None" "None" "NLL" $seed $gpu
#  done
#done

model="Layer"
loss="LayerLoss"

for seed in 0 1 2 3 4
do
  /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 14_measure_runtimes_gdu.py $split_attribute $fold $horizon $model "FT" "CS" $loss $seed $gpu
  /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 14_measure_runtimes_gdu.py $split_attribute $fold $horizon $model "FT" "MMD" $loss $seed $gpu
done

#training_types="FT E2E"
#similarity_measures="MMD CS"
#for training_type in $training_types
#do
#  for similarity_measure in $similarity_measures
#  do
#    for seed in 0 1 2 3 4
#    do
#      /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 14_measure_runtimes_gdu.py $split_attribute $fold $horizon $model "$training_type" "$similarity_measure" $loss $seed $gpu
#    done
#  done
#done
