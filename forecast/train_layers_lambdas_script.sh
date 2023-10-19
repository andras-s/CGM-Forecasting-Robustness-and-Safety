#! /bin/sh
seed=4
gpu=3

# measures="MMD CS"
# for similarity_measure in $measures
# do
#     for lambda_OLS in 0 0.01 0.1 1 10
#     do
#         for lambda_l1 in 0 0.01 0.1 1 10
#         do
#             /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer_sm_lambdas_seed_gpu.py "$similarity_measure" $lambda_OLS $lambda_l1 $seed $gpu
#         done
#     done
# done


# MMD
#/local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer_sm_lambdas_seed_gpu.py "MMD" 0.01 0.1 $seed $gpu
#/local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer_sm_lambdas_seed_gpu.py "MMD" 0.01 1 $seed $gpu
#/local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer_sm_lambdas_seed_gpu.py "MMD" 0.01 10 $seed $gpu
#for lambda_OLS in 0.1 1 10
#do
#    for lambda_l1 in 0 0.01 0.1 1 10
#    do
#        /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer_sm_lambdas_seed_gpu.py "MMD" $lambda_OLS $lambda_l1 $seed $gpu
#    done
#done

# CS
# for lambda_l1 in 0 0.01 0.1 1 10
# do
#     /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer_sm_lambdas_seed_gpu.py "CS" 0 $lambda_l1 $seed $gpu
# done

/local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer_sm_lambdas_seed_gpu.py "CS" 10 10 $seed $gpu
# /local/home/ansass/anaconda3/envs/forecast/bin/python3.9 train_layer_sm_lambdas_seed_gpu.py "CS" 0.01 0.01 $seed $gpu
