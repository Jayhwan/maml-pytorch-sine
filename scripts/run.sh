#!/bin/bash


for steps in 5; do
# for steps in 1; do
    for k_val in 5 10; do
    # for k_val in 10; do
        for algo in "maml" "tr_maml" "taro_maml" "vmaml"; do
        # for algo in "vmaml"; do
            # create the experiment ID
            exp_id="step_$steps/K_$k_val"

            python main.py --algo_name $algo --inner_steps $steps --K $k_val --seed 0 --exp_id ${exp_id}/seed_0 &
            python main.py --algo_name $algo --inner_steps $steps --K $k_val --seed 1 --exp_id ${exp_id}/seed_1 &
            python main.py --algo_name $algo --inner_steps $steps --K $k_val --seed 2 --exp_id ${exp_id}/seed_2 &
            python main.py --algo_name $algo --inner_steps $steps --K $k_val --seed 3 --exp_id ${exp_id}/seed_3 &
            python main.py --algo_name $algo --inner_steps $steps --K $k_val --seed 4 --exp_id ${exp_id}/seed_4 &
            wait
        done
    done
done

