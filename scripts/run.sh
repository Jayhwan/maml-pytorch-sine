#!/bin/bash


for steps in 1 5; do
    for k_val in 5 10; do
        for seed in {0..4}; do
            # create the experiment ID
            exp_id="step_$steps/K_$k_val/seed_$seed"

            # run main.py with the specified experiment settings and exp_id
            python main.py --algo_name maml --inner_steps $steps --K $k_val --seed $seed --exp_id ${exp_id} &
            python main.py --algo_name tr_maml --inner_steps $steps --K $k_val --seed $seed --exp_id ${exp_id} &
            python main.py --algo_name taro_maml --inner_steps $steps --K $k_val --seed $seed --exp_id ${exp_id} &
            python main.py --algo_name vmaml --inner_steps $steps --K $k_val --seed $seed --exp_id ${exp_id} &
            wait
        done
    done
done

