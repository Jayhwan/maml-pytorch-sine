import os
from os.path import join as pjoin
import random
import argparse
from datetime import datetime

import yaml
import torch
import numpy as np

from models.MAML_higher import MAML

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sine")
    parser.add_argument("--algo_name", type=str, default="maml_higher")

    # Hyperparameters
    parser.add_argument("--inner_lr", type=float, default=0.001)
    parser.add_argument("--meta_lr", type=float, default=0.001)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=70000)
    parser.add_argument("--num_eval_tasks", type=int, default=5000)
    parser.add_argument("--num_plot_tasks", type=int, default=5)

    # Seed
    parser.add_argument("--seed", default=0)
    
    # Log dir
    parser.add_argument("--logdir", type=str, default="results")
    parser.add_argument("--exp_id", type=str, default="debug")

    
    args = parser.parse_args()

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    results_path = pjoin(args.logdir, args.task_name, args.algo_name, args.exp_id, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    if not os.path.isdir(results_path):
        os.makedirs(results_path)
        
    config = {}
    for key, value in vars(args).items():
        config[key] = value

    with open(pjoin(results_path, "config.yaml"), 'w') as f:
        yaml.dump(config, f)

    if args.algo_name == "maml_higher":
        algo = MAML(args.task_name, inner_lr=args.inner_lr, meta_lr=args.meta_lr, K=args.K, inner_steps=args.inner_steps, results_path=results_path)
    # elif args.algo_name == "tr_maml":
    #     algo = TRMAML(args.task_name, inner_lr=args.inner_lr, meta_lr=args.meta_lr, K=args.K, inner_steps=args.inner_steps, results_path=results_path)
    # elif args.algo_name == "taro_maml":
    #     p_lr = 0.0001
    #     algo = TaroMAML(args.task_name, inner_lr=args.inner_lr, meta_lr=args.meta_lr, K=args.K, inner_steps=args.inner_steps, results_path=results_path,
    #                     p_lr=p_lr)
    # elif args.algo_name == "vmaml":
    #     radius = 0.05
    #     num_ve_iterations = args.inner_steps
    #     algo = VMAML(args.task_name, inner_lr=args.inner_lr, meta_lr=args.meta_lr, K=args.K, inner_steps=1, results_path=results_path,
    #                  radius=radius, num_ve_iterations=num_ve_iterations)
    # elif args.algo_name == "imaml":
    #     cg_steps = 5
    #     lam = 2.0
    #     algo = iMAML(args.task_name, inner_lr=args.inner_lr, meta_lr=args.meta_lr, K=args.K, inner_steps=args.inner_steps, results_path=results_path,
    #                  cg_steps=cg_steps, lam=lam)
    else:
        raise NotImplementedError
    
    algo.train(args.num_iterations)
    algo.evaluate(args.num_eval_tasks, K=args.K, n_steps=args.inner_steps, lr=args.inner_lr)
    algo.plot(args.num_plot_tasks, K=args.K, lr=args.inner_lr)
