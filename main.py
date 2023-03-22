import random
import argparse

import torch
import numpy as np

from models.MAML import MAML

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sine")
    parser.add_argument("--algo_name", type=str, default="maml")

    # Hyperparameters
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument("--meta_lr", type=float, default=0.001)
    parser.add_argument("--num_iterations", type=int, default=10000)
    parser.add_argument("--num_eval_tasks", type=int, default=100)

    # Seed
    parser.add_argument("--seed", default=0)

    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.algo_name == "maml":
        algo = MAML(args.task_name, inner_lr=args.inner_lr, meta_lr=args.meta_lr)
    else:
        raise NotImplementedError
    
    algo.train(args.num_iterations)
    result = algo.evaluate(args.num_eval_tasks)

    for k, v in result.items():
        print(f"{k}:\t{v:.2f}")
