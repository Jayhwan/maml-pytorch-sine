from copy import deepcopy
from os.path import join as pjoin

import higher
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data.sine import SINE
from utils import simplex_proj, get_logger, cg_solve
from models.building_blocks import MLP


class MAML():
    def __init__(self, task_name, inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=25, results_path="./results"):
        # Construct Model
        if task_name == "sine":
            self.task = SINE()
            self.model = MLP()
        else:
            raise NotImplementedError
        
        # Set Optimizer
        self.criterion = nn.MSELoss()
        self.meta_optimizer = optim.Adam(self.model.parameters(), meta_lr)

        # Hyperparamters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.plot_every = 10
        self.print_every = 500
        
        # logdir
        self.results_path = results_path
        log_path = pjoin(results_path, "execution.log")
        self.logger = get_logger(log_path)
    
    def train_log(self, epoch_loss, iteration, num_iterations):
        epoch_loss = 0.5 * np.array(epoch_loss)
        self.logger.info(f"{iteration}/{num_iterations}")
        self.logger.info(f"MSE(mean): {np.mean(epoch_loss):.4f}\tMSE(worst): {np.max(epoch_loss):.4f}")
        self.logger.info(f"MSE(std): {np.std(epoch_loss):.4f}\tMSE(Top 90%): {np.mean(np.sort(epoch_loss)[:int(0.9*len(epoch_loss))]):.4f}")
    
    def train(self, num_iterations):
        losses = []
        for iteration in range(1, num_iterations+1):
            
            X, y = self.task.sample_data(batch_size=self.tasks_per_meta_batch,
                                                     num_samples=2*self.K, mode="train")
            
            X_train, X_test = X[:, :self.K], X[:, self.K:]
            y_train, y_test = y[:, :self.K], y[:, self.K:]
            
            self.meta_optimizer.zero_grad()
            inner_optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
            for i in range(self.tasks_per_meta_batch):
                with higher.innerloop_ctx(self.model, inner_optimizer) as (fnet, diffopt):
                    for _ in range(self.inner_steps):
                        loss = self.criterion(fnet(X_train[i]), y_train[i])
                        diffopt.step(loss)

                    loss = self.criterion(fnet(X_test[i]), y_test[i])
                    loss.backward()
                    losses.append(loss.item())
                    
            self.meta_optimizer.step()

            # log metrics
            if iteration % self.print_every == 0:
                self.train_log(losses, iteration, num_iterations)
                losses = []

    def evaluate(self, num_tasks, K=5, n_steps=5, lr=0.001):
        losses = []

        # test_loss = 0.0
        X, y = self.task.sample_data(num_tasks, 2*K, mode="test")

        X_train, X_test = X[:, :self.K], X[:, self.K:]
        y_train, y_test = y[:, :self.K], y[:, self.K:]
        
        for i in range(num_tasks):
            model = deepcopy(self.model)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for step in range(n_steps):
                optimizer.zero_grad()
                loss = self.criterion(model(X_train), y_train)
                loss.backward()
                optimizer.step()
            
            losses.append(self.criterion(model(X_test), y_test).item())

        losses = 0.5 * np.array(losses)
        results = {}
        results["mse_loss_avg"] = np.mean(losses)
        results["mse_loss_worst"] = np.max(losses)
        results["mse_loss_std"] = np.std(losses)
        results["mse_loss_90percentile"] = np.mean(np.sort(losses)[:int(0.9*len(losses))])
        for k, v in results.items():
            self.logger.info(f"{k}:\t{v:.2f}")
        np.save(f"{self.results_path}/performance.npy", losses)
        torch.save(self.model.state_dict(), f"{self.results_path}/{self}.pt")
        
    def __str__(self):
        return "MAML_higher"
