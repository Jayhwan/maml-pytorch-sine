from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data.sine import SINE
from models.building_blocks import MLP


class MAML():
    def __init__(self, task_name, inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=25):
        # Construct Model
        if task_name == "sine":
            self.task = SINE()
            self.model = MLP()
        else:
            raise NotImplementedError
        
        # Set Optimizer
        self.weights = list(self.model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = optim.Adam(self.weights, meta_lr)

        # Hyperparamters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.plot_every = 10
        self.print_every = 500
        self.meta_losses = []

    def inner_loop(self, X, y):
        X_train, X_test = X[:self.K], X[self.K:]
        y_train, y_test = y[:self.K], y[self.K:]

        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]
        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X_train, temp_weights), y_train)
            
            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        loss = self.criterion(self.model.parameterised(X_test, temp_weights), y_test) / self.K
        return loss
    
    def train(self, num_iterations):
        epoch_loss = 0.0
        for iteration in range(1, num_iterations+1):

            # compute meta loss
            meta_loss = 0
            batch_X, batch_y = self.task.sample_data(batch_size=self.tasks_per_meta_batch,
                                                     num_samples=2*self.K, mode="train")
            for i in range(self.tasks_per_meta_batch):
                meta_loss += self.inner_loop(batch_X[i], batch_y[i])
            
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))

            # if iteration % self.plot_every == 0:
            #     self.meta_losses.append(epoch_loss / self.plot_every)
            #     f1.write(str(epoch_loss / self.plot_every) + '\n')
            #     epoch_loss = 0

    def evaluate(self, num_tasks, K=5, n_steps=5, lr=0.01):
        losses = []

        X, y = self.task.sample_data(num_tasks, 2*K, mode="test")
        for i in range(num_tasks):
            model = deepcopy(self.model)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for step in range(n_steps):
                optimizer.zero_grad()
                loss = self.criterion(model(X[i, :K]), y[i, :K])
                loss.backward()
                optimizer.step()
            
            losses.append(self.criterion(model(X[i, :K]), y[i, :K]).item())

        losses = np.array(losses)
        results = {}
        results["mse_loss_avg"] = np.mean(losses)
        results["mse_loss_worst"] = np.max(losses)
        results["mse_loss_std"] = np.std(losses)
        results["mse_loss_90percentile"] = np.mean(np.sort(losses)[:int(0.9*num_tasks)])
        return results
