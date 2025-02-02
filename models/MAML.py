from copy import deepcopy
from os.path import join as pjoin

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
        
        # logdir
        self.results_path = results_path
        log_path = pjoin(results_path, "execution.log")
        self.logger = get_logger(log_path)

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

        loss = self.criterion(self.model.parameterised(X_test, temp_weights), y_test)
        return loss
    
    def train_log(self, epoch_loss, iteration, num_iterations):
        epoch_loss = 0.5 * np.array(epoch_loss)
        self.logger.info(f"{iteration}/{num_iterations}")
        self.logger.info(f"MSE(mean): {np.mean(epoch_loss):.4f}\tMSE(worst): {np.max(epoch_loss):.4f}")
        self.logger.info(f"MSE(std): {np.std(epoch_loss):.4f}\tMSE(Top 90%): {np.mean(np.sort(epoch_loss)[:int(0.9*len(epoch_loss))]):.4f}")
        # print(f"{iteration}/{num_iterations}", end="\t")
        # print(f"MSE(mean): {np.mean(epoch_loss):.4f}\tMSE(worst): {np.max(epoch_loss):.4f}", end="\t")
        # print(f"MSE(std): {np.std(epoch_loss):.4f}\tMSE(Top 90%): {np.mean(np.sort(epoch_loss)[:int(0.9*self.tasks_per_meta_batch)]):.4f}")
    
    def train(self, num_iterations):
        losses = []
        for iteration in range(1, num_iterations+1):

            # compute meta loss
            meta_loss = 0.0
            batch_X, batch_y = self.task.sample_data(batch_size=self.tasks_per_meta_batch,
                                                     num_samples=2*self.K, mode="train")
            for i in range(self.tasks_per_meta_batch):
                loss = self.inner_loop(batch_X[i], batch_y[i])
                meta_loss += loss
                losses.append(loss.item())
            
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            # epoch_loss.append(meta_loss.item() / self.tasks_per_meta_batch)
            if iteration % self.print_every == 0:
                self.train_log(losses, iteration, num_iterations)
                losses = []

    def evaluate(self, num_tasks, K=5, n_steps=5, lr=0.001):
        losses = []

        # test_loss = 0.0
        X, y = self.task.sample_data(num_tasks, 2*K, mode="test")
        for i in range(num_tasks):
            model = deepcopy(self.model)
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for step in range(n_steps):
                optimizer.zero_grad()
                loss = self.criterion(model(X[i, :K]), y[i, :K])
                loss.backward()
                optimizer.step()
            
            losses.append(self.criterion(model(X[i, K:]), y[i, K:]).item())
            # test_loss += self.criterion(model(X[i, K:]), y[i, K:]).item()
            # if (i+1) % self.tasks_per_meta_batch == 0:
            #     losses.append(test_loss / self.tasks_per_meta_batch)
            #     test_loss = 0.0

        losses = 0.5 * np.array(losses)
        results = {}
        results["mse_loss_avg"] = np.mean(losses)
        results["mse_loss_worst"] = np.max(losses)
        results["mse_loss_std"] = np.std(losses)
        results["mse_loss_90percentile"] = np.mean(np.sort(losses)[:int(0.9*len(losses))])
        for k, v in results.items():
            # print(f"{k}:\t{v:.2f}")
            self.logger.info(f"{k}:\t{v:.2f}")
        np.save(f"{self.results_path}/performance.npy", losses)
        torch.save(self.model.state_dict(), f"{self.results_path}/{self}.pt")
    
    def plot(self, num_tasks, K=5, n_steps=5, lr=0.01):
        X, y = self.task.sample_data(batch_size=num_tasks, mode="plot")
        sampled_steps = [1, n_steps]
        
        for i in range(num_tasks):
            losses = []
            pred_ys = []
            idx = torch.randint(1000, size=(2*K, ))
            
            model = deepcopy(self.model)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            pred_ys.append(model(X[i]))
            
            for step in range(1, n_steps+1):
                optimizer.zero_grad()
                loss = self.criterion(model(X[i, idx[:K]]), y[i, idx[:K]])
                loss.backward()
                optimizer.step()
                
                if step in sampled_steps:
                    pred_ys.append(model(X[i]))
            
                losses.append(loss.item())

            plt.figure(figsize=(14.4, 4.8))
            
            # plot the model functions
            plt.subplot(1, 2, 1)
            
            plt.plot(X[i], y[i], '-', color=(0, 0, 1, 0.5), label='true function')
            plt.scatter(X[i, idx[:K]], y[i, idx[:K]], label='data')
            plt.plot(X[i], pred_ys[0].detach().numpy(), ':', color=(0.7, 0, 0, 1), label='initial weights')
            
            for j, step in enumerate(sampled_steps):
                plt.plot(X[i], pred_ys[j+1].detach().numpy(), 
                        '-.' if step == 1 else '-', color=(0.5, 0, 0, 1),
                        label='model after {} steps'.format(step))
                
            plt.legend(loc='lower right')
            plt.title(f"Model fit: {str(self)}")

            # plot losses
            plt.subplot(1, 2, 2)
            plt.plot(losses)
            plt.title("Loss over time")
            plt.xlabel("gradient steps taken")
            # plt.show()
            plt.savefig(f"{self.results_path}/sample_{str(self)}_{i}.png")
    
    def __str__(self):
        return "MAML"    


class TRMAML(MAML):
    def __init__(self, task_name, inner_lr, meta_lr, K=5, inner_steps=1, tasks_per_meta_batch=25, results_path="./results"):
        super().__init__(task_name, inner_lr, meta_lr, K, inner_steps, tasks_per_meta_batch, results_path)
        
    def train(self, num_iterations):
        losses = []
        for iteration in range(1, num_iterations+1):

            # compute meta loss
            # meta_loss = 0.0
            worst_meta_loss = 0.0  
            batch_X, batch_y = self.task.sample_data(batch_size=self.tasks_per_meta_batch,
                                                     num_samples=2*self.K, mode="train")
            for i in range(self.tasks_per_meta_batch):
                loss = self.inner_loop(batch_X[i], batch_y[i])
                # meta_loss += task_loss
                losses.append(loss.item())
                if loss > worst_meta_loss:
                    worst_meta_loss = loss
                # meta_loss += self.inner_loop(batch_X[i], batch_y[i])
            
            # compute meta gradient of loss with respect to maml weights 
            meta_grads = torch.autograd.grad(worst_meta_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            # epoch_loss.append(meta_loss.item() / self.tasks_per_meta_batch)
            if iteration % self.print_every == 0:
                self.train_log(losses, iteration, num_iterations)
                losses = []
                
    def __str__(self):
        return "TR-MAML"    


class TaroMAML(MAML):
    def __init__(self, task_name, inner_lr, meta_lr, p_lr=0.001, K=10, inner_steps=1, tasks_per_meta_batch=25, results_path="./results"):
        super().__init__(task_name, inner_lr, meta_lr, K, inner_steps, tasks_per_meta_batch, results_path)

        self.p_lr = p_lr
        
    def train(self, num_iterations):
        losses = []
        p = np.ones(self.tasks_per_meta_batch) / self.tasks_per_meta_batch
        for iteration in range(1, num_iterations+1):

            # compute meta loss
            meta_loss = 0.0
            p_meta_loss = 0.0
            batch_X, batch_y = self.task.sample_data(batch_size=self.tasks_per_meta_batch,
                                                     num_samples=2*self.K, mode="train")
            for i in range(self.tasks_per_meta_batch):
                loss = self.inner_loop(batch_X[i], batch_y[i])
                meta_loss += loss
                losses.append(loss.item())
                p_meta_loss += p[i] * loss
            p[i] += self.p_lr * p_meta_loss
            p = simplex_proj(p)
            
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(p_meta_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            # epoch_loss.append(meta_loss.item() / self.tasks_per_meta_batch)
            if iteration % self.print_every == 0:
                self.train_log(losses, iteration, num_iterations)
                losses = []
                
    def __str__(self):
        return "TaRo-BOBA"
    
    
class VMAML(MAML):
    def __init__(self, task_name, inner_lr, meta_lr, radius=0.05, num_ve_iterations=5,
                  K=10, inner_steps=1, tasks_per_meta_batch=25, results_path="./results"):
        super().__init__(task_name, inner_lr, meta_lr, K, inner_steps, tasks_per_meta_batch, results_path)

        self.radius = radius
        self.num_ve_iterations = num_ve_iterations
        
    def inner_loop(self, X, y, temp_weights, compute_loss=False):
        X_train, X_test = X[:self.K], X[self.K:]
        y_train, y_test = y[:self.K], y[self.K:]
    
        if compute_loss:
            loss = self.criterion(self.model.parameterised(X_test, temp_weights), y_test)
        else:
            for step in range(self.inner_steps):
                loss = self.criterion(self.model.parameterised(X_train, temp_weights), y_train)
                
                # compute grad and update inner loop weights
                grad = torch.autograd.grad(loss, temp_weights)
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
                        
            loss = 0.0
            
        return temp_weights, loss
        
    def train(self, num_iterations):
        losses = []
        
        for iteration in range(1, num_iterations+1):
            
            # compute meta loss
            meta_loss = 0.0
            task_weights_list = [[w.clone() for w in self.weights] for _ in range(self.tasks_per_meta_batch)]
            batch_X, batch_y = self.task.sample_data(batch_size=self.tasks_per_meta_batch,
                                                     num_samples=2*self.K, mode="train")

            for j in range(self.num_ve_iterations):
                dist_square = torch.tensor(0.)
                for i in range(self.tasks_per_meta_batch):
                    task_weights, _ = self.inner_loop(batch_X[i], batch_y[i], task_weights_list[i])
                    task_weights_list[i] = task_weights
                    dist_square += sum(list(map(lambda p: torch.sum(torch.square(p[1] - p[0])), zip(task_weights, self.weights))))
                
                d = torch.sqrt(dist_square)
                r = self.radius
                #print("Before Projection", j, dist_square, torch.sqrt(dist_square))
                
                if d > r:
                    for i in range(self.tasks_per_meta_batch):
                        task_weights_list[i] = list(map(lambda p: (r*p[0] + (d-r)*p[1])/d, zip(task_weights_list[i], self.weights)))
                
            for i in range(self.tasks_per_meta_batch):
                _, loss = self.inner_loop(batch_X[i], batch_y[i], task_weights_list[i], True)
                meta_loss += loss
                losses.append(loss.item())
            
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            
            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()
            
           # log metrics
            # epoch_loss.append(meta_loss.item() / self.tasks_per_meta_batch)
            if iteration % self.print_every == 0:
                self.train_log(losses, iteration, num_iterations)
                losses = []
    
    def __str__(self):
        return "VariMAML"
    
    
class iMAML(MAML):
    def __init__(self, task_name, inner_lr, meta_lr, K=5, inner_steps=1, cg_steps=1, lam=1, tasks_per_meta_batch=25, results_path="./results"):
        super().__init__(task_name, inner_lr, meta_lr, K, inner_steps, tasks_per_meta_batch, results_path)

        self.cg_steps = cg_steps
        self.lam = lam
        
    def inner_loop(self, X, y, temp_weights, compute_loss=False):
        X_train, X_test = X[:self.K], X[self.K:]
        y_train, y_test = y[:self.K], y[self.K:]
    
        if compute_loss:
            loss = self.criterion(self.model.parameterised(X_test, temp_weights), y_test)
        else:
            for step in range(self.inner_steps):
                loss = self.criterion(self.model.parameterised(X_train, temp_weights), y_train)
                regu_loss = 0.0
                for w in temp_weights:
                    regu_loss += 0.5 * self.lam * torch.sum(w ** 2)
                loss += regu_loss
                
                # compute grad and update inner loop weights
                grad = torch.autograd.grad(loss, temp_weights)
                temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]
                        
            loss = 0.0
            
        return temp_weights, loss
    
    def hessian_vector_product(self, vector, params=None, x=None, y=None):
        """
        Performs hessian vector product on the train set in task with the provided vector
        """
        if x is not None and y is not None:
            xt, yt = x[:self.K], y[:self.K]
        else:
            raise NotImplementedError
        if params is not None:
            self.set_params(params)
        tloss = self.criterion(self.model(xt), yt)
        # tloss = self.get_loss(xt, yt)
        
        grad_ft = torch.autograd.grad(tloss, self.model.parameters(), create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_ft])
        # vec = utils.to_device(vector, self.use_gpu)
        h = torch.sum(flat_grad * vector)
        hvp = torch.autograd.grad(h, self.model.parameters())
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat
    
    def matrix_evaluator(self, x, y, regu_coef=1.0, lam_damping=10.0):
        def evaluator(v):
            hvp = self.hessian_vector_product(v, x=x, y=y)
            Av = (1.0 + regu_coef) * v + hvp / (self.lam + lam_damping)
            return Av
        return evaluator
    
    def get_params(self):
        return torch.cat([param.data.view(-1) for param in self.model.parameters()], 0).clone()
    
    def set_params(self, param_vals):
        offset = 0
        for param in self.model.parameters():
            param.data.copy_(param_vals[offset:offset + param.nelement()].view(param.size()))
            offset += param.nelement()
            
    def train(self, num_iterations):
        losses = []
        
        for iteration in range(1, num_iterations+1):
            
            # compute meta loss
            meta_loss = 0.0
            task_weights_list = [[w.clone() for w in self.weights] for _ in range(self.tasks_per_meta_batch)]
            batch_X, batch_y = self.task.sample_data(batch_size=self.tasks_per_meta_batch,
                                                     num_samples=2*self.K, mode="train")

            meta_grad = 0.0
            for i in range(self.tasks_per_meta_batch):
                # Obtain task parameters using iterative optimization
                task_weights, _ = self.inner_loop(batch_X[i], batch_y[i], task_weights_list[i])
                task_weights_list[i] = task_weights
                
                # Compute partial out-level gradient
                _, loss = self.inner_loop(batch_X[i], batch_y[i], task_weights, compute_loss=True)
                flat_grad = torch.autograd.grad(loss, task_weights)
                flat_grad = torch.cat([g.contiguous().view(-1) for g in flat_grad])
                
                meta_loss += loss
                losses.append(loss.item())

                # Use an iterative solver along with reverse mode differentiationto compute task meta-gradient
                if self.cg_steps == 1:
                    meta_grad += flat_grad
                else:
                    evaluator = self.matrix_evaluator(batch_X[i], batch_y[i])
                    meta_grad += cg_solve(evaluator, flat_grad, self.cg_steps)
            meta_grad /= self.tasks_per_meta_batch
            
            # assign meta gradient to weights and take optimisation step
            offset = 0
            for w in self.weights:
                this_grad = meta_grad[offset:offset + w.nelement()].view(w.size())
                # w.grad.copy_(this_grad)
                w.grad = this_grad
                offset += w.nelement()
            self.meta_optimiser.step()
            
            # log metrics
            # epoch_loss.append(meta_loss.item() / self.tasks_per_meta_batch)
            if iteration % self.print_every == 0:
                self.train_log(losses, iteration, num_iterations)
                losses = []


class iMAML_Const(MAML):
    def __init__(self, task_name, inner_lr, meta_lr, radius=0.05, num_ve_iterations=5,
                 K=10, inner_steps=1, tasks_per_meta_batch=25, results_path="./results"):
        super().__init__(task_name, inner_lr, meta_lr, K, inner_steps, tasks_per_meta_batch, results_path)

        self.radius = radius
        self.num_ve_iterations = num_ve_iterations

    def inner_loop(self, X, y):
        X_train, X_test = X[:self.K], X[self.K:]
        y_train, y_test = y[:self.K], y[self.K:]

        temp_weights = [w.clone() for w in self.weights]
        for step in range(self.num_ve_iterations):
            loss = self.criterion(self.model.parameterised(X_train, temp_weights), y_train)

            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

            dist_square = sum(list(map(lambda p: torch.sum(torch.square(p[1] - p[0])), zip(temp_weights, self.weights))))

            d = torch.sqrt(dist_square)
            r = self.radius

            if d > r:
                temp_weights = list(map(lambda p: (r*p[0] + (d-r)*p[1])/d, zip(temp_weights, self.weights)))

        loss = self.criterion(self.model.parameterised(X_test, temp_weights), y_test)

        return loss

    def train(self, num_iterations):
        losses = []

        for iteration in range(1, num_iterations + 1):

            # compute meta loss
            meta_loss = 0.0
            batch_X, batch_y = self.task.sample_data(batch_size=self.tasks_per_meta_batch,
                                                     num_samples=2*self.K, mode="train")
            for i in range(self.tasks_per_meta_batch):
                loss = self.inner_loop(batch_X[i], batch_y[i])
                meta_loss += loss
                losses.append(loss.item())

            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            # epoch_loss.append(meta_loss.item() / self.tasks_per_meta_batch)
            if iteration % self.print_every == 0:
                self.train_log(losses, iteration, num_iterations)
                losses = []

    def __str__(self):
        return "iMAML_Const"