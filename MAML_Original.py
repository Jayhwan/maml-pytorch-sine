import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

def simplex_proj(beta):
    beta_sorted = np.flip(np.sort(beta))
    rho = 1
    for i in range(len(beta)-1):
        j = len(beta) - i
        test = beta_sorted[j-1] + (1 - np.sum(beta_sorted[:j]))/(j)
        if test > 0:
            rho = j
            break

    lam = (1-np.sum(beta_sorted[:rho]))/(rho)
    return np.maximum(beta + lam,0)

class Sine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """

    def __init__(self, amplitude, phase, frequency, xmin, xmax):
        self.amplitude = amplitude
        self.phase = phase
        self.frequency = frequency
        self.xmin = xmin
        self.xmax = xmax

    def true_function(self, x):
        """
        Compute the true function on the given x.
        """

        return self.amplitude * np.sin(self.phase + self.frequency * x)

    def sample_data(self, size=1):
        """
        Sample data from this task.

        returns:
            x: the feature vector of length size
            y: the target vector of length size
        """

        x = np.random.uniform(self.xmin, self.xmax, size)
        y = self.true_function(x)

        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1)

        return x, y

class Sine_Task_Distribution():
    """
    The task distribution for sine regression tasks for MAML
    """

    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, frequency_min, frequency_max, x_min, x_max):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.x_min = x_min
        self.x_max = x_max

    def sample_task(self):
        """
        Sample from the task distribution.

        returns:
            Sine_Task object
        """
        amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max)
        phase = np.random.uniform(self.phase_min, self.phase_max)
        frequency = np.random.uniform(self.frequency_min, self.frequency_max)
        return Sine_Task(amplitude, phase, frequency, self.x_min, self.x_max)

    def sample_hard_task(self):
        """
        Sample from the task distribution.

        returns:
            Sine_Task object
        """
        con = np.random.uniform(0, 1)
        if con < 0.95:
            amplitude = np.random.uniform(self.amplitude_min, self.amplitude_min + 0.95)
        else:
            amplitude = np.random.uniform(self.amplitude_max - 0.05, self.amplitude_max)

        phase = np.random.uniform(self.phase_min, self.phase_max)
        frequency = np.random.uniform(self.frequency_min, self.frequency_max)
        return Sine_Task(amplitude, phase, frequency, self.x_min, self.x_max)

class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(1, 40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40, 40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40, 1))
        ]))

    def forward(self, x):
        return self.model(x)

    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = nn.functional.linear(x, weights[0], weights[1])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[2], weights[3])
        x = nn.functional.relu(x)
        x = nn.functional.linear(x, weights[4], weights[5])
        return x

class MAML():
    def __init__(self, model, tasks, inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch = 1000):

        # important objects
        self.tasks = tasks
        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.plot_every = 10
        self.print_every = 500
        self.meta_losses = []

    def inner_loop(self, task):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]

        # perform training on data sampled from task
        X, y = task.sample_data(self.K)
        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K

            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # sample new data for meta-update and compute loss
        X, y = task.sample_data(self.K)
        loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K

        return loss

    def main_loop(self, num_iterations):
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):

            # compute meta loss
            meta_loss = 0
            for i in range(self.tasks_per_meta_batch):
                task = self.tasks.sample_task()
                meta_loss += self.inner_loop(task)

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

            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                f1.write(str(epoch_loss / self.plot_every) + '\n')
                epoch_loss = 0

class TRMAML():
    def __init__(self, model, tasks, inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=1000):

        # important objects
        self.tasks = tasks
        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.plot_every = 10
        self.print_every = 500
        self.meta_losses = []

    def inner_loop(self, task):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]

        # perform training on data sampled from task
        X, y = task.sample_data(self.K)
        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K

            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # sample new data for meta-update and compute loss
        X, y = task.sample_data(self.K)
        loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K

        return loss

    def main_loop(self, num_iterations):
        epoch_loss = 0

        for iteration in range(1, num_iterations + 1):

            # compute meta loss
            meta_loss = 0
            for i in range(self.tasks_per_meta_batch):
                task = self.tasks.sample_task()
                task_loss = self.inner_loop(task)
                if meta_loss < task_loss:
                    meta_loss = task_loss

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

            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                f1.write(str(epoch_loss / self.plot_every) + '\n')
                epoch_loss = 0

class TaRoMAML():
    def __init__(self, model, tasks, p_lr, inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=1000):

        # important objects
        self.tasks = tasks
        self.model = model
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.p_lr = p_lr
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.tasks_per_meta_batch = tasks_per_meta_batch

        # metrics
        self.plot_every = 10
        self.print_every = 500
        self.meta_losses = []

    def inner_loop(self, task):
        # reset inner model to current maml weights
        temp_weights = [w.clone() for w in self.weights]

        # perform training on data sampled from task
        X, y = task.sample_data(self.K)
        for step in range(self.inner_steps):
            loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K

            # compute grad and update inner loop weights
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

        # sample new data for meta-update and compute loss
        X, y = task.sample_data(self.K)
        loss = self.criterion(self.model.parameterised(X, temp_weights), y) / self.K
        return loss

    def main_loop(self, num_iterations):
        epoch_loss = 0
        p = np.ones(self.tasks_per_meta_batch) / self.tasks_per_meta_batch

        for iteration in range(1, num_iterations + 1):

            # compute meta loss
            meta_loss = 0
            for i in range(self.tasks_per_meta_batch):
                task = self.tasks.sample_task()
                meta_loss += p[i] * self.inner_loop(task)
                p[i] -= self.p_lr * meta_loss
            p = simplex_proj(p)
            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            meta_parameter.append(self.model.model.state_dict())

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            epoch_loss += meta_loss.item() / self.tasks_per_meta_batch
            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss / self.plot_every))

            if iteration % self.plot_every == 0:
                self.meta_losses.append(epoch_loss / self.plot_every)
                f1.write(str(epoch_loss / self.plot_every) + '\n')
                epoch_loss = 0

def loss_on_task(initial_model, K, num_steps, optim = torch.optim.SGD, Exp = "Basic"):
    """
    trains the model on a random sine task and measures the loss curve.

    for each n in num_steps_measured, records the model function after n gradient updates.
    """

    # copy MAML model into a new object to preserve MAML weights during training
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(1, 40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
    ]))

    if exp == "TaRoMAML":
        i = np.random.randint(0, num_iterations)
        model.load_state_dict(meta_parameter[i])
    else:
        model.load_state_dict(initial_model.state_dict())

    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), 0.01)

    # train model on a random task
    if Exp == "Basic":
        task = tasks.sample_task()
    else:
        task = tasks.sample_hard_task()

    X, y = task.sample_data(K)
    losses = []

    for step in range(1, num_steps + 1):
        loss = criterion(model(X), y) / K
        losses.append(loss.item())

        # compute grad and update inner loop weights
        model.zero_grad()
        loss.backward()
        optimiser.step()

    return losses

def losses(initial_model, n_samples, K = 5, n_steps = 5, optim=torch.optim.SGD, Exp = "Basic"):
    """
    returns the average learning trajectory of the model trained for ``n_iterations`` over ``n_samples`` tasks
    """

    x = np.linspace(-5, 5, 2)  # dummy input for test_on_new_task
    avg_losses = [0] * n_steps
    worst_losses = [0] * n_steps
    std_devs = [0] * n_steps

    for i in range(n_samples):
        losses = loss_on_task(initial_model, K, n_steps, optim, Exp = Exp)
        avg_losses = [l + l_new for l, l_new in zip(avg_losses, losses)]
        std_devs = [l + l_new * l_new for l, l_new in zip(std_devs, losses)]
        for j in range(n_steps):
            if worst_losses[j] < losses[j]:
                worst_losses[j] = losses[j]

    avg_losses = [l / n_samples for l in avg_losses]
    std_devs = [l / n_samples - k * k for l, k in zip(std_devs, avg_losses)]

    return avg_losses, worst_losses, std_devs

def mixed_pretrained(iterations=500):
    """
    returns a model pretrained on a selection of ``iterations`` random tasks.
    """

    # set up model
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(1, 40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
    ]))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # fit the model
    for i in range(iterations):
        model.zero_grad()
        x, y = tasks.sample_task().sample_data(10)
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()

    return model



# Main Function
meta_parameter = []
num_iterations = 65000

# 1. Default Experiment hyper-parameter
exp = ""
test = "Basic" #Basic, Hard
number_of_sample = 25 #N

# NN (# layer, dim_layer)
# optimizer (SGD, ADAM)

sample_per_task = 10 #K = 5, 10
convergence = 1 #convergence(inner_step) = 1, 2, 5
test_step = 5 #5, 10

p_lr = 0.001 #??
inner_lr = 0.01 #0.01 trash
meta_lr = 0.001 #??

def expTaRo():
    # TaRoMAML Experiment
    exp = "TaRoMAML"

    for p_lr in [0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.015]:
        for inner_lr in [0.003, 0.005, 0.007, 0.009, 0.011, 0.015]:
            for meta_lr in [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.003]:
                for convergence in [1, 2, 5]:
                    meta_parameter = []
                    print("Experiment: " + str(exp) + "_(N, K)=(" + str(
                        number_of_sample) + "," + str(sample_per_task) + ")" + "_step=" + str(
                        convergence) + "_lr=(" + str(p_lr) + "," + str(inner_lr) + "," + str(
                        meta_lr) + ")")

                    # Training Procedure
                    f1 = open("Sinuoid/TaRoMAML/training/" + str(exp) + "_" + str(test) + "_(N, K)=(" + str(
                        number_of_sample) + "," + str(sample_per_task) + ")" + "_step=(" + str(
                        convergence) + "," + str(test_step) + ")" + "_lr=(" + str(p_lr) + "," + str(inner_lr) + "," + str(
                        meta_lr) + ")" + ".csv", "w")

                    tasks = Sine_Task_Distribution(amplitude_min=0.1, amplitude_max=5, phase_min=0, phase_max=2 * np.pi,
                                                   frequency_max=1, frequency_min=1, x_min=-5, x_max=5)

                    maml = TaRoMAML(MAMLModel(), tasks, p_lr=p_lr, inner_lr=inner_lr, meta_lr=meta_lr,
                                    K=sample_per_task, inner_steps=convergence,
                                    tasks_per_meta_batch=number_of_sample)

                    maml.main_loop(num_iterations=num_iterations)
                    f1.close()

                    for test_step in [5, 10]:
                        for test in ["Basic", "Hard"]:
                            # Test Procedure
                            avg_losses, worst_losses, std_devs = losses(maml.model.model, n_samples=5000,
                                                                        K=sample_per_task, n_steps=test_step,
                                                                        Exp=test)
                            f2 = open("Sinuoid/TaRoMAML/test/" + str(exp) + "_" + str(test) + "_(N, K)=(" + str(
                                number_of_sample) + "," + str(sample_per_task) + ")" + "_step=(" + str(
                                convergence) + "," + str(test_step) + ")" + "_lr=(" + str(p_lr) + "," + str(inner_lr) + "," + str(
                                meta_lr) + ")" + ".csv", "w")
                            for i in range(test_step):
                                f2.write(
                                    str(avg_losses[i]) + ',' + str(worst_losses[i]) + ',' + str(std_devs[i]) + '\n')
                        f2.close()


# MAML Experiment
exp = "MAML" #MAML, TR-MAML, TaRo-BOBA-MAML, iMAML, TR-iMAML, TaRo-BOBA-iMAML

for inner_lr in [0.003, 0.005, 0.007, 0.009, 0.011, 0.015]:
    for meta_lr in [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.003]:
        for convergence in [1, 2, 5]:
            print("Experiment: " + str(exp) + "_(N, K)=(" + str(
                number_of_sample) + "," + str(sample_per_task) + ")" + "_step=" + str(
                convergence) + "_lr=(" + str(inner_lr) + "," + str(
                meta_lr) + ")")

            # Training Procedure
            f1 = open("Sinuoid/MAML/training/" + str(exp) + "_" + str(test) + "_(N, K)=(" + str(
                number_of_sample) + "," + str(sample_per_task) + ")" + "_step=(" + str(
                convergence) + "," + str(test_step) + ")" + "_lr=(" + str(inner_lr) + "," + str(
                meta_lr) + ")" + ".csv", "w")

            tasks = Sine_Task_Distribution(amplitude_min=0.1, amplitude_max=5, phase_min=0, phase_max=2 * np.pi,
                                           frequency_max=1, frequency_min=1, x_min=-5, x_max=5)

            maml = MAML(MAMLModel(), tasks, inner_lr=inner_lr, meta_lr=meta_lr, K=sample_per_task,
                        inner_steps=convergence, tasks_per_meta_batch=number_of_sample)

            maml.main_loop(num_iterations=num_iterations)
            f1.close()

            for test_step in [5, 10]:
                for test in ["Basic", "Hard"]:
                    #Test Procedure
                    avg_losses, worst_losses, std_devs = losses(maml.model.model, n_samples=5000,
                                                                K=sample_per_task, n_steps=test_step,
                                                                Exp=test)
                    f2 = open("Sinuoid/MAML/test/" + str(exp) + "_" + str(test) + "_(N, K)=(" + str(
                        number_of_sample) + "," + str(sample_per_task) + ")" + "_step=(" + str(
                        convergence) + "," + str(test_step) + ")" + "_lr=(" + str(inner_lr) + "," + str(
                        meta_lr) + ")" + ".csv", "w")
                    for i in range(test_step):
                        f2.write(
                            str(avg_losses[i]) + ',' + str(worst_losses[i]) + ',' + str(std_devs[i]) + '\n')
                    f2.close()







# TRMAML Experiment
exp = "TRMAML"
convergence = 10

for inner_lr in [0.003, 0.005, 0.007, 0.009, 0.011, 0.015]:
    for meta_lr in [0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.003]:
        print("Experiment: " + str(exp) + "_(N, K)=(" + str(
            number_of_sample) + "," + str(sample_per_task) + ")" + "_step=" + str(
            convergence) + "_lr=(" + str(inner_lr) + "," + str(
            meta_lr) + ")")

        # Training Procedure
        f1 = open("Sinuoid/TRMAML/training/" + str(exp) + "_" + str(test) + "_(N, K)=(" + str(
            number_of_sample) + "," + str(sample_per_task) + ")" + "_step=(" + str(
            convergence) + "," + str(test_step) + ")" + "_lr=(" + str(inner_lr) + "," + str(
            meta_lr) + ")" + ".csv", "w")

        tasks = Sine_Task_Distribution(amplitude_min=0.1, amplitude_max=5, phase_min=0, phase_max=2 * np.pi,
                                       frequency_max=1, frequency_min=1, x_min=-5, x_max=5)


        maml = TRMAML(MAMLModel(), tasks, inner_lr=inner_lr, meta_lr=meta_lr,
                      K=sample_per_task, inner_steps=convergence,
                      tasks_per_meta_batch=number_of_sample)

        maml.main_loop(num_iterations=num_iterations)
        f1.close()

        for test_step in [5, 10]:
            for test in ["Basic", "Hard"]:
                # Test Procedure
                avg_losses, worst_losses, std_devs = losses(maml.model.model, n_samples=5000,
                                                            K=sample_per_task, n_steps=test_step,
                                                            Exp=test)
                f2 = open("Sinuoid/TRMAML/test/" + str(exp) + "_" + str(test) + "_(N, K)=(" + str(
                    number_of_sample) + "," + str(sample_per_task) + ")" + "_step=(" + str(
                    convergence) + "," + str(test_step) + ")" + "_lr=(" + str(inner_lr) + "," + str(
                    meta_lr) + ")" + ".csv", "w")
                for i in range(test_step):
                    f2.write(
                        str(avg_losses[i]) + ',' + str(worst_losses[i]) + ',' + str(std_devs[i]) + '\n')
                f2.close()


