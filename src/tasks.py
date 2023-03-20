import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Sine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    
    def __init__(self, amplitude, phase, xmin, xmax):
        self.amplitude = amplitude
        self.phase = phase
        self.xmin = xmin
        self.xmax = xmax
        
    def true_function(self, x):
        """
        Compute the true function on the given x.
        """
        
        # return self.amplitude * np.sin(self.phase + x)
        return self.amplitude * np.sin(self.phase + x)
        
    def sample_data(self, num_points=1, device="cuda"):
        """
        Sample data from this task.
        
        returns: 
            x: the feature vector of length size
            y: the target vector of length size
        """
        batch_size = self.amplitude.shape[0]
        
        x = np.random.uniform(self.xmin, self.xmax, (batch_size, num_points))
        y = self.true_function(x)
        
        x = torch.tensor(x, dtype=torch.float).unsqueeze(-1).to(device)
        y = torch.tensor(y, dtype=torch.float).unsqueeze(-1).to(device)
        
        return x, y

class Sine_Task_Distribution():
    """
    The task distribution for sine regression tasks for MAML
    """
    
    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.x_min = x_min
        self.x_max = x_max
        
    def sample_task(self, batch_size):
        """
        Sample from the task distribution.
        
        returns:
            Sine_Task object
        """
        amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max, size=(batch_size, 1))
        phase = np.random.uniform(self.phase_min, self.phase_max, size=(batch_size, 1))
        return Sine_Task(amplitude, phase, self.x_min, self.x_max)


if __name__ == "__main__":
    task_distribution = Sine_Task_Distribution(amplitude_min=0.1, amplitude_max=0.5, phase_min=0, phase_max=2*np.pi,
                                               x_min=-5.0, x_max=5.0)
    tasks = task_distribution.sample_task(batch_size=16)
    print(tasks.amplitude, tasks.phase)
    x, y = tasks.sample_data(num_points=10)
