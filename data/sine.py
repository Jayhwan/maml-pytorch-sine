import random

import torch
import numpy as np

def to_torch(x):
    return torch.from_numpy(x).float().unsqueeze(-1)


class SINE():
    def __init__(self, x_min=-5.0, x_max=5.0):
        self.x_min = x_min
        self.x_max = x_max

        self.amplitude_min = 0.1
        self.amplitude_max = 5.0

        self.phase_min = 0.0
        self.phase_max = 2 * np.pi

        # self.frequency_min = 1.0
        # self.frequency_max = 3.0
        
    def true_function(self, X, amplitude, phase, frequency):
        return amplitude * np.sin(phase + frequency * X)

    def sample_data(self, batch_size=16, num_samples=5, mode="train"):
        if mode == "train":
            amplitude = []
            for i in range(batch_size):
                if random.random() < 0.95:
                    amplitude.append(np.random.uniform(self.amplitude_min, self.amplitude_min+0.95))
                else:
                    amplitude.append(np.random.uniform(self.amplitude_max - 0.05, self.amplitude_max))
            amplitude = np.array(amplitude).reshape(-1, 1)    
        else:
            amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max, 
                                          size=(batch_size, 1))
        
        phase = np.random.uniform(self.phase_min, self.phase_max, size=(batch_size, 1))
        # frequency = np.random.uniform(self.frequency_min, self.frequency_max, size=(batch_size, 1))
        frequency = 1.0

        if mode == "plot":
            X = np.expand_dims(np.linspace(self.x_min, self.x_max, num=1000), axis=0).repeat(batch_size, axis=0)
        else:
            X = np.random.uniform(self.x_min, self.x_max, size=(batch_size, num_samples))
        # y = amplitude * np.sin(phase + frequency * X)
        y = self.true_function(X, amplitude, phase, frequency)

        X = to_torch(X)
        y = to_torch(y)

        return X, y
