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

        self.frequency_min = 1.0
        self.frequency_max = 3.0

    def sample_data(self, batch_size=16, num_samples=5, mode="train"):
        if mode == "train":
            if random.random() < 0.95:
                amplitude = np.random.uniform(self.amplitude_min, self.amplitude_min+0.95, 
                                              size=(batch_size, 1))
            else:
                amplitude = np.random.uniform(self.amplitude_max - 0.05, self.amplitude_max,
                                              size=(batch_size, 1))
        else:
            amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max, 
                                          size=(batch_size, 1))
        
        phase = np.random.uniform(self.phase_min, self.phase_max, size=(batch_size, 1))
        frequency = np.random.uniform(self.frequency_min, self.frequency_max, size=(batch_size, 1))

        X = np.random.uniform(self.x_min, self.x_max, size=(batch_size, num_samples))
        y = amplitude * np.sin(phase + frequency * X)

        X = to_torch(X)
        y = to_torch(y)

        return X, y