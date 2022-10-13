"""
@file example_usage.py
@author Ryan Missel

Minimal example to highlight the scheduler's use in a training scheme.
"""
import torch
import matplotlib.pyplot as plt
from source import CosineAnnealingWarmRestartsWithDecayAndLinearWarmup


class Net(torch.nn.Module):
    def __init__(self):
        """ Simple dummy PyTorch network """
        super(Net, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Define model and loss
model = Net()
mse = torch.nn.MSELoss()

# Define Adam optimizer
optim = torch.optim.SGD(model.parameters(), lr=5e-4)

# New scheduler class, instantiated as normal but with additional parameters
cyclic_scheduler = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(optim, T_0=10000, T_mult=1,
                                                                       warmup_steps=2000, decay=0.75)

# Basic iteration loop for LR
lrs = []
for _ in range(120000):
    # Model input/output
    x = torch.rand([1])
    x_hat = model(x)

    # Dummy loss and update
    loss = mse(x, x_hat)
    loss.backward()
    optim.step()

    # Do the scheduler step and record the networks learning rate
    cyclic_scheduler.step()
    lrs.append(optim.param_groups[0]['lr'])

# Plot the LRs over iterations
plt.plot(lrs)
plt.show()
