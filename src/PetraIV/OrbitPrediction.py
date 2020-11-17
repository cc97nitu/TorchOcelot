import sys

sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from OcelotMinimal.cpbd import elements

import PlotTrajectory
from LatticePetraIV import PetraIVLattice
from Simulation.Models import SecondOrderModel

# general properties
dtype = torch.float32
device = torch.device("cpu")

# create model of PetraIV
dim = 2

lattice = PetraIVLattice()
model = SecondOrderModel(lattice, dim, dtype).to(device)

print("ideal model transfer map:", model.firstOrderOneTurnMap())

for m in model.maps:
    # add trainable bias to map
    bias = torch.zeros(dim, dtype=dtype)
    m.w1.bias = nn.Parameter(bias)

# create perturbed version of PetraIV
perturbedLattice = PetraIVLattice()

for element in perturbedLattice.sequence:
    if type(element) is elements.Quadrupole:
        element.dx = torch.normal(mean=0., std=1e-5, size=(1,)).item()  # same sigma as in TM-PNN

perturbedLattice.update_transfer_maps()

perturbedModel = SecondOrderModel(perturbedLattice, dim, dtype).to(device)
for m in perturbedModel.maps:
    if type(m.element) is elements.Quadrupole:
        # add kick from quadrupole misalignment
        offset = torch.zeros(dim, dtype=dtype)
        offset[0] = m.element.dx

        kick = torch.matmul((m.w1.weight - torch.eye(dim, dtype=dtype)), offset)

        m.w1.bias = nn.Parameter(kick)

perturbedModel.requires_grad_(False)

# create reference particle
xRef = torch.zeros((1, dim), dtype=dtype)

# plot initial trajectories
fig, axes = plt.subplots(3, sharex=True)
PlotTrajectory.plotTrajectories(axes[0], PlotTrajectory.track(model, xRef, 1), lattice)
axes[0].set_ylabel("ideal")

PlotTrajectory.plotTrajectories(axes[1], PlotTrajectory.track(perturbedModel, xRef, 1), lattice)
axes[1].set_ylabel("perturbed")

# optimization setup
outputAtBPM = True

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5, )

refLabel = perturbedModel(xRef, outputAtBPM=outputAtBPM)

# train loop
for epoch in range(100):
    optimizer.zero_grad()

    out = model(xRef, outputAtBPM=outputAtBPM)
    loss = criterion(refLabel, out)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-10)
    optimizer.step()

    if epoch % 10 == 9:
        print("loss: {}".format(loss.item()))

# plot final trajectory
PlotTrajectory.plotTrajectories(axes[2], PlotTrajectory.track(model, xRef, 1), lattice)
axes[2].set_xlabel("pos / m")
axes[2].set_ylabel("trained")

plt.show()
plt.close()

# take a look at the one-turn maps
print("perturbed model:")
print(perturbedModel.firstOrderOneTurnMap())

print("trained model:")
print(model.firstOrderOneTurnMap())
