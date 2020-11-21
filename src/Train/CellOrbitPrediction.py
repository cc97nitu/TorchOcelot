import sys

sys.path.append("../")

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from OcelotMinimal.cpbd import elements

import PlotTrajectory
from Simulation.Lattice import SIS18_Lattice, SIS18_Cell
from Simulation.Models import SecondOrderModel

from PetraIV.LatticePetraIV import PetraIVLattice

# general properties
dtype = torch.float32
device = torch.device("cpu")

# create model of PetraIV
dim = 2
Lattice = PetraIVLattice

lattice = Lattice()
model = SecondOrderModel(lattice, dim, dtype).to(device)

print("ideal model transfer map:", model.firstOrderOneTurnMap())

for m in model.maps:
    # add trainable bias to map
    bias = torch.zeros(dim, dtype=dtype)
    m.w1.bias = nn.Parameter(bias)

# model.setTrainable("quadrupoles")

# create perturbed version of PetraIV
perturbedLattice = Lattice()

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
optimizer = optim.Adam(model.parameters(), lr=1e-6, )

refLabel = perturbedModel(xRef, outputAtBPM=outputAtBPM)

# train loop
print("training model")
for epoch in range(50):
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

# all plots shall have the same y-range
axes[0].set_ylim(axes[1].get_ylim())
axes[2].set_ylim(axes[1].get_ylim())

plt.show()
plt.close()

# # take a look at the one-turn maps
# print("perturbed model:")
# print(perturbedModel.firstOrderOneTurnMap())
#
# print("trained model:")
# print(model.firstOrderOneTurnMap())

# # take a look at the deviation per map
# for i in range(len(model.maps)):
#     if type(model.maps[i].element) is elements.Quadrupole:
#         print(model.maps[i].w1.bias)
#         print(perturbedModel.maps[i].w1.bias)
#
#         print(model.maps[i].w1.weight)
#         print(perturbedModel.maps[i].w1.weight)
#
#         print(model.maps[i].w2)
#         print(perturbedModel.maps[i].w2)

