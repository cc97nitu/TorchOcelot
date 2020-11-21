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
from Simulation.Lattice import SIS18_Lattice, SIS18_Cell  # correctors are defined as Hcor
from Simulation.Models import SecondOrderModel

from PetraIV.LatticePetraIV import PetraIVLattice  # correctors are defined as Marker

# general properties
dtype = torch.float32
device = torch.device("cpu")

# create model of PetraIV
dim = 2
Lattice = PetraIVLattice

lattice = Lattice()
model = SecondOrderModel(lattice, dim, dtype).to(device)

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
fig, axes = plt.subplots(4, sharex=True)
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
print("training model")
for epoch in range(200):
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
axes[2].set_ylabel("trained")


"""
************
correct orbits 
************
"""

# find correctors in model
correctors = list()
for m in model.maps:
    if type(m.element) is elements.Marker:
        # add bias
        m.w1.bias = nn.Parameter(torch.zeros(dim, dtype=dtype))
        correctors.append(m)

if not correctors:
    print("no correctors present")
    exit()
else:
    print("found {} correctors".format(len(correctors)))


def applyCorrectorSettings(settings: np.array):
    # update correctors
    for corrector, kick in zip(correctors, settings):
        corrector.w1.bias[1] = kick

    # observe trajectory
    out = model(xRef, outputAtBPM=outputAtBPM)

    # max orbit deviation
    maxDev = torch.abs(out)[0].max()  # restrict to x-coord
    return maxDev.item()


# minimize loss using correctors
print("optimizing corrector settings")
optimRes = scipy.optimize.minimize(applyCorrectorSettings, np.zeros(len(correctors)), tol=1e-6, method="Nelder-Mead",
                                   options={"maxiter": 200, "disp": True, 'fatol': 1e-6, 'xatol': 1e-6})
print(optimRes)

# show corrections
PlotTrajectory.plotTrajectories(axes[3], PlotTrajectory.track(model, xRef, 1), lattice)

axes[3].set_xlabel("pos / m")
axes[3].set_ylabel("corrected")

# all plots shall have the same y-range
axes[0].set_ylim(axes[1].get_ylim())
axes[2].set_ylim(axes[1].get_ylim())
axes[3].set_ylim(axes[1].get_ylim())

plt.show()
plt.close()
