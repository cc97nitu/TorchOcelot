import sys
sys.path.append("../")

import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from OcelotMinimal.cpbd import elements

import Simulation.Elements
from Simulation.Lattice import SIS18_Lattice, SIS18_Lattice_minimal
from Simulation.Models import LinearModel
import PlotTrajectory
from Tune import getTuneChromaticity

# specify device and dtype
dtype = torch.float32
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("running on {}".format(device))

# create model of SIS18
dim = 6

lattice = SIS18_Lattice()
model = LinearModel(lattice, dim, dtype)
model = model.to(device)
model.setTrainable("quadrupoles")

# get index of first quad
for i in range(len(lattice.sequence)):
    if type(lattice.sequence[i]) is elements.Quadrupole:
        idxFirstQuad = i

# create model of perturbed accelerator
perturbedLattice = SIS18_Lattice()

perturbedLattice.sequence[idxFirstQuad].k1 *= 0.9
perturbedLattice.update_transfer_maps()

perturbedModel = LinearModel(perturbedLattice, dim, dtype)
perturbedModel = perturbedModel.to(device)
perturbedModel.requires_grad_(False)

# load bunch
bunch = np.loadtxt("../../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.as_tensor(bunch, dtype=dtype)[:20,:dim]
bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim
bunch = bunch.to(device)

# bunch = torch.tensor([[1e-3, 0, 1e-3, 0, 0, 0],], dtype=dtype)
# bunch = bunch.to(device)


# show initial tunes
print("initial tunes: ideal={}, perturbed={}".format(model.getTunes(), perturbedModel.getTunes()))

# plot envelope
fig, axes = plt.subplots(3, sharex=True)

# PlotTrajectory.plotBeamSigma(axes[0], PlotTrajectory.track(model, bunch, 1), lattice)
PlotTrajectory.plotTrajectories(axes[0], PlotTrajectory.track(model, bunch, 1), lattice)
axes[0].set_ylabel("before")

# PlotTrajectory.plotBeamSigma(axes[1], PlotTrajectory.track(perturbedModel, bunch, 1), perturbedLattice)
PlotTrajectory.plotTrajectories(axes[1], PlotTrajectory.track(perturbedModel, bunch, 1), perturbedLattice)
axes[1].set_ylabel("perturbed")

# build training set from perturbed model
outputPerElement = False  # model output shall contain phase space after each element, overwrites outputAtBPM
outputAtBPM = True

with torch.no_grad():
    bunchLabels = perturbedModel(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM)

bunch = bunch.to("cpu")
bunchLabels = bunchLabels.to("cpu")

trainSet = torch.utils.data.TensorDataset(bunch, bunchLabels)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=100,
                                          shuffle=True, num_workers=2)

# optimization setup
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train loop
print("initial loss: {}, initial regularization {}".format(criterion(model(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM), bunchLabels), model.symplecticRegularization()))

t0 = time.time()
for epoch in range(2000):
    for i, data in enumerate(trainLoader):
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward, backward
        output = model(inputs, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM)
        loss = criterion(output, labels) + 10 * model.symplecticRegularization()
        # loss = criterion(output, labels) + model.symplecticRegularization()
        # loss = criterion(output, labels)
        loss.backward()

        # do step in gradient descent
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e-3)

        optimizer.step()

        # # report progress
        # if i % 100 == 99:
        #     print(loss.item())

    if epoch % 100 == 99:
        print("\r" + "epoch: {}".format(epoch + 1))

print("training completed within {:.2f}s".format(time.time() - t0))
print("final loss: {}, final regularization {}".format(criterion(model(bunch, outputPerElement=outputPerElement, outputAtBPM=outputAtBPM), bunchLabels), model.symplecticRegularization()))

# plot envelope of trained model
# PlotTrajectory.plotBeamSigma(axes[2], PlotTrajectory.track(model, bunch.to(device), 1), lattice)
PlotTrajectory.plotTrajectories(axes[2], PlotTrajectory.track(model, bunch.to(device), 1), lattice)
axes[2].set_ylabel("after")
axes[2].set_xlabel("pos / m")


# show plot
axes[0].set_ylim(axes[1].get_ylim())
axes[2].set_ylim(axes[1].get_ylim())

plt.show()
plt.close()

# print tune obtained from phase advance
print("final tunes: {}".format(model.getTunes()))

# tunes = getTuneChromaticity(model.to("cpu"), 200, dtype)
# print(tunes)