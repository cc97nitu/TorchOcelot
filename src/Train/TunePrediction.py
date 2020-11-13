import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from OcelotMinimal.cpbd import elements

import Simulation.Elements
from Simulation.Lattice import SIS18_Lattice
from Simulation.Models import LinearModel
import PlotTrajectory


# specify device and dtype
dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load bunch
dim = 6

bunch = np.loadtxt("../../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.as_tensor(bunch, dtype=dtype)[:,:dim]
bunch.to(device)
bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim

# create model of SIS18 cell
lattice = SIS18_Lattice()
model = LinearModel(lattice, dim, dtype).to(device)
# model.requires_grad_(False)

# create model of perturbed cell
perturbedLattice = SIS18_Lattice()

for element in perturbedLattice.sequence:
    if type(element) is elements.Quadrupole:
        # perturb first quadrupole
        element.k1 = 0.95 * element.k1
        break

perturbedLattice.update_transfer_maps()
perturbedModel = LinearModel(perturbedLattice, dim, dtype).to(device)
perturbedModel.requires_grad_(False)

# plot envelope
fig, axes = plt.subplots(3, sharex=True)

PlotTrajectory.plotBeamSigma(axes[0], PlotTrajectory.track(model, bunch, 1), lattice)
axes[0].set_xlabel("pos / m")
axes[0].set_ylabel("before")

PlotTrajectory.plotBeamSigma(axes[1], PlotTrajectory.track(perturbedModel, bunch, 1), perturbedLattice)
axes[1].set_ylabel("perturbed")

# build training set from perturbed model
with torch.no_grad():
    bunchLabels = perturbedModel(bunch, outputPerElement=True)

trainSet = torch.utils.data.TensorDataset(bunch, bunchLabels)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=400,
                                          shuffle=True, num_workers=2)

# optimization setup
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# train loop
print(model.symplecticRegularization())

for epoch in range(5):
    for i, data in enumerate(trainLoader):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward, backward
        output = model(inputs, outputPerElement=True)
        loss = criterion(output, labels) + model.symplecticRegularization()
        # loss = criterion(output, labels)
        loss.backward()

        # do step in gradient descent
        optimizer.step()

        # # report progress
        # if i % 100 == 99:
        #     print(loss.item())

    # calculate loss over bunch
    with torch.no_grad():
        loss = criterion(model(bunch, outputPerElement=True), bunchLabels)

    print("loss: {}, regularization: {}".format(loss.item(), model.symplecticRegularization()))

# plot envelope of trained model
PlotTrajectory.plotBeamSigma(axes[2], PlotTrajectory.track(model, bunch, 1), lattice)
axes[2].set_ylabel("after")

plt.show()
plt.close()
