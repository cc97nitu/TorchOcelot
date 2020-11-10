import sys
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from ocelot.cpbd import elements

import Simulation.Elements
from Simulation.Lattice import SIS18_Cell_minimal
from Simulation.Models import LinearModel
import PlotTrajectory



# load bunch
dim = 2

bunch = np.loadtxt("../../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.from_numpy(bunch)[:,:dim]
bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim

# create model of SIS18 cell
lattice = SIS18_Cell_minimal()
model = LinearModel(lattice, dim)
# model.requires_grad_(False)

# create model of perturbed cell
perturbedLattice = SIS18_Cell_minimal()

for element in reversed(perturbedLattice.sequence):
    if type(element) is elements.Quadrupole:
        # perturb first quadrupole
        element.k1 = 0.8 * element.k1
        break

perturbedLattice.update_transfer_maps()
perturbedModel = LinearModel(perturbedLattice, dim)
perturbedModel.requires_grad_(False)

# build training set from perturbed model
with torch.no_grad():
    bunchLabels = perturbedModel(bunch, outputPerElement=True)

trainSet = torch.utils.data.TensorDataset(bunch, bunchLabels)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=400,
                                          shuffle=True, num_workers=2)

# optimization setup
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# train loop
print(model.symplecticRegularization())

for epoch in range(20):
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
