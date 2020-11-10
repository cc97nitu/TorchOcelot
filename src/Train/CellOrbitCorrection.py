import sys
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from ocelot.cpbd import elements

import Simulation.Elements
from Simulation.Lattice import SIS18_Cell_hCor
from Simulation.Models import LinearModel
import PlotTrajectory


# create model of SIS18 cell
dim = 4
lattice = SIS18_Cell_hCor()
model = LinearModel(lattice, dim)
model.requires_grad_(False)

# load bunch
bunch = np.loadtxt("../../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.from_numpy(bunch)[:,:4]
bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim

# build training set from ideal model
with torch.no_grad():
    bunchLabels = model(bunch, outputPerElement=True)

trainSet = torch.utils.data.TensorDataset(bunch, bunchLabels)
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=400,
                                          shuffle=True, num_workers=2)

# add horizontal kick to the first dipole
for m in model.modules():
    if type(m) is Simulation.Elements.LinearMap:
        if type(m.element) is elements.RBend:
            bias = torch.zeros(dim, dtype=torch.double)
            bias[1] = 1e-3
            m.bias = nn.Parameter(bias)
            m.bias.requires_grad_(False)
            break

# activate bias on correctors and mark them as trainable
kicks = list()
for m in model.modules():
    if type(m) is Simulation.Elements.LinearMap:
        if type(m.element) is elements.Hcor:
            bias = torch.zeros(dim, dtype=torch.double)
            m.bias = nn.Parameter(bias)
            m.bias.requires_grad_(True)
            kicks.append(m.bias)

# optimization setup
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train loop

for epoch in range(10):
    for i, data in enumerate(trainLoader):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward, backward
        output = model(inputs, outputPerElement=True)
        # loss = criterion(output, labels) + 2 * kickReg()
        loss = criterion(output, labels)
        loss.backward()

        # do step in gradient descent
        optimizer.step()

        # # report progress
        # if i % 100 == 99:
        #     print(loss.item())

    # calculate loss over bunch
    with torch.no_grad():
        loss = criterion(model(bunch, outputPerElement=True), bunchLabels)

    print("loss: {}, regularization: {}".format(loss.item(), None))

with torch.no_grad():
    print("final loss: {}".format(criterion(bunchLabels, model(bunch, outputPerElement=True)).item()))

# plot trajectories from trained model
PlotTrajectory.plotBeamCentroid(PlotTrajectory.track(model, bunch, 1), lattice)

# what happened to the correctors?
for m in model.modules():
    if type(m) is Simulation.Elements.LinearMap:
        if type(m.element) is elements.Hcor:
            print(m.bias)

