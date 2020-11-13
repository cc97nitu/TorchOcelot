import sys
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from OcelotMinimal.cpbd import elements

import Simulation.Elements
from Simulation.Lattice import SIS18_Cell_minimal
from Simulation.Models import LinearModel
import PlotTrajectory


# load bunch
bunch = np.loadtxt("../../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.from_numpy(bunch)
bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim

# create model of SIS18 cell
dim = 6
lattice = SIS18_Cell_minimal()
model = LinearModel(lattice, dim)
model.requires_grad_(False)

# first dipole has trainable kick
for m in model.modules():
    if type(m) is Simulation.Elements.LinearMap:
        if type(m.element) is elements.RBend:
            bias = torch.zeros(6, dtype=torch.double)
            m.bias = nn.Parameter(bias)
            m.bias.requires_grad_(True)
            break


# perturbed model: add horizontal kick to the first dipole
realModel = LinearModel(lattice, dim)
realModel.requires_grad_(False)

for m in realModel.modules():
    if type(m) is Simulation.Elements.LinearMap:
        if type(m.element) is elements.RBend:
            bias = torch.tensor([0,1e-3,0,0,0,0], dtype=torch.double)
            m.bias = nn.Parameter(bias)
            m.bias.requires_grad_(False)
            break

# build training set from perturbed model
with torch.no_grad():
    bunchLabels = realModel(bunch, outputPerElement=True)

dataset = torch.utils.data.TensorDataset(bunch, bunchLabels)
trainSet, valSet = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=400,
                                          shuffle=True, num_workers=2)
valLoader = torch.utils.data.DataLoader(valSet, batch_size=400,
                                          shuffle=True, num_workers=2)

# optimization setup
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train loop
for epoch in range(60):
    for i, data in enumerate(trainLoader):
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward, backward
        output = model(inputs, outputPerElement=True)
        loss = criterion(output, labels)
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

    print("train loss: {}".format(loss.item()))

# what happened to the RBend?
for m in model.modules():
    if type(m) is Simulation.Elements.LinearMap:
        if type(m.element) is elements.RBend:
            print(m.bias)
            break
