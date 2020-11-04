import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from Simulation.Lattice import SIS18_Lattice_minimal
from Simulation.Models import LinearModel, SecondOrderModel


# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on {}".format(str(device)))

# create model of SIS18
print("building model")
dim = 6
lattice = SIS18_Lattice_minimal(nPasses=1)
model = SecondOrderModel(lattice, dim)
model.to(device)

# load bunch
print("loading bunch")
bunch = np.loadtxt("../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.from_numpy(bunch)[:100]

# prepare training
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

dummyTarget = torch.zeros(bunch.shape, dtype=torch.double)
trainSet = torch.utils.data.TensorDataset(bunch, dummyTarget)
trainloader = torch.utils.data.DataLoader(trainSet, batch_size=4,
                                          shuffle=True, num_workers=2)

# train loop
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 25 == 24:  # print every 25 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 25))
            running_loss = 0.0

