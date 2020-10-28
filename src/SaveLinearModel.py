import time
import numpy as np
import torch

from Simulation.Lattice import SIS18_Lattice_minimal
from Simulation.Models import LinearModel


# create model of SIS18
print("building model")
dim = 6
lattice = SIS18_Lattice_minimal(nPasses=1)
model = LinearModel(lattice, dim)

# save weights
torch.save(model.state_dict(), "/dev/shm/linearModel.pth")

# load it
newModel = LinearModel(lattice, dim)
newModel.load_state_dict(torch.load("/dev/shm/linearModel.pth"))

# load bunch
print("loading bunch")
bunch = np.loadtxt("../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.from_numpy(bunch)

# compare
diff = model(bunch) - newModel(bunch)

print(diff[0])
