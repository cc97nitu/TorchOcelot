import time
import numpy as np
import torch

from Simulation.Lattice import SIS18_Lattice_minimal
from Simulation.Models import SecondOrderModel, LinearModel


# choose device
dtype = torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on {}".format(str(device)))

# create model of SIS18
print("building model")
dim = 6
lattice = SIS18_Lattice_minimal(nPasses=1)
model = LinearModel(lattice, dim, dtype=dtype)
model.to(device)

# load bunch
print("loading bunch")
bunch = np.loadtxt("../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.as_tensor(bunch, dtype=dtype)[:1000]

# track
turns = int(1e3)  # turns during injection plateau
# turns = int(1.6e5)  # turns during injection plateau

print("started tracking {:.0e} particles for {:.1e} turns".format(len(bunch), turns))

with torch.no_grad():
    multiTurnOutput = list()
    t0 = time.time()

    y = bunch.to(device)
    for i in range(turns):
        # y = model(y)

        y = model(y, outputPerElement=True)
        multiTurnOutput.append(y)
        y = y[:, :, -1]

    print("tracking finished within {:.2f}s".format(time.time() - t0))

print(y[0])
