import time
import numpy as np
import matplotlib.pyplot as plt

import torch

from Simulation.Lattice import SIS18_Lattice_minimal, SIS18_Cell_minimal
from Simulation.Models import LinearModel


# create model of SIS18
print("building model")
dim = 6
lattice = SIS18_Lattice_minimal(nPasses=1)
model = LinearModel(lattice, dim)

# load bunch
print("loading bunch")
bunch = np.loadtxt("../res/bunch_6d_n=1e5.txt.gz")
bunch = torch.from_numpy(bunch)[:100]
bunch = bunch - bunch.permute(1, 0).mean(dim=1)  # set bunch centroid to 0 for each dim

# track
turns = int(1e0)  # turns during injection plateau

print("started tracking {:.0e} particles for {:.1e} turns".format(len(bunch), turns))

with torch.no_grad():
    t0 = time.time()

    multiTurnOutput = list()
    y = bunch
    for i in range(turns):
        # y = model(y)
        # multiTurnOutput.append(y)

        y = model(y, outputPerElement=True)
        multiTurnOutput.append(y)
        y = y[-1]

    print("tracking finished within {:.2f}s".format(time.time() - t0))

# prepare tracks for plotting
trackResults = torch.stack(multiTurnOutput[0])  # restrict to first turn, indices: element, particle, dim

# # plot individual trajectories
# trackResults = trackResults.permute((1,2,0))  # particle, dim, element

# for particle in trackResults:
#     # x-coord
#     plt.plot(lattice.positions, particle[0])
#
# plt.show()
# plt.close()

# # plot beam position
# trackResults = trackResults.permute((2,0,1))  # dim, element, particle
# beamCentroid = torch.mean(trackResults, dim=2)
#
# bunchCentroid = bunch.permute(1, 0).mean(dim=1)  # dim, particle
# print(bunchCentroid)
#
# plt.plot(lattice.positions, beamCentroid[0].numpy())
# plt.show()
# plt.close()

# plot beam position
trackResults = trackResults.permute((2,0,1))  # dim, element, particle
beamSigma = torch.std(trackResults, dim=2)

bunchSigma = bunch.permute(1, 0).std(dim=1)  # dim, particle
print(bunchSigma)

plt.plot(lattice.endPositions, beamSigma[1].numpy())
plt.show()
plt.close()
