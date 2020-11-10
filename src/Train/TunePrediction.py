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
dim = 4

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

maps = [(r, element.k1) for r, _, element, _ in lattice.getTransferMaps(dim)]
pertMaps = [(r, element.k1) for r, _, element, _ in perturbedLattice.getTransferMaps(dim)]

print(maps[-2])
print(pertMaps[-2])
