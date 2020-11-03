import numpy

import torch
import torch.nn as nn


class LinearMap(nn.Linear):
    def __init__(self, rMatrix: numpy.ndarray):
        # dimension of transfer matrix
        dim = rMatrix.shape

        super().__init__(in_features=dim[0], out_features=dim[1], bias=False)

        # set initial weights
        weightMatrix = torch.from_numpy(rMatrix)
        self.weight = nn.Parameter(weightMatrix)
        return


class SecondOrderMap(nn.Module):
    def __init__(self, rMatrix: numpy.ndarray, tMatrix: numpy.ndarray):
        super(SecondOrderMap, self).__init__()

        # dimension of transfer matrix
        dim = rMatrix.shape

        self.w1 = nn.Linear(in_features=dim[0], out_features=dim[1], bias=False)
        w1Weights = torch.from_numpy(rMatrix)
        self.w1.weight = nn.Parameter(w1Weights)

        self.w2 = torch.from_numpy(tMatrix)
        self.w2 = torch.reshape(self.w2, (6,6,6))
        self.w2.requires_grad_(True)
        return

    def forward(self, x):
        # evaluate linear map
        x1 = self.w1(x)

        # second order map, expressions are equivalent to bij,...i,...j->...b
        x2 = torch.einsum("...i,...j->...ij", x, x)
        x2 = torch.einsum("...ij,bij->...b", x2, self.w2)
        return x1 + x2

if __name__ == "__main__":
    from Lattice import SIS18_Lattice_minimal

    # get transfer map
    lattice = SIS18_Lattice_minimal()
    latticeElementIterator = lattice.getTransferMaps()

    rMatrix, _, _, _ = next(latticeElementIterator)

    # create layer
    myLayer = LinearMap(rMatrix)


