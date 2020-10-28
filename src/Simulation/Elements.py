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


if __name__ == "__main__":
    from Lattice import SIS18_Lattice_minimal

    # get transfer map
    lattice = SIS18_Lattice_minimal()
    latticeElementIterator = lattice.getTransferMaps()

    rMatrix, _, _, _ = next(latticeElementIterator)

    # create layer
    myLayer = LinearMap(rMatrix)


