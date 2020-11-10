import numpy

import torch
import torch.nn as nn


class LinearMap(nn.Linear):
    def __init__(self, element, rMatrix: numpy.ndarray):
        self.element = element

        # dimension of transfer matrix
        dim = rMatrix.shape

        # set symplectic structure matrix
        if dim[0] == 2:
            self.symStruct = torch.tensor([[0,1],[-1,0]], dtype=torch.double)
        elif dim[0] == 4:
            self.symStruct = torch.tensor([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]], dtype=torch.double)
        elif dim[0] == 6:
            self.symStruct = torch.tensor(
                [[0,1,0,0,0,0],[-1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,-1,0,0,0],[0,0,0,0,0,1],
                 [0,0,0,0,-1,0]], dtype=torch.double)
        else:
            raise NotImplementedError("phase space dimension of {} not supported".format(dim))

        super().__init__(in_features=dim[0], out_features=dim[1], bias=False)

        # set initial weights
        weightMatrix = torch.from_numpy(rMatrix)
        self.weight = nn.Parameter(weightMatrix)

        return

    def symplecticRegularization(self):
        """Calculate norm of Transpose(J).S.J-S symplectic condition."""
        penalty = torch.matmul(self.weight.transpose(1,0), torch.matmul(self.symStruct, self.weight)) - self.symStruct
        penalty = torch.norm(penalty)
        return penalty


class SecondOrderMap(nn.Module):
    def __init__(self, element, rMatrix: numpy.ndarray, tMatrix: numpy.ndarray):
        super(SecondOrderMap, self).__init__()
        self.element = element

        # dimension of transfer matrix
        dim = rMatrix.shape

        # first order
        self.w1 = nn.Linear(in_features=dim[0], out_features=dim[1], bias=False)
        w1Weights = torch.from_numpy(rMatrix)
        self.w1.weight = nn.Parameter(w1Weights)

        # second order
        w2 = torch.from_numpy(tMatrix)
        w2 = torch.reshape(w2, (6,6,6))
        w2.requires_grad_(True)
        self.register_parameter("w2", nn.Parameter(w2))

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
    latticeElementIterator = lattice.getTransferMaps(dim=2)

    rMatrix, _, element, _ = next(latticeElementIterator)

    # create layer
    myLayer = LinearMap(element, rMatrix)


