import torch
import torch.nn as nn

from Simulation.Elements import LinearMap, SecondOrderMap


class LinearModel(nn.Module):
    def __init__(self, lattice, dim=4):
        super().__init__()

        # create maps
        self.maps = nn.ModuleList()

        for rMatrix, _, _, _ in lattice.getTransferMaps(dim=dim):
            layer = LinearMap(rMatrix)
            self.maps.append(layer)

        return

    def forward(self, x, outputPerElement: bool = False):
        if outputPerElement:
            outputs = list()
            for map in self.maps:
                x = map(x)
                outputs.append(x)

            return outputs
        else:
            for map in self.maps:
                x = map(x)

            return x


class SecondOrderModel(nn.Module):
    def __init__(self, lattice, dim=4):
        super().__init__()

        # create maps
        self.maps = nn.ModuleList()

        for rMatrix, tMatrix, _, _ in lattice.getTransferMaps(dim=dim):
            layer = SecondOrderMap(rMatrix, tMatrix)
            self.maps.append(layer)

        return

    def forward(self, x):
        for map in self.maps:
            x = map(x)

        return x
