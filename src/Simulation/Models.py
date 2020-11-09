import torch
import torch.nn as nn

from Simulation.Elements import LinearMap, SecondOrderMap


class Model(nn.Module):
    def __init__(self, lattice):
        super(Model, self).__init__()
        self.lattice = lattice
        return

    def forward(self, x, outputPerElement: bool = False):
        if outputPerElement:
            outputs = list()
            for map in self.maps:
                x = map(x)
                outputs.append(x)

            return torch.stack(outputs).permute(1,2,0)  # particle, dim, element
        else:
            for map in self.maps:
                x = map(x)

            return x


class LinearModel(Model):
    def __init__(self, lattice, dim=4):
        super().__init__(lattice)

        # create maps
        self.maps = nn.ModuleList()

        for rMatrix, _, element, _ in lattice.getTransferMaps(dim=dim):
            layer = LinearMap(element, rMatrix)
            self.maps.append(layer)

        return



class SecondOrderModel(Model):
    def __init__(self, lattice, dim=4):
        super().__init__(lattice)

        # create maps
        self.maps = nn.ModuleList()

        for rMatrix, tMatrix, element, _ in lattice.getTransferMaps(dim=dim):
            layer = SecondOrderMap(element, rMatrix, tMatrix)
            self.maps.append(layer)

        return

