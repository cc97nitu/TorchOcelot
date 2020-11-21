import numpy as np
import torch

from Simulation.Lattice import DummyLattice
from Simulation.Models import LinearModel


def driftMap(length: float):
    m = np.array([[1, length, 0, 0],[0,1,0,0],[0,0,1,length],[0,0,0,1]])
    return m


def quadMap(length: float, k1: float):
    m = np.array([[1,0,0,0],[-1*length*k1,1,0,0],[0,0,1,0],[0,0,length*k1,0]])
    return m


class Drift(object):
    def __init__(self, length: float):
        self.drift1 = driftMap(length)
        return

    def __call__(self, *args, **kwargs):
        return np.matmul(self.drift1, args[0])


class Quadrupole(object):
    def __init__(self, length, k1):
        self.drift1 = driftMap(length / 2)
        self.kick1 = quadMap(length, k1)
        self.drift2 = driftMap(length / 2)
        return

    def __call__(self, *args, **kwargs):
        x = args[0]
        x = np.matmul(self.drift1, x)
        x = np.matmul(self.kick1, x)
        x = np.matmul(self.drift2, x)
        return x


if __name__ == "__main__":
    # define FODO
    k1 = 0.5

    d1 = Drift(1)
    qf = Quadrupole(1, k1)
    d2 = Drift(2)
    qd = Quadrupole(1, -k1)

    # create particle
    x0 = np.array([1e-3, 2e-3, 1e-3, 0])

    # track
    x = qd(d2(qf(d1(x0))))

    print(x)

    print(d1(x0))

    #####################3
    # Ocelot
    lattice = DummyLattice()
    ocelotModel = LinearModel(lattice)

    print(ocelotModel(torch.as_tensor(x0, dtype=torch.float32)))