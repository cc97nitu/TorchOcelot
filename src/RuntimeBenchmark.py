import time
import numpy as np
import torch

from Simulation.Lattice import SIS18_Lattice_minimal
from Simulation.Models import SecondOrderModel


def track(model, bunch, turns: int):

    with torch.no_grad():
        t0 = time.time()

        x = bunch.to(device)
        for turn in range(turns):
            x = model(x)

        t = time.time() - t0

    return t


if __name__ == "__main__":
    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("running on {}".format(str(device)))

    # create model of SIS18
    print("building model")
    dim = 6
    lattice = SIS18_Lattice_minimal(nPasses=1)
    model = SecondOrderModel(lattice, dim)
    model.to(device)

    # load bunch
    print("loading bunch")
    bunch = np.loadtxt("../res/bunch_6d_n=1e5.txt.gz")
    bunch = torch.from_numpy(bunch)
    bunch.to(device)

    # track
    print("started tracking")
    particles = [10**i for i in range(1,6)]
    benchmark = [[n, track(model, bunch[:n], 100,)] for n in particles]

    benchmark = np.array(benchmark)
    print(benchmark)

    # dump results
    np.savetxt("../dump/runtimeBenchmark_{}.npy".format(device), benchmark)
