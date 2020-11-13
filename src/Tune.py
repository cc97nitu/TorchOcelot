import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.fft

from Simulation.Lattice import SIS18_Lattice_minimal, SIS18_Lattice
from Simulation.Models import LinearModel, SecondOrderModel


def getTuneFFT(x: torch.Tensor, model, turns: int):
    # track particle through lattice
    with torch.no_grad():
        phaseSpace = [x, ]

        for turn in range(turns):
            x = model(x)
            phaseSpace.append(x)

        phaseSpace = torch.stack(phaseSpace)

    # do fft of x Coord
    xCoord = phaseSpace.transpose(1,0)[0]

    xCoord = xCoord - torch.mean(xCoord)    # remove dispersion
    fft = torch.fft.fft(xCoord)
    fft = torch.abs(fft)

    sampleRatePerTurn = 1
    freqs = np.fft.fftfreq(len(fft), sampleRatePerTurn)
    idx = np.argsort(freqs)

    # remove negative frequencies
    idx = idx[len(idx)//2:]

    return freqs[idx], fft[idx]


def getTuneChromaticity(model, turns: int, dtype: torch.dtype):
    # set up particles
    dp = np.linspace(-5e-3, 5e-3, 9)
    bunch = torch.tensor([[1e-2,0,1e-2,0,0,i] for i in dp], dtype=dtype)

    # track and fft
    fftList = list()
    for particle in range(len(bunch)):
        freqs, fft = getTuneFFT(bunch[particle], model, turns)
        fftList.append(fft)

    # get tunes from spectrum
    tunes = [freqs[np.argmax(fft)] for fft in fftList]
    fit = np.polyfit(dp, tunes, deg=1)

    return fit


if __name__ == "__main__":
    # create model of SIS18
    dim = 6
    dtype = torch.float32
    # lattice = SIS18_Lattice_minimal(nPasses=1)
    lattice = SIS18_Lattice(nPasses=1)
    model = SecondOrderModel(lattice, dim, dtype=dtype)

    t0 = time.time()
    print(getTuneChromaticity(model, 100))
    print("chromaticity obtained within {:.2f}".format(time.time() - t0))
    # # set up particles
    # dp = np.linspace(-5e-3, 5e-3, 9)
    # x = torch.tensor([[1e-2,0,1e-2,0,0,i] for i in dp], dtype=dtype)
    # # x = torch.tensor([[1e-2,0,1e-2,0,0,0],], dtype=torch.double)
    #
    #
    # # find tune
    # t0 = time.time()
    #
    # turns = 500
    #
    # freqs, _ = getTuneFFT(x[0], model, turns)
    # fftList = list()
    #
    # for particle in range(len(x)):
    #     _, fft = getTuneFFT(x[particle], model, turns)
    #     fftList.append(fft)
    #
    # print("fft completed within {}".format(time.time() - t0))
    #
    # # plot spectra
    # for particle in fftList:
    #     plt.plot(freqs, particle)
    #
    # plt.xlabel("tune")
    #
    # plt.show()
    # plt.close()
    #
    # # get tunes from spectrum
    # print(dp, x[0][-1])
    #
    # tunes = [freqs[np.argmax(fft)] for fft in fftList]
    #
    # plt.plot(dp, tunes)
    # plt.show()
    # plt.close()
    #
    # fit = np.polyfit(dp, tunes, deg=1)
    # print(fit)
