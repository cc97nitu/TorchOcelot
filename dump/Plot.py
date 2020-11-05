import numpy as np
import matplotlib.pyplot as plt

# load data
elegant = np.loadtxt("elegant.npy").transpose()
cpu_100 = np.loadtxt("runtimeBenchmark_cpu_turns=100.npy").transpose()
cpu_1000 = np.loadtxt("runtimeBenchmark_cpu_turns=1000.npy").transpose()
cuda_100 = np.loadtxt("runtimeBenchmark_cuda_turns=100.npy").transpose()
cuda_1000 = np.loadtxt("runtimeBenchmark_cuda_turns=1000.npy").transpose()

# plot
plt.plot(cpu_100[0], cpu_100[1], linestyle="dotted", color="blue", marker="o")
plt.plot(cuda_100[0], cuda_100[1], linestyle="dotted", color="orange", marker="o")
plt.plot(cpu_1000[0], cpu_1000[1], linestyle="solid", color="blue", marker="o", label="cpu")
plt.plot(cuda_1000[0], cuda_1000[1], linestyle="solid", color="orange", marker="o", label="gpu")

plt.plot(elegant[0], elegant[1], linestyle="solid", color="red", marker="o", label="elegant")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("number particles")
plt.ylabel("time / s")
plt.legend()

plt.show()
plt.close()
