import numpy as np

from OcelotMinimal.cpbd import elements

from Simulation.Lattice import Lattice

np.set_printoptions(precision=3, suppress=False)

# element of interest
element = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)

# obtain map
lattice = Lattice([element], )

maps = list()
for r, _, _, _ in lattice.getTransferMaps(dim=6):
    maps.append(r)


sben = maps[1]
print(sben)

# test for symplecticity
sym = np.array([[0,1,0,0,0,0],[-1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,-1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,-1,0]])

def reg(x):
    return np.matmul(x.transpose(), np.matmul(sym, x)) - sym


print(reg(sben))

total = np.matmul(maps[2], np.matmul(maps[1], maps[0]))

print(reg(total))