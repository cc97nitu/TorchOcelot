from ocelot.cpbd import elements
from ocelot.cpbd import optics
from ocelot.cpbd.magnetic_lattice import MagneticLattice


class SIS18_Lattice_minimal(MagneticLattice):
    """SIS18 multiturn lattice consisting of dipoles and quadrupoles."""
    def __init__(self, nPasses: int = 1):
        # specify beam line elements
        rb1 = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)
        rb2 = elements.RBend(l=2.617993878, angle=0.2617993878, e1=0.1274090354, e2=0.1274090354)

        qs1f = elements.Quadrupole(l=1.04, k1=3.12391e-01)
        # qs1f = elements.Quadrupole(l=1.04, k1=3.05576e-01)
        qs2d = elements.Quadrupole(l=1.04, k1=-4.78047e-01)
        # qs2d = elements.Quadrupole(l=1.04, k1=-4.94363e-01)
        qs3t = elements.Quadrupole(l=0.4804, k1=2 * 0.311872401)

        d1 = elements.Drift(0.645)
        d2 = elements.Drift(0.9700000000000002)
        d3 = elements.Drift(6.839011704000001)
        d4 = elements.Drift(0.5999999999999979)
        d5 = elements.Drift(0.7097999999999978)
        d6 = elements.Drift(0.49979999100000283)

        cell = [d1, rb1, d2, rb2, d3, qs1f, d4, qs2d, d5, qs3t, d6]

        # lattice consists of twelve identical cells
        lattice = list()
        for i in range(12 * nPasses):
            lattice += cell

        # purpose of this?
        method = optics.MethodTM()
        method.global_method = optics.SecondTM


        super(SIS18_Lattice_minimal, self).__init__(lattice, method=method)

        # log element positions
        self.positions = list()
        totalLength = 0

        for element in self.sequence:
            self.positions.append(totalLength)
            totalLength += element.l

        return

    def getTransferMaps(self, dim=4):
        for i, tm in enumerate(optics.get_map(self, self.totalLen, optics.Navigator(self))):
            # get type and length of element
            element = self.sequence[i]
            elementLength = self.sequence[i].l

            # get transfer matrices
            rMatrix = tm.r_z_no_tilt(tm.length, 0)  # first order 6x6
            tMatrix = tm.t_mat_z_e(tm.length, 0)  # second order 6x6x6

            # cutoff longitudinal coordinates
            rMatrix = rMatrix[:dim, :dim]
            tMatrix = tMatrix[:dim, :dim, :dim].reshape((dim, -1))

            yield rMatrix, tMatrix, element, elementLength


if __name__ == "__main__":
    lattice = SIS18_Lattice_minimal(nPasses=2)

    for r, t, element, length in lattice.getTransferMaps(dim=6):
        print("{} length={}".format(type(element), length))

    lastQuad = lattice.sequence[-2]

    print("num elements: {}".format(len(lattice.sequence)))
