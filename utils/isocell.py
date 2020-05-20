import numpy as np

class Isocell(object):
    """docstring for Isocell"""
    #TODO: Modify and update this accordingly

    # Distribute points on a unit circle according to the Isocell method; the circle
    # is divided into cells of equal area and a point is created in each cell; this
    # method is useful to distribute rays uniformely in space.
    #
    # Syntax:
    #
    #   [A0,Xr,Yr[,Xc,Yc]] = isocell_distribution(Nobj,N0,isrand)
    #
    # Inputs args are:
    #   Nobj    the objective number of cells/points
    #   N0      the initial division (the number of cells near the center)
    #   isrand  indicates if the points are set randomly or not in each cell
    #             -1,0    point is set on cell middle point
    #                1    point is set randomly in its cell
    #                2    point is set randomly (only along radial dir.)
    #                3    point is set randomly with a greater probability in the center
    #                4    point is set randomly (only along radial dir.)
    #
    # Outputs args are:
    #
    #   A0      the cell area (also the weight of each point)
    #   Xr,Yr   are the coordinates of the distributed points
    #   Xc,Yc   are the coordinates of the cells borders
    #
    # Example: distribute at least 200 points on the unit circle with no random
    #
    #   >> [A0,Xr,Yr,Xc,Yc]=isocell_distribution(200,3,0);
    #   >> figure
    #   >> plot(Xc,Yc,'b')
    #   >> hold on
    #   >> plot(Xr,Yr,'.r')
    #   >> axis equal

    #   >> dirk = [Xr; Yr; sqrt(1-Xr.^2-Yr.^2)];
    #   >> dirk = dirk';
    #   >> figure, plot_vertices(dirk);
    #
    #  Theodore Tsesmelis (2020)


    def __init__(self, rays = 10000, div = 3, isrand = 3, draw_cells=False):
        self.__Nobj = rays
        self.__N0 = div
        self.__isrand = isrand
        self.__isdraw = draw_cells

        # initialize area of cells
        self.A0 = []

        # initialize cell center points, from where rays are passing
        self.Xr = []
        self.Yr = []
        self.Zr = []

        # initialize cell perimeter points
        self.Xc = []
        self.Yc = []
        self.Zc = []

        self.__isocell_distribution()

    def __isocell_distribution(self):
        # Number of divisions
        n = np.sqrt(self.__Nobj / self.__N0)
        n = int(np.ceil(n))
        Ntot = int(self.__N0 * n ** 2)

        # init
        Xr = np.zeros(Ntot)
        Yr = np.zeros(Ntot)
        # Rr = np.zeros(Ntot)
        # Tr = np.zeros(Ntot)
        Xc = np.array([])
        Yc = np.array([])

        # cell area
        A0 = np.pi / Ntot

        # distance between circles
        dR = 1 / n

        # rings
        nn = 0
        if self.__isdraw:
            nu = 10
            uu = np.linspace(0,nu, nu+1).reshape(-1,1)
            uu = uu / nu

        for i in range(1,n+1):
            R = i * dR
            nc = self.__N0 * (2 * i - 1)
            dth = 2 * np.pi / nc
            th0 = np.random.rand(1) * dth

            if self.__isrand == -1:
                th0 = 0

            th0 = th0 + np.arange(0,nc) * dth
            ind = nn + np.arange(0,nc)

            if self.__isrand == 1:
                R = R - np.random.rand(1, nc) * dR
                th = th0 + np.random.rand(1, nc) * dth
            elif self.__isrand == 2:
                R = R - np.random.rand(1, nc) * dR
                th = th0 + dth / 2
            elif self.__isrand == 3:
                rr = (1 + np.random.randn(1, nc) / 6.5) / 2
                R = R - rr * dR
                rr = (1 + np.random.randn(1, nc) / 6.5) / 2
                th = th0 + rr * dth / 2
            elif self.__isrand == 4:
                rr = (1 + np.random.randn(1, nc) / 6.5) / 2
                R = R - rr * dR
                th = th0 + dth / 2
            else:
                R = R - dR / 2
                th = th0 + dth / 2

            Xr[ind] = R * np.cos(th)
            Yr[ind] = R * np.sin(th)
            nn = nn + nc

            if self.__isdraw:
                    Rext = i * dR
                    tt = np.arange(0.,360.1, 0.5) / 180 * np.pi
                    Xc = np.hstack([Xc, np.nan, Rext * np.cos(tt)])
                    Yc = np.hstack([Yc, np.nan, Rext * np.sin(tt)])
                    rr = Rext - uu * dR
                    xx = np.vstack([rr * np.cos(th0), np.nan * np.ones((1, nc))])
                    yy = np.vstack([rr * np.sin(th0), np.nan * np.ones((1, nc))])
                    xx = xx.flatten('F') # or xx.reshape((1, (nu + 2) * nc), order='F')
                    yy = yy.flatten('F') # or yy.reshape((1, (nu + 2) * nc), order='F')
                    Xc = np.hstack([Xc, np.nan, xx])
                    Yc = np.hstack([Yc, np.nan, yy])

        self.A0 = A0

        self.Xr = Xr
        self.Yr = Yr
        self.Zc = np.sqrt(1-Xr**2 - Yr**2)

        self.Xc = Xc
        self.Yc = Yc
        self.Zc = np.sqrt(1-Xc**2 - Yc**2)

