'''
Created on 19/04/2018

@author: Mads
'''
import numpy as np


class WindResource(object):
    def __init__(self, f, a, k, ti):
        wdir = np.linspace(0, 360, len(f), endpoint=False)
        indexes = np.argmin((np.tile(np.arange(360), (len(f), 1)).T - wdir + (360 / len(f) / 2)) % 360, 1)
        self.f = np.array(f)[indexes] / (360 / len(f))
        self.a = np.array(a)[indexes]
        self.k = np.array(k)[indexes]
        self.ti = np.array(ti)[indexes]

    def weibull_weight(self, WS, A, k):
        def cdf(ws, A=A, k=k):
            return 1 - np.exp(-(ws / A) ** k)
        dWS = np.diff(WS[:2], 1, 0) / 2
        return cdf(WS + dWS) - cdf(WS - dWS)

    def __call__(self, turbine_positions, wdir, wsp):
        # f(turbine_positions, wdir, wsp) -> WS[nWT,nWdir,nWsp], TI[nWT,nWdir,nWsp), Weight[nWdir,nWsp]
        WD, WS = np.meshgrid(wdir, wsp)
        weight = self.weibull_weight(WS, self.a[WD], self.k[WD]) * self.f[wdir]
        WD, WS = np.tile(WD, (len(turbine_positions), 1, 1)), np.tile(WS, (len(turbine_positions), 1, 1))
        return WD, WS, self.ti[WD], weight


def main():
    if __name__ == '__main__':
        f = [0.035972, 0.039487, 0.051674, 0.070002, 0.083645, 0.064348, 0.086432, 0.117705, 0.151576, 0.147379, 0.10012, 0.05166]
        A = [9.176929, 9.782334, 9.531809, 9.909545, 10.04269, 9.593921, 9.584007, 10.51499, 11.39895, 11.68746, 11.63732, 10.08803]
        k = [2.392578, 2.447266, 2.412109, 2.591797, 2.755859, 2.595703, 2.583984, 2.548828, 2.470703, 2.607422, 2.626953, 2.326172]
        ti = np.zeros_like(f) + .1
        wr = WindResource(f, A, k, ti)

        print(wr(turbine_positions=(0, 0), wdir=[0, 30, 60], wsp=[4, 5]))


main()
