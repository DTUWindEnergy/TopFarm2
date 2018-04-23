'''
Created on 19/04/2018

@author: Mads
'''
import numpy as np


class WindResource(object):
    def __init__(self, f, a, k, ti):
        wdir = np.linspace(0, 360, len(f), endpoint=False)
        indexes = np.argmin((np.broadcast_to(np.arange(360), (len(f), 360)).T - wdir) % 360, 1)
        self.f = f[indexes] / (360 / len(f))
        self.a = a[indexes]
        self.k = k[indexes]
        self.ti = ti[indexes]

    def weibull_weight(self, WS, A, k):
        cdf = lambda ws, A=A, k=k: 1 - np.exp(-(ws / A) ** k)
        dWS = np.diff(WS[:2], 1, 0) / 2
        return cdf(WS + dWS) - cdf(WS - dWS)

    def __call__(self, turbine_positions, wdir, wsp):
        # f(turbine_positions, wdir, wsp) -> WS[nWT,nWdir,nWsp], TI[nWT,nWdir,nWsp), Weight[nWdir,nWsp]
        WD, WS = np.meshgrid(wdir, wsp)
        weight = self.weibull_weight(WS, self.a[WD], self.k[WD]) * self.f[wdir]
        return WD, WS, self.ti[WD], weight


if __name__ == '__main__':
    f = np.array("3.597152 3.948682 5.167395 7.000154 8.364547 6.43485 8.643194 11.77051 15.15757 14.73792 10.01205 5.165975".split(), dtype=np.float)
    A = np.array("9.176929  9.782334 9.531809 9.909545 10.04269 9.593921 9.584007 10.51499 11.39895 11.68746 11.63732 10.08803".split(), dtype=np.float)
    k = np.array("2.392578 2.447266 2.412109 2.591797 2.755859 2.595703 2.583984 2.548828 2.470703 2.607422 2.626953 2.326172".split(), dtype=np.float)
    ti = np.zeros_like(f) + .1
    wr = WindResource(f, A, k, ti)

    print(wr((0, 0), [0, 30, 60], [4, 5]))
