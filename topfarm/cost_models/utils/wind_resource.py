'''
Created on 19/04/2018

@author: Mads
'''
import numpy as np
class WindResource(object):
    def __init__(self,f,a,k,ti):
        self.f = f
        self.a = a
        self.k = k
        self.ti = ti
    
    def weibull_weight(self, ws):
        cdf = lambda ws, A=self.A,k=self.k : 1 - np.exp(-(ws / A) ** k)
        dws = np.diff(ws,0)
        return cdf(ws+dws) - cdf(ws-dws)  
        
    def __call__(self, turbine_positions, wdir, wsp):
        
        WS = np.broadcast_to(wsp, (len(turbine_positions), len(wdir),len(wsp)))
        weight = self.weibull_weight(self.a, self.k, WS)
        #TODO: add weight from wdir dist
        return WS, np.zeros_like(WS)+self.ti, weight
                             
