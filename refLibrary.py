import numpy as np
import math

class refSignal(object):

    def __init__(self, N):
        self.N = N
        self.refsignal = np.zeros((N,1), dtype = float)

    def stepCMD(self):
        self.refsignal[math.floor(self.N/5): 2*math.floor(self.N/5)] = 1
        self.refsignal[3*math.floor(self.N/5): 4*math.floor(self.N/5)] = -1

    def regCMD(self):
        self.refsignal = np.zeros((self.N,1), dtype = float)

