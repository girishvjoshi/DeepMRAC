import numpy as np
from integration import Integrate

class refModel(Integrate):

    def __init__(self, start_state):
        self.state = start_state
        self.timeStep = 0.05
        self.naturalFreq = 2
        self.damping = 0.5
        self.substeps = 1
        self.recordSTATE = self.state

    def stepRefModel(self, ref_signal):
        self.state = self.simModel(ref_signal)

    def dynamicRefModel(self, state, ref_signal):
        x1dot = state[1]
        x2dot =  -self.naturalFreq**2*state[0]-2*self.damping*self.naturalFreq*state[1] + self.naturalFreq**2*ref_signal
        xdot = np.reshape([x1dot, x2dot], (2,1))
        return xdot

    def simModel(self, ref_signal):

        xstep = self.euler(self.dynamicRefModel, self.state, ref_signal, self.timeStep, self.substeps)
        
        return xstep
