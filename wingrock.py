import numpy as np
from integration import Integrate
# import tensorflow as tf

class wingRock(Integrate):

    def __init__(self, start_state):
        self.state = start_state
        self.timeStep = 0.05
        self.trueWeights = np.array([0.2314, 0.06918, -0.6245, 0.0095, 0.0214])#np.array([0.2314, 0.14, -0.6245, 0.25, 0.214])#
        self.lDelta = 1
        self.substeps = 1
        self.recordSTATE = self.state
        self.recordTRUE_UNCERTAINTY = 0
        #Data Recording
        self.TRUE_DELTA_REC = []

    def applyCntrl(self, action):
        self.state = self.simModel(action)
    
    def dynamicswingRock(self, state, action):
        delta = self.trueWeights[0]*state[0] + self.trueWeights[1]*state[1] + self.trueWeights[2]*np.abs(state[0])*state[1] + self.trueWeights[3]*np.abs(state[1])*state[1] + self.trueWeights[4]*state[1]**3
        self.TRUE_DELTA_REC.append(delta[0])
        x1dot = self.state[1]
        x2dot = delta + self.lDelta*action
        xdot = np.reshape([x1dot, x2dot],(2,1))
        return xdot
    
    def simModel(self, action):
        
        xstep = self.euler(self.dynamicswingRock, self.state, action, self.timeStep, self.substeps)
        
        return xstep






