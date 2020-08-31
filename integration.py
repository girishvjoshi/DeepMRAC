import numpy as np

class Integrate(object):

    def __init__(self):
        pass

    def rk4(self, sys, state, action, timeStep, subSteps):
        for i in range(0, subSteps):
            k1 = sys(state, action)
            k2 = sys(state + 0.5*timeStep*k1, action)
            k3 = sys(state + 0.5*timeStep*k2, action)
            k4 = sys(state + timeStep*k3, action)

        xstep = state + timeStep/6*(k1 + 2*k2 + 2*k3 + k4)

        return xstep
    
    def euler(self, sys, state, action, timeStep, subSteps):
        k1 = sys(state, action)
        xstep = state + timeStep*k1

        return xstep
