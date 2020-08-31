from wingrock import wingRock
import tensorflow as tf 
from bnn_controller import MRAC
from refmodel import refModel
from refLibrary import refSignal
import numpy as np
import matplotlib.pyplot as plt


learningON = 1
sim_endTime = 100
start_state = np.reshape([1,1],(2,1))
env = wingRock(start_state)
ref_env = refModel(start_state)
N = int(sim_endTime/env.timeStep)
ref_cmd = refSignal(N)

def main():

    with tf.Session() as sess:
    
        agent = MRAC(sess,2,1,10)

        sess.run(tf.global_variables_initializer())

        ref_cmd.stepCMD()
        n_idx = 0

        pos_rec = [start_state[0]]
        ref_pos_rec = [start_state[0]]
        vel_rec = [start_state[1]]
        ref_vel_rec = [start_state[1]]
        ref_rec = [0]

        for idx in range(0, N):

            adap_cntrl = agent.total_Cntrl(env.state, ref_env.state, ref_cmd.refsignal[n_idx])
            env.applyCntrl(adap_cntrl)
            ref_env.stepRefModel(ref_cmd.refsignal[n_idx])
            pos_rec.append(env.state[0])
            ref_pos_rec.append(ref_env.state[0])
            vel_rec.append(env.state[1])
            ref_vel_rec.append(ref_env.state[1])
            ref_rec.append(ref_cmd.refsignal[n_idx])
            n_idx = n_idx+1

    
    plt.figure(1)
    ax1 = plt.subplot(211)
    plt.plot(pos_rec, color='red', label='$x(t)$')
    plt.plot(ref_pos_rec, color='black', linestyle='--', label='$x_{rm}(t)$')
    plt.plot(ref_rec, label='$r(t)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Position $x(t)$')
    plt.title('Deep-MRAC with $\\nu_{ad}=W^{T}\phi^\sigma_{n}(x)$')
    # plt.legend()

    ax2=plt.subplot(212)
    plt.plot(vel_rec, color='red', label='$\dot{x}(t)$')
    plt.plot(ref_vel_rec, color='black', linestyle='--', label='$\dot{x}_{rm}(t)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Velocity $\dot{x}(t)$')
    plt.legend()
    # plt.show()

    # print(env.TRUE_DELTA_REC)
    plt.figure(3)
    plt.plot(agent.ADAP_CNTRL_REC, color='red', label='$\\nu_{ad}$')
    plt.plot(env.TRUE_DELTA_REC, color='black', linestyle='--', label='$\Delta(x)$')
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Uncertainty $\Delta(x)$')
    plt.legend()
    plt.title('Deep-MRAC with $\\nu_{ad}=W^{T}\phi^\sigma_{n}(x)$')
    # plt.show()

    # plt.figure(4)
    # dDist = np.reshape(agent.DADAP_CNTRL_REC,(2000,2))
    # plt.plot(dDist, color='red', label='$\\d(nu_{ad})$')

    data = np.reshape(agent.NET_PARAM,(N,agent.n_hidden_layer3))
    plt.figure(5)
    plt.plot(data)
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('Outer Layer Weights')
    plt.title('Deep-MRAC with $\\nu_{ad}=W^{T}\phi^\sigma_{n}(x)$')
    plt.show()

    # data1 = np.reshape(agent.cntrl_SAMPLES,(N, 5))
    # data1_mean = np.mean(data1,axis=1)
    # data1_var = 2*(np.var(data1, axis=1))+0.05
    # X = np.linspace(0,N, len(data1))
    # print(data1_var)
    # data1_up = data1_mean+data1_var
    # data1_dwn = data1_mean-data1_var
    # plt.figure(6)
    # plt.fill_between(X,-np.array(data1_up, dtype=float), -np.array(data1_dwn, dtype=float),color='#ffc78f')
    # # plt.plot(X,-data1_up)
    # # plt.plot(X,-data1_dwn)
    # # plt.plot(X,-data1_mean)
    # plt.plot(agent.ADAP_CNTRL_REC, color='red', label='$\\nu_{ad}$')
    # plt.plot(env.TRUE_DELTA_REC, color='black', linestyle='--', label='$\Delta(x)$')
    # plt.grid(True)
    # plt.xlabel('time')
    # plt.ylabel('Uncertainty $\Delta(x)$')
    # plt.legend()
    # plt.title('Deep-MRAC with $\\nu_{ad}=W^{T}\phi^\sigma_{n}(x)$')
    # plt.show()

if __name__ == '__main__':
    main()