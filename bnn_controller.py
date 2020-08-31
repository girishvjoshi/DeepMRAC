import numpy as np
import tensorflow as tf
from replay_buffer import ReplayBuffer
import scipy.linalg as sp

#Defining Basyesian NN variances
stddev_var = 1.0
log_stddev_var = 0.001
epsilon_prior = 0.1
sigma_prior = 0.01

class MRAC(object):

    def __init__(self, sess, state_dim, action_dim, MRAC_gain = 0.1,lr = 0.05, lr_flag=1, buffer_size = 5000):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        naturalFreq = 2
        damping = 0.5
        self.timeStep = 0.05
        self.gain = MRAC_gain
        self.lr = lr
        self.lr_decay = 1
        self.lr_status = lr_flag
        self.buffer_size = buffer_size
        self.batch_size = 100
        self.feedbackGainP = -naturalFreq^2
        self.feedbackGainD = -2*damping*naturalFreq
        A = np.reshape([0,1,self.feedbackGainP,self.feedbackGainD], (2,2))
        Q = 1*np.eye(2)
        self.B = np.reshape([0,1],(2,1))
        self.P = sp.solve_lyapunov(A.transpose(),Q)

        #Replay Buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Network Definition
        self.n_hidden_layer1 = 100
        self.n_hidden_layer2 = 200
        self.n_hidden_layer3 = 10
        self.netWeights = np.zeros((self.n_hidden_layer3,1), dtype = float)
        self.phi = np.zeros((self.n_hidden_layer3,1), dtype = float)
        self._placeholders()
        self.basis, self.out, self.sampled_net_params = self._adapNet()
        self.network_params = tf.trainable_variables()
        
        # Calculate the prior 
        self.regualizer = self._regularizer(self.sampled_net_params, self.network_params)
        # Likelihood
        self.sample_log_likelihood = tf.reduce_sum(self._log_gaussian(self.delta_ph, self.out, sigma_prior))
        #Calculating the expected Lower bound
        self.elbo = -self.sample_log_likelihood + self.regualizer/(self.n_hidden_layer1+self.n_hidden_layer2)
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.elbo)

        #Data Recording
        self.TOTAL_CNTRL_REC = []
        self.ADAP_CNTRL_REC = []
        self.NET_PARAM = []
        self.cntrl_SAMPLES = []

    def _placeholders(self):
        self.state_ph = tf.placeholder(tf.float32, (None, 2), name='state')
        self.delta_ph = tf.placeholder(tf.float32, (None, self.action_dim), name='true_uncertainty')

    def _adapNet(self):
        w1_mu, w1_logvar, epsilon_w1 = self._weight_init([self.state_dim, self.n_hidden_layer1], 'w1')
        b1_mu, b1_logvar, epsilon_b1 = self._base_init([self.n_hidden_layer1], 'b1')

        w2_mu, w2_logvar, epsilon_w2 = self._weight_init([self.n_hidden_layer1,self.n_hidden_layer2], 'w2')
        b2_mu, b2_logvar, epsilon_b2 = self._base_init([self.n_hidden_layer2], 'b2')

        w3_mu, w3_logvar, epsilon_w3 = self._weight_init([self.n_hidden_layer2,self.n_hidden_layer3], 'w3')
        b3_mu, b3_logvar, epsilon_b3 = self._base_init([self.n_hidden_layer3], 'b3')

        w4_mu, w4_logvar, epsilon_w4 = self._weight_init([self.n_hidden_layer3,self.action_dim], 'w4')
        b4_mu, b4_logvar, epsilon_b4 = self._base_init([self.action_dim], 'b4')

        w1 = w1_mu + tf.multiply(tf.log(1. + tf.exp(w1_logvar)), epsilon_w1)
        w2 = w2_mu + tf.multiply(tf.log(1. + tf.exp(w2_logvar)), epsilon_w2)
        w3 = w3_mu + tf.multiply(tf.log(1. + tf.exp(w3_logvar)), epsilon_w3)
        w4 = w4_mu + tf.multiply(tf.log(1. + tf.exp(w4_logvar)), epsilon_w4)

        b1 = b1_mu + tf.multiply(tf.log(1. + tf.exp(b1_logvar)), epsilon_b1)
        b2 = b2_mu + tf.multiply(tf.log(1. + tf.exp(b2_logvar)), epsilon_b2)
        b3 = b3_mu + tf.multiply(tf.log(1. + tf.exp(b3_logvar)), epsilon_b3)
        b4 = b4_mu + tf.multiply(tf.log(1. + tf.exp(b4_logvar)), epsilon_b4)

        h1 = tf.nn.tanh(tf.matmul(self.state_ph, w1)+b1)
        h2 = tf.nn.tanh(tf.matmul(h1, w2)+b2)

        basis = tf.nn.tanh(tf.matmul(h2, w3)+b3)
        
        out = tf.matmul(basis, w4)+b4

        return basis, out, [w1, b1, w2, b2, w3, b3, w4, b4]

    def evalCNTRL(self, state):
        self.sess.run(self.out, feed_dict={self.state_ph: np.reshape(state, [1, self.state_ph])})

    def updateBasis(self, state):
        basis = self.sess.run(self.basis, feed_dict={self.state_ph:np.reshape(state, [1, self.state_dim])})
        return basis

    def _weight_init(self, shape, var_name):
        initial_mu = tf.truncated_normal(shape, stddev = stddev_var)
        initial_logvar = tf.truncated_normal(shape, mean=0.0, stddev = log_stddev_var)
        epsilon_w = self._get_random(shape, 0.0, epsilon_prior)
        return tf.Variable(initial_mu, name=var_name+'mean'), tf.Variable(initial_logvar, name=var_name+'_logvar'), epsilon_w

    def _base_init(self, shape, var_name):
        initial_value = tf.truncated_normal(shape, mean=0.0, stddev = stddev_var)
        epsilon_b = self._get_random(shape, 0.0, epsilon_prior)
        return tf.Variable(initial_value, name=var_name+'mean'),  tf.Variable(initial_value, name=var_name+'log_var'), epsilon_b       

    def _get_random(self, shape, mu, std_dev):
        return tf.random_normal(shape, mean=mu, stddev = std_dev)

    def _log_gaussian(self, x, mu, sigma):
        return -0.5*tf.log(2*np.pi)-tf.log(sigma)-(x-mu)**2/(2*sigma**2)

    def _regularizer(self,W,net_params):
        [w1,w1l, b1, b1l,w2,w2l, b2, b2l,w3,w3l, b3, b3l,w4,w4l, b4, b4l] = net_params
        mean_vec = [w1, b1, w2, b2, w3, b3, w4, b4]
        log_var_vec = [w1l, b1l, w2l, b2l, w3l, b3l, w4l, b4l]
        for i in range(len(W)):
            sample_log_pw, sample_log_qw = 0., 0.
            sample_log_pw += tf.reduce_sum(self._log_gaussian(W[i], 0., sigma_prior))
            sample_log_qw += tf.reduce_sum(self._log_gaussian(W[i], mean_vec[i], tf.log(1. + tf.exp(log_var_vec[i]))))

        regualizer = tf.reduce_sum(sample_log_qw-sample_log_pw)

        return regualizer

    def total_Cntrl(self, state, ref_state, ref_signal):
        lin_cntrl = self.linear_Cntrl(state, ref_signal)
        adap_cntrl = self.mrac_Cntrl(state, ref_state)
        total_cntrl = lin_cntrl + adap_cntrl
        self.TOTAL_CNTRL_REC.append(total_cntrl[0])
        self.ADAP_CNTRL_REC.append(-adap_cntrl[0])
        return total_cntrl

    def linear_Cntrl(self, state, ref_signal):
        fb = self.feedbackGainP*state[0]+self.feedbackGainD*state[1]
        ff = -self.feedbackGainP*ref_signal
        cntrl = fb+ff
        return cntrl

    def mrac_Cntrl(self, state, ref_state):
        self.phi = self.updateBasis(state)
        # print(self.phi)
        if self.lr_status:
            self.updateNetWeights(state, ref_state)
            # print(self.netWeights)
            self.NET_PARAM.append(np.reshape(self.netWeights, (1,10)))
        
        cntrl = np.dot(self.phi, self.netWeights)

        sample_cntrl = self.draw_samples(state)

        self.cntrl_SAMPLES.append(np.reshape(sample_cntrl, (1,5)))
        
        self.replay_buffer.add(state, cntrl)       
        # Update the Features
        if self.replay_buffer.size() % 100 == 0 and self.replay_buffer.size() > 0:
            self.updateDMRAC_NET()
        
        return cntrl 
        
    def updateNetWeights(self, state, ref_state):
        error = state-ref_state
        temp = np.dot(np.dot(np.dot(self.phi.transpose(), np.transpose(error)), self.P), self.B)
        self.netWeights = self.netWeights + self.timeStep*(self.gain*temp)

    def updateDMRAC_NET(self):
        N = self.batch_size
        for steps  in range(100):
            state, deltas = self.replay_buffer.sample_batch(N)
            self.sess.run(self.optimize, feed_dict={self.state_ph:np.reshape(state,[N, self.state_dim]), self.delta_ph:np.reshape(deltas,[N, self.action_dim])})

    def add_to_buffer(self, state, delta):
        self.replay_buffer.add(np.reshape(state,(self.state_dim,)), np.reshape(delta, (self.action_dim,)))

    def draw_samples(self, state):
        n_samples = 5
        sample = []
        for i in range(n_samples):
            basis = self.updateBasis(state)
            output = np.dot(basis, self.netWeights)
            sample.append(output)

        return sample
            

        