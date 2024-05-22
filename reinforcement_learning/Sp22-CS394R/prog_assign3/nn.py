import numpy as np
from algo import ValueFunctionWithApproximation

import tensorflow as tf

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        self.state_dims = state_dims
        self.X = tf.placeholder(tf.float32, shape=[None, state_dims])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])  # Single output of state value

        # Define the neural network layers
        layer1 = tf.layers.dense(self.X, 32, activation=tf.nn.relu)
        layer2 = tf.layers.dense(layer1, 32, activation=tf.nn.relu)
        out_layer = tf.layers.dense(layer2, 1)

        self.Y_hat = out_layer
        self.loss_op = 0.5 * tf.losses.mean_squared_error(self.Y, self.Y_hat)

        # Use Adam optimizer with default learning rate
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss_op)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def __call__(self,s):
        # TODO: implement this method
        s = np.reshape(s, (1, self.state_dims))
        pred_state_val = self.sess.run(self.Y_hat, feed_dict={self.X: s})
        return pred_state_val[0, 0]

    def update(self,alpha,G,s_tau):
        # TODO: implement this method
        s_tau = np.reshape(s_tau, (1, self.state_dims))
        G = np.reshape(G, (1, 1))
        self.sess.run(self.train_op, feed_dict={self.X: s_tau, self.Y: G})


