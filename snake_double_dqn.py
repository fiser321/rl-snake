#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys,random,cv2
import snake_env as game
from collections import deque

GAME = 'snake' # the name of the game being played for log files
ACTIONS = 4 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 10000 # timesteps to observe before training
EXPLORE = 500000 # frames over which to anneal epsilon
FINAL_EPSILON = 0.001 # final value of epsilon
#INITIAL_EPSILON = 0.0001 # starting value of epsilon
INITIAL_EPSILON = 0.00001 # starting value of epsilon
#REPLAY_MEMORY = 50000 # number of previous transitions to remember
REPLAY_MEMORY = 10000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1

class DoubleDqn(object):
    def __init__(self):

        self.epsilon_max = 1
        self.replace_target_iter = 600
        self.memory_size = 1000
        self.gamma = 0.99
        self.batch_size = 32
        self.epsilon = 0.00
        self.learn_step_counter = 0
        self.epsilon_increment = 0.001
        self.sess = tf.Session()
        self.sess = tf.InteractiveSession()
        #self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        pass

    def weight_variable(self, shape, c_names, name):
        #initial = tf.truncated_normal(shape, stddev=0.01)
        initial = tf.random_normal_initializer(0., 0.3)
        return tf.get_variable(name, shape, initializer=initial, collections=c_names)

    def bias_variable(self, shape, c_names, name):
        #initial = tf.constant(0.01, shape=shape)
        initial = tf.constant_initializer(0.1)
        return tf.get_variable(name, shape, initializer=initial, collections=c_names)

    def conv2d(self, x, W, stride):
        return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def createNetwork(self, s, c_names):
        # network weights
        W_conv1 = self.weight_variable([8, 8, 4, 32], c_names, name="w1")
        b_conv1 = self.bias_variable([32], c_names, name = "b1")

        W_conv2 = self.weight_variable([4, 4, 32, 64], c_names, name="w2")
        b_conv2 = self.bias_variable([64],c_names, name="b2")

        W_conv3 = self.weight_variable([3, 3, 64, 64], c_names, name="w3")
        b_conv3 = self.bias_variable([64], c_names, name="b3")

        W_fc1 = self.weight_variable([2304, 512], c_names, name="wf1")
        b_fc1 = self.bias_variable([512], c_names, name="bf1")

        W_fc2 = self.weight_variable([512, ACTIONS], c_names, name="wf2")
        b_fc2 = self.bias_variable([ACTIONS], c_names, name="bf2")

        # input layer
        #s = tf.placeholder("float", [None, 84, 84, 4])
        # hidden layers
        h_conv1 = tf.nn.relu(self.conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)
        # h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 2304])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        # readout layer
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2
        return readout

    def train_net(self):

        self.q_target = tf.placeholder(tf.float32, [None, ACTIONS], name='Q_target')
        self.s = tf.placeholder("float", [None, 84, 84, 4])
        self.s_ = tf.placeholder("float", [None, 84, 84, 4])
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        self.q_eval = self.createNetwork(self.s, c_names=c_names)
        with tf.variable_scope('loss'):
            #readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = self.createNetwork(self.s_, c_names=c_names)

        #game_state = game.GameEnv()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        game_state = game.gameState()
        #game_state.reset()
        self.D = deque()
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = game_state.frameStep(do_nothing)
        x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

        epsilon = INITIAL_EPSILON
        t = 0
        while 1 != 0:
            # choose an action epsilon greedily
            readout_t = self.q_eval.eval(feed_dict={self.s: [s_t]})[0]
            a_t = np.array([0,0,0,0])
            action_index = 0
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
                #print("----------Random Action----------", a_t)
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
            #scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            x_t1_colored, r_t, terminal = game_state.frameStep(a_t)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (84, 84)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (84, 84, 1))
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
            self.D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(self.D) > REPLAY_MEMORY:
                self.D.popleft()
            if t > OBSERVE:
                self.learn()
            s_t = s_t1
            t += 1
            if terminal:
                 x1,x2 = game_state.retScore()
                 print("epsilon:  ", epsilon, "action: ", action_index, "t:  ", t, "sc:  ", x1, "ep:  ", x2)
            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(self.sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_op)
            print('\ntarget_params_replaced\n')
        minibatch = random.sample(self.D, self.batch_size)
        # get the batch variables
        s_j_batch = [d[0] for d in minibatch]
        a_batch = [d[1] for d in minibatch]
        r_batch = [d[2] for d in minibatch]
        s_j1_batch = [d[3] for d in minibatch]
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: s_j1_batch,
                self.s: s_j1_batch,
            })
        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: s_j_batch})
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        reward = r_batch
        #new_reward = np.max(q_next, axis=1)
        max_act4next = np.argmax(q_eval4next, axis=1)
        new_reward = q_next[batch_index, max_act4next]
        eval_act_index = [np.argmax(e) for e in a_batch]
        target_reward = [reward[index] if minibatch[index][4] else reward[index] + self.gamma * new_reward[index] for index in range(self.batch_size)]
        #q_target[batch_index, eval_act_index] = reward + self.gamma * new_reward
        q_target[batch_index, eval_act_index] = target_reward
        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.s: s_j_batch,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)
        self.learn_step_counter += 1

if __name__ == "__main__":
    RL = DoubleDqn()
    RL.train_net()
