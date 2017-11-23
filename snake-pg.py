import tensorflow as tf
import numpy as np
import cv2

class PoliceGradient(object):

    def __init__(self, actions, gamma, observe, explore, learning_rate, batch_size):
        self.actions = actions
        self.gamma = gamma
        self.observe = observe
        self.explore = explore
        self.lr = learning_rate
        self.batch_size = batch_size
        self.ep_obs,self.ep_as,self.ep_rs = [],[],[]
        self.build_net()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        self.tf_obs = tf.placeholder(tf.float32, [None, 80, 80, 4], name="observations")
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.01)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.01, shape=shape)
            return tf.Variable(initial)
        def conv2d(x, W, stride):
            return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
        def max_pool_2x2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        W_conv1 = weight_variable([8, 8, 4, 32])
        b_conv1 = bias_variable([32])

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        W_fc1 = weight_variable([1600, 512])
        b_fc1 = bias_variable([512])
        W_fc2 = weight_variable([512, self.actions])
        b_fc2 = bias_variable([self.actions])

        # input layer
        #s = tf.placeholder("float", [None, 80, 80, 4])
        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(self.tf_obs, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        # h_pool2 = max_pool_2x2(h_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
        # h_pool3 = max_pool_2x2(h_conv3)
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        # readout layer
        all_act = tf.matmul(h_fc1, W_fc2) + b_fc2
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax
        neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
        loss = tf.reduce_mean(neg_log_prob * self.tf_vt)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        print(observation)
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: [observation]})
        v = prob_weights.ravel()
        print("v:.....................: ", v)
        action = np.random.choice(range(prob_weights.shape[1]), p=v)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        discounted_ep_rs_norm = self.discount_and_norm_rewards()
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs: self.ep_obs,
            self.tf_acts: np.array(self.ep_as),
            self.tf_vt: discounted_ep_rs_norm,
        })

        self.ep_rs,self.ep_as,self.ep_obs = [], [],[]
        return discounted_ep_rs_norm

    def discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

    def trans_screen(self, x_t, flag=False, old_t=None):
        x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
        if flag:
            x_t1 = np.reshape(x_t, (80, 80, 1))
            new_t = np.append(x_t1, old_t[:, :, :3], axis=2)
        else:
            new_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        return new_t

    def do_main(self):
        import snake_env as game
        game_state = game.GameEnv()
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_pg")
        if checkpoint and checkpoint.model_checkpoint_path:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
        t = 0
        game_state.reset()
        for e in range(30000):
            do_nothing = np.zeros(self.actions)
            do_nothing[0] = 1
            x_t, r_0, terminal = game_state.step(do_nothing)
            s_t = self.trans_screen(x_t)

            while True:
                a_t = np.zeros([self.actions])
                action = self.choose_action(s_t)
                print("action:      ", action)
                a_t[action] = 1
                x_t1_colored, r_t, terminal = game_state.step(a_t)
                s_t1 = self.trans_screen(x_t1_colored, flag=True, old_t=s_t)
                self.store_transition(s_t, action, r_t)

                if terminal:
                    vt = self.learn()
                    break
                s_t = s_t1
                t += 1

            if t % 10 == 0:
                saver.save(self.sess, 'saved_pg/' + 'snake' + '-dqn', global_step=t)

if __name__ == "__main__":
    RL = PoliceGradient(actions=4, gamma=0.99, observe=1000, explore=8000, learning_rate=0.01, batch_size=32)
    RL.do_main()