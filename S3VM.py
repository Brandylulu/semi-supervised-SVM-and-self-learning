import numpy as np
import sklearn
import tensorflow as tf
tf.random.set_random_seed(1996)
from layers import *


def k_neighbor(k, data, sample, inverse=False):
    """
    k_neighbor: return 1st to kth nearest neighbors for a given sample point
    :param k:
    int, number of neighbors to be returned
    :param data:
    (N, dim)
    :param sample:
    (dim, )
    :param inverse:
    bool, if inverse return k the most farthest neighbors

    :return:
    (k, ) index array of neighbors
    """
    norm = np.linalg.norm(data - sample, axis=1)

    if not inverse:
        return np.argsort(norm)[:k]
    else:
        return np.argsort(norm)[-k:]


def RBF(X, gamma=1.0, kernel_dim=50):
    """
    RBF: return RBF kernel-ed data

    :param X:
    (N, dim)
    :param gamma:
    float, gamma for RBF kernel equation
    :param kernel_dim:
    int, number of basis

    :return:
    (N, dim), transformed data
    """
    random = np.random.RandomState(1)
    random_weights = (np.sqrt(2 * gamma) * random.normal(size=(X.shape[1], kernel_dim)))
    random_offset = random.uniform(0, 2 * np.pi, size=kernel_dim)

    project = X @ random_weights + random_offset
    X_tran = np.sqrt(2) / np.sqrt(kernel_dim) * np.cos(project)

    a = np.ones(len(X_tran))
    X_tran = np.c_[X_tran, a]

    return X_tran


class LabelCache():
    def __init__(self):
        self.maxsize = 100 # From Karlen et al., 'Large Scale Manifold Transduction'
        self.cache = []
        self.cache_ths = int(self.maxsize*0.5)
        self.input_ths = 0.6

    def add(self, input):
        if len(self.cache) + len(input) < self.maxsize:
            self.cache.extend(input)
        else:
            self.cache[0:len(self.cache) + len(input) - self.maxsize] = []
            self.cache.extend(input)

    def do_update(self, input):
        frac_input = sum(input)/len(input)

        # if 'y=1' is too overwhelmed
        if sum(self.cache) > self.cache_ths:
            if frac_input >= self.input_ths:
                return False
            else:
                return True
        # if 'y=-1' is too overwhelmed
        elif sum(self.cache) < -self.cache_ths:
            if frac_input <= -self.input_ths:
                return False
            else:
                return True
        else:
            return True


class S3VM:
    def __init__(self, data, sess, y=None, kernel='RBF', kernel_dim=50, lr_init=0.5, lr_decay=0.001, pest=0.5, gamma=1.0,
                 k_cluster=10, k_unlabel=5, w_dim=None, lamb=1.0, epoch=50, batch=4, self_training=False, self_training_ths=10000,
                 min_unlabel_data=None, verbose=False, verbose_loss=False):
        self.data = data
        self.sess = sess
        self.y = y
        self.kernel = kernel
        self.kernel_dim = kernel_dim
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.pest = pest
        self.gamma = gamma
        self.k_cluster = k_cluster
        self.k_unlabel = k_unlabel
        self.epoch = epoch
        self.batch = batch
        self.w_dim = w_dim
        self.n_data = data.shape[0]
        self.label = np.zeros((self.n_data,))
        self.num_train = 0
        self.lamb = lamb
        self.verbose = verbose
        self.verbose_loss = verbose_loss
        self.self_training = self_training
        self.self_training_ths = self_training_ths
        self.min_unlabel_data = min_unlabel_data

    def init_cluster(self):
        # avg = np.mean(self.data, 0)
        # mode1_sub_ind = np.array(self.n_data*np.random.rand(30)).astype(int)
        # mode1_sub_data = self.data[mode1_sub_ind, :]
        # furthest_from_mean = k_neighbor(1, mode1_sub_data, avg, inverse=True)
        # mode1_ind = mode1_sub_ind[furthest_from_mean]
        mode1_ind = int(self.n_data * np.random.rand(1))
        mode1 = self.data[mode1_ind, :]
        mode2_ind = k_neighbor(1, data=self.data, sample=mode1, inverse=True)[0]

        cluster1 = k_neighbor(self.k_cluster, data=self.data, sample=mode1, inverse=False)
        cluster2 = k_neighbor(self.k_cluster, data=self.data, sample=self.data[mode2_ind, :], inverse=False)

        return np.r_[mode1_ind, cluster1], np.r_[mode2_ind, cluster2]

    def init_weight_bias(self, dim):
        self.w = np.random.normal(scale=0.2, size=dim)
        self.b = np.random.normal(size=1) * 0.05

    def decision_function(self, x):
        if self.kernel == 'RBF':
            x = RBF(x, self.gamma, self.kernel_dim)
        output = self.sess.run(self.output, feed_dict={self.x: x})
        return np.sign(np.squeeze(output))

    def sgd(self, x, y):
        # Example for single-layer, single input and label
        self.lr = self.lr_init / (1 + self.lr_decay * self.num_train)
        hinge_loss = 0 > 1 - y * (np.matmul(self.w, x) + self.b)
        if hinge_loss:
            grad = self.lamb * self.w
        else:
            grad = -y * x + self.lamb * self.w
            self.b += self.lr * y
        self.w -= self.lr * grad

    def draw_batch(self, data, counter):
        batch_data = data[counter*self.batch:(counter+1)*self.batch]
        if batch_data.shape[1] > self.data.shape[1]:
            return batch_data[:, :self.data.shape[1]], np.expand_dims(batch_data[:, -1], axis=-1)
        else:
            return batch_data

    def draw_graph(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.lr = tf.train.exponential_decay(self.lr_init, global_step=self.global_step, decay_steps=1000, decay_rate=0.9, staircase=True)

        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.data.shape[1]))
        self.label = tf.placeholder(dtype=tf.float32, shape=(None, 1))

        x2, w1, _ = linear(self.x, 10, with_w=True, name="linear1")
        x2 = tf.tanh(x2)
        l2_loss = tf.nn.l2_loss(w1)

        x3, w2, _ = linear(x2, 10, with_w=True, name="linear2")
        x3 = tf.tanh(x3)
        l2_loss += tf.nn.l2_loss(w2)

        x4, w3, _ = linear(x3, 10, with_w=True, name="linear3")
        x4 = tf.tanh(x4)
        l2_loss += tf.nn.l2_loss(w3)

        x5, w4, _ = linear(x4, 1, with_w=True, name="linear4")
        l2_loss += tf.nn.l2_loss(w4)
        output = tf.tanh(x5)
        self.output = output

        hinge_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(self.label, output))))
        loss = hinge_loss + self.lamb*l2_loss

        self.loss = loss
        self.hinge_loss = hinge_loss
        self.l2_loss = l2_loss
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(loss)

        tf.summary.scalar('loss/hinge_loss', self.hinge_loss)
        tf.summary.scalar('loss/l2_norm', self.l2_loss)
        tf.summary.scalar('loss/total_loss', self.loss)

    def train(self):
        # Initialize binray clusters
        cluster1, cluster2 = self.init_cluster()
        self.cluster1 = cluster1
        self.cluster2 = cluster2
        labeled_ind = np.r_[cluster1, cluster2]
        ind = np.ones((self.n_data,), bool)
        ind[labeled_ind] = False
        unlabeld_ind = ind

        if self.y is not None:
            if self.y[cluster1][0] == 1:
                self.label[cluster1] = 1
                self.label[cluster2] = -1
            else:
                self.label[cluster1] = -1
                self.label[cluster2] = 1
        else:
            self.label[cluster1] = 1
            self.label[cluster2] = -1

        if self.kernel == 'RBF':
            self.data = RBF(self.data, gamma=self.gamma, kernel_dim=self.kernel_dim)
        self.init_weight_bias(self.data.shape[1])

        x_label = self.data[labeled_ind, :]
        y_label = self.label[labeled_ind]
        x_unlabel = self.data[unlabeld_ind, :]

        train_data_length = len(x_label)
        print(train_data_length)
        num_batch_per_train_epoch = int(train_data_length / self.batch)
        num_batch_per_train_epoch_init = num_batch_per_train_epoch

        buffer = LabelCache()

        self.draw_graph()
        self.summaries = tf.summary.merge_all()

        init = tf.initializers.global_variables()

        self.sess.run(init)
        summary_writer = tf.summary.FileWriter('./log', self.sess.graph)

        for i in range(self.epoch):
            s = np.arange(x_label.shape[0])
            np.random.shuffle(s)
            x_label_shuffle = x_label[s]
            y_label_shuffle = y_label[s]

            s = np.arange(x_unlabel.shape[0])
            np.random.shuffle(s)
            x_unlabel_shuffle = x_unlabel[s]

            counter = 0
            for j in range(num_batch_per_train_epoch):
                # train labeled data
                x_label_batch, y_label_batch = self.draw_batch(np.c_[x_label_shuffle, y_label_shuffle], counter)
                # self.sgd(x_label_batch, y_label_batch)
                _, summary, step, loss = \
                    self.sess.run([self.opt, self.summaries, self.global_step, self.loss], feed_dict={self.x: x_label_batch, self.label: y_label_batch})
                summary_writer.add_summary(summary, global_step=step)

                if self.verbose_loss:
                    print('INFO: {} epoch / {} batch / labeled:\t loss: {}'.format(i, j, loss))

                # train unlabeled data
                x_unlabel_batch = self.draw_batch(x_unlabel_shuffle, counter)
                y_unlabel_batch = [np.sign(np.sum(y_label[k_neighbor(self.k_unlabel, x_label, item)])) for item in x_unlabel_batch]

                test = y_unlabel_batch
                buffer.add(y_unlabel_batch)

                y_unlabel_batch = np.expand_dims(y_unlabel_batch, axis=-1)
                if i < 2000 or len(buffer.cache) < buffer.maxsize*0.8 or buffer.do_update(test):
                    _, summary, step, loss = self.sess.run([self.opt, self.summaries, self.global_step, self.loss], feed_dict={self.x: x_unlabel_batch, self.label: y_unlabel_batch})
                    summary_writer.add_summary(summary, global_step=step)
                elif self.verbose:
                    print('INFO: {} epoch / {} batch\tBalance constraint; grad is not updated'.format(i, j))
                counter += 1

                if self.verbose_loss:
                    print('INFO: {} epoch / {} batch / un-labeled:\t loss: {}'.format(i, j, loss))

            # Do Self training if all conditions are met
            if self.self_training and i > self.self_training_ths and i % 5 == 0 and len(x_unlabel) > self.min_unlabel_data:
                vote = np.array([np.sum(y_label[k_neighbor(self.k_unlabel, x_label, item)]) for item in x_unlabel_batch])
                label_for_new = np.squeeze(y_unlabel_batch[(vote >= int(self.k_unlabel * 0.5)+1) | (vote <= int(-self.k_unlabel * 0.5)-1)])
                new_labeled_data = x_unlabel_batch[(vote >= int(self.k_unlabel * 0.5)+1) | (vote <= int(-self.k_unlabel * 0.5)-1)]

                x_label = np.r_[x_label, new_labeled_data]
                y_label = np.r_[y_label, label_for_new]
                x_unlabel = np.array([item for item in x_unlabel if item not in new_labeled_data])

                if self.verbose:
                    print('Do self-labeling...')
                    if len(new_labeled_data) > 0:
                        print('{} data is assigned new label'.format(len(new_labeled_data)))
                        print('vote results for each data : {}'.format(vote))
                    else:
                        print('No new labeled data')

                if num_batch_per_train_epoch_init*3 > num_batch_per_train_epoch:
                    num_batch_per_train_epoch = int(len(x_label) / self.batch)