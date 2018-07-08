from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

class Network:
    def __init__(self):
        print('init network')
        self.input_size = [48,1]

        self.num_convolutions = 5
        self.division_factor = pow(2,self.num_convolutions)
        factor = 24
        #Filters dimmensions

        #First convolution
        self.rf_1 = 11
        self.nin_1 = self.input_size[1]
        self.nf_1 = 1 * factor
        #Second convolution
        self.rf_2 = 4
        self.nf_2 = 2 * factor
        # Third convolution
        self.rf_3 = 4
        self.nf_3 = 4 * factor
        # forth convolution
        self.rf_4 = 3
        self.nf_4 = 6 * factor
        # fifth convolution
        self.rf_5 = 3
        self.nf_5 = 8 * factor

        self.cnn_fc = 1 * self.nf_5
        #Fully connected
        self.fc_1 = 32 * factor
        self.fc_2 = 32 * factor



        self.num_class = 27

        self.learning_rate = 1e-4
        #1e-5 = 59%


        self.max_steps = 30



        self.sess = tf.Session()

        self.build()
        self.saver = tf.train.Saver()


    def build(self):


        def weight_variable(shape):
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)

        def bias_variable(shape):
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)

        def conv2d(x, W):
          return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):
          return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                strides=[1, 2, 2, 1], padding='SAME')

        def max_pool_3x3(x):
          return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                                strides=[1, 3, 3, 1], padding='SAME')
        # Input layer
        self.x = tf.placeholder(tf.float32, [None, self.input_size[0], self.input_size[0], self.input_size[1]], name='x')
        self.y_ = tf.placeholder(tf.float32, [None, self.num_class],  name='y_')
        x_image = self.x
        # Convolutional layer 1
        W_conv1 = weight_variable([self.rf_1, self.rf_1, self.nin_1, self.nf_1])
        b_conv1 = bias_variable([self.nf_1])

        h_conv1 = tf.nn.leaky_relu(conv2d(x_image, W_conv1) + b_conv1,alpha=0.1)
        h_pool1 = max_pool_2x2(h_conv1)

        # Convolutional layer 2
        W_conv2 = weight_variable([self.rf_2, self.rf_2, self.nf_1, self.nf_2])
        b_conv2 = bias_variable([self.nf_2])

        h_conv2 =tf.nn.leaky_relu(conv2d(h_pool1, W_conv2) + b_conv2,alpha=0.1)
        h_pool2 = max_pool_2x2(h_conv2)

        self.keep_prob = tf.placeholder(tf.float32)
        h_pool2 = tf.nn.dropout(h_pool2, self.keep_prob)
        #Convolution layer 3
        W_conv3 = weight_variable([self.rf_3, self.rf_3, self.nf_2, self.nf_3])
        b_conv3 = bias_variable([self.nf_3])

        h_conv3 = tf.nn.leaky_relu(conv2d(h_pool2, W_conv3) + b_conv3,alpha=0.1)
        h_pool3 = max_pool_2x2(h_conv3)
        # Convolution layer 4
        W_conv4 = weight_variable([self.rf_4, self.rf_4, self.nf_3, self.nf_4])
        b_conv4 = bias_variable([self.nf_4])

        h_conv4 = tf.nn.leaky_relu(conv2d(h_pool3, W_conv4) + b_conv4,alpha=0.1)
        h_pool4 = max_pool_2x2(h_conv4)
        # Convolution layer 5
        W_conv5 = weight_variable([self.rf_5, self.rf_5, self.nf_4, self.nf_5])
        b_conv5 = bias_variable([self.nf_5])
        h_conv5 = tf.nn.leaky_relu(conv2d(h_pool4, W_conv5) + b_conv5,alpha=0.1)
        h_pool5 = max_pool_3x3(h_conv5)

        # Fully connected layer 1
        h_pool2_flat = tf.reshape(h_pool5, [-1, int(self.cnn_fc)])

        W_fc1 = weight_variable([int(self.cnn_fc), self.fc_1])
        b_fc1 = bias_variable([self.fc_1])

        h_fc1 = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,alpha=0.1)

        # Dropout

        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Fully connected layer 2 (Output layer)
        W_fc2 = weight_variable([self.fc_2, self.num_class])
        b_fc2 = bias_variable([self.num_class])

        out = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self.y = tf.nn.softmax(out, name='y')
        # Evaluation functions
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        l2 = 0.00001 * (tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(W_conv2))
        self.CE = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=out) + l2
        #self.print_1 = tf.Print(tf.argmax(self.y, 1),[tf.argmax(self.y, 1)], summarize=30, message="y")
        #self.print_2 = tf.Print(tf.argmax(self.y_, 1),[tf.argmax(self.y_, 1)], summarize=30, message="y_")


        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        # correct_prediction = tf.equal(self.print_1, self.print_2)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        # Training algorithm
        # self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.CE)

        self.sess.run(tf.initialize_all_variables())





    def store_checkpoint(self,DIR):
        save_path = self.saver.save(self.sess, DIR)
        print("Model saved in file: %s" % save_path)

    def loadNetwork(self,DIR):
        self.saver.restore(self.sess, DIR)
        print("Model restored.")

    def save_model_as_pb(self,DIR):
        tf.train.write_graph(self.sess.graph_def, DIR, 'my_model.pb')

    def train_batch(self, batch_xs, batch_ys):
        #print("trining batch", batch_xs, batch_ys)
        lr,acc,loss= self.sess.run([self.train_step, self.accuracy, self.CE],
                      feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 0.8})
        return lr,acc, loss

    def test_batch(self, batch_xs, batch_ys):
        result = self.sess.run([self.accuracy, self.y], feed_dict={self.x: batch_xs, self.y_: batch_ys, self.keep_prob: 1.0})
        return result

    def feed_batch(self, batch_xs):
        return self.sess.run(self.y, feed_dict={self.x: batch_xs, self.keep_prob: 1.0})




# if __name__ =='__main__':
#     network =Network()
#     max_steps = 10000
#     for step in range(max_steps):
#         batch_xs, batch_ys = mnist.train.next_batch(50)
#         network.train_batch(batch_xs, batch_ys)
#         if (step % 100) == 0:
#             print(step, network.test_batch(mnist.test.images, mnist.test.labels))
#     print(max_steps, network.test_batch(mnist.test.images, mnist.test.labels))