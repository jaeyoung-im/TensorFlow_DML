from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

"""Read MNIST Dataset"""
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot = True)


"""Make Function"""
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, mean=0.0, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


"""Building Architecture"""
x = tf.placeholder(tf.float32, [None, 784])

x_image = tf.reshape(x, [-1, 28, 28, 1])

w1 = weight_variable([5, 5, 1, 32], 'W_conv1')
b1 = bias_variable([32], 'b_conv1')

h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w1), b1))
h_pool1 = max_pool_2x2(h_conv1)

w2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b2 = bias_variable([64], name='b_conv2')

h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_pool1, w2), b2))
h_pool2 = max_pool_2x2(h_conv2)


w3 = weight_variable([7 * 7 * 64, 1024], name='W_fc1')
b3 = bias_variable([1024], name='b_fc1')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool2_flat, w3), b3))

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w4 = weight_variable([1024, 10], name='W_fc2')
b4 = bias_variable([10], name='b_fc2')
y = tf.nn.bias_add(tf.matmul(h_fc1_drop, w4), b4)


"""Training & Evaluating Model"""
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist_data.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels, keep_prob: 1.0}))