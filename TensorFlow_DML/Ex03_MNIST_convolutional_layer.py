from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

"""Read MNIST Dataset"""
# Download and read in the data automatically.
#
# MNIST data is split into three parts:
# 1. Training data (mnist_data.train) : 55,000 data
# 2. Validation data (mnist_data.validation) : 5,000 data
# 3. Test data (mnist_data.test) : 10,000 data
#
# Each part has two parts: an image and a corresponding label
# ex) mnist_data.train.images (image), mnist_data.train.labels (label)
#
# Each image is 28 pixels by 28 pixels, and it is plattend to a vector of 784 by 1
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot = True)


"""Building Architecture"""
x = tf.placeholder(tf.float32, [None, 784])
# x isn't a specific value. It's a placeholder, a value that we'll input when we ask TensorFlow to run a computation.
# We want to be able to input any number of MNIST images, each flattened into a 784-dimensional vector.
# We represent this as a 2-D tensor of floating-point numbers, with a shape [None, 784].
# None means that a dimension can be of any length.

x_image = tf.reshape(x, [-1, 28, 28, 1])
# We want to be able to input any number of MNIST imges, each reshaped into a 28x28x1-dimensional tensor
# x_image isn a reshaped tensor of x.
# We represent this as a 4-D tensor, with a shape [-1, 28, 28, 1].
# None(= -1) means that a dimension can be of any length.
# 1 means that MNIST images have a color space as a gray scale

w1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], 0.0, 1.0, tf.float32), name='W_conv1')
b1 = tf.Variable(tf.truncated_normal([32], 0.0, 1.0, tf.float32), name='b_conv1')
# We create these Variables by giving tf.Variable the initial value of the Variable:
# In this case, we initialize both w1 and b1 as tensors full of truncated normal distribution values.
# Since we are going to learn w1 and b1, it doesn't matter very much what they initially are.
# Because, the convolution will compute 32 features for each 5x5 patch, w1 has a shape of [5, 5, 1, 32]
# The first two dimensions are the patch size, the next is the number of input channels,
# and the last is the number of output channels.
# b1 has a shape of [32] so we can add it to the each output channel.

h_conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_image, w1, strides=[1, 1, 1, 1], padding='SAME'), b1))
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
# The max_pool method will reduce the image size to 14x14.

w2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], 0.0, 1.0, tf.float32), name='W_conv2')
b2 = tf.Variable(tf.truncated_normal([64], 0.0, 1.0, tf.float32), name='b_conv2')
# We create these Variables by giving tf.Variable the initial value of the Variable:
# In this case, we initialize both w2 and b2 as tensors full of truncated normal distribution values.
# Since we are going to learn w2 and b2, it doesn't matter very much what they initially are.
# Because, the convolution will compute 64 features for each 5x5 patch, w2 has a shape of [5, 5, 32, 64]
# The first two dimensions are the patch size, the next is the number of input channels,
# and the last is the number of output channels.
# b2 has a shape of [64] so we can add it to the each output channel.

h_conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(h_pool1, w2, strides=[1, 1, 1, 1], padding='SAME'), b2))
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# We then convolve h_pool1 with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
# The max_pool method will reduce the image size to 7x7.

w3 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], 0.0, 1.0, tf.float32), name='W_fc1')
b3 = tf.Variable(tf.truncated_normal([1024], 0.0, 1.0, tf.float32), name='b_fc1')
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_pool2_flat, w3), b3))
# Now that the image size has been reduced to 7x7,
# we add a fully-connected layer with 1024 neurons to allow processing on the entire image.
# We reshape the tensor from the pooling layer into a batch of vectors,
# multiply by a weight matrix, add a bias, and apply a ReLU.

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# To reduce overfitting, we will apply dropout before the output layer.
# We create a placeholder for the probability that a neuron's output is kept during dropout.
# This allows us to turn dropout on during training, and turn it off during testing.
# TensorFlow's tf.nn.dropout op automatically handles scaling neuron outputs in addition to masking them,
# so dropout just works without any additional scaling.

w4 = tf.Variable(tf.truncated_normal([1024, 10], 0.0, 1.0, tf.float32), name='W_fc2')
b4 = tf.Variable(tf.truncated_normal([10], 0.0, 1.0, tf.float32), name='b_fc2')
y = tf.nn.bias_add(tf.matmul(h_fc1_drop, w4), b4)
# Finally, we add a layer, just like for the one layer softmax regression above.


"""Training & Evaluating Model"""
y_ = tf.placeholder(tf.float32, [None, 10])
# A new placeholder to input the correct answers:

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# tf.nn.softmax_cross_entropy_with_logits internally applies the softmax
# on the model's unnormalized model prediction and sums across all classes,
# and tf.reduce_mean takes the average over these sums.

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# We ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.0001.

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# First let's figure out where we predicted the correct label.
# tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis.
# For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label.
# We can use tf.equal to check if our prediction matches the truth.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean.
# For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist_data.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# Run the training step 20000 times
# Each step of the loop, we get a "batch" of one 50 random data points from our training set.
# We run train_step feeding in the batches data to replace the placeholders.

print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels, keep_prob: 1.0}))
# Finally, we ask for our accuracy on our test data.
# This should be approximately 99.2%