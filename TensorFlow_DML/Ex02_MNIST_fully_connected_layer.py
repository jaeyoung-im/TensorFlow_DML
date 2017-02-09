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

w1 = tf.Variable(tf.truncated_normal([784, 400], 0.0, 1.0, tf.float32))
b1 = tf.Variable(tf.truncated_normal([400], 0.0, 1.0, tf.float32))
# We create these Variables by giving tf.Variable the initial value of the Variable:
# In this case, we initialize both w and b as tensors full of zeros.
# Since we are going to learn w1 and b1, it doesn't matter very much what they initially are.
# Notice that w1 has a shape of [784, 400]
# because we want to multiply the 784-dimensional image vectors by it to produce 400-dimensional vectors of evidence
# for the hidden layer's nodes. b1 has a shape of [400] so we can add it to the output.

w2 = tf.Variable(tf.truncated_normal([400, 10], 0.0, 1.0, tf.float32))
b2 = tf.Variable(tf.truncated_normal([10], 0.0, 1.0, tf.float32))
# We create these Variables by giving tf.Variable the initial value of the Variable:
# In this case, we initialize both w and b as tensors full of zeros.
# Since we are going to learn w and b, it doesn't matter very much what they initially are.
# Notice that w has a shape of [400, 10]
# because we want to multiply the 400-dimensional hidden layer vectors by it to produce 10-dimensional vectors of evidence
# for the difference classes. b has a shape of [10] so we can add it to the output.

h = tf.nn.relu(tf.matmul(x, w1) + b1)
# First, we multiply x by w1 with the expression tf.matmul(x, w1).
# This is flipped from when we multiplied them in our equation,
# where we had w1x, as a small trick to deal with x being a 2D tensor with multiple inputs.
# We then add b1, and finally apply tf.nn.relu

y = tf.matmul(h, w2) + b2
# First, we multiply h by w2 with the expression tf.matmul(h, w2).
# This is flipped from when we multiplied them in our equation,
# where we had w2h, as a small trick to deal with h being a 2D tensor with multiple inputs.
# We then add b2


"""Training"""
y_ = tf.placeholder(tf.float32, [None, 10])
# A new placeholder to input the correct answers:

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
# Implement the cross-entropy function
# Firest, tf.log computes the logarithm of each element of y.
# Next, we multiply each element of y_ with the corresponding element of tf.log(y).
# Then tf.reduce_sum adds the elements in the second dimension of y, due to the reduction_indices=[1] parameter.
# Finally, tf.reduce_mean computes the mean over all the examples in the batch.

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
# We ask TensorFlow to minimize cross_entropy using the gradient descent algorithm with a learning rate of 0.5.
# Gradient descent is a simple procedure, where TensorFlow simply shifts each variable a little bit in the direction that reduces the cost

init = tf.global_variables_initializer()
# Initialize the variables we created.

sess = tf.Session()
sess.run(init)
# We can now launch the model in a Session, and now we run the operation that initializes the variables:

for i in range(10000):
    batch_xs, batch_ys = mnist_data.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# Run the training step 10000 times
# Each step of the loop, we get a "batch" of one hundred random data points from our training set.
# We run train_step feeding in the batches data to replace the placeholders.


"""Evaluating Model"""
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# First let's figure out where we predicted the correct label.
# tf.argmax is an extremely useful function which gives you the index of the highest entry in a tensor along some axis.
# For example, tf.argmax(y,1) is the label our model thinks is most likely for each input, while tf.argmax(y_,1) is the correct label.
# We can use tf.equal to check if our prediction matches the truth.

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# That gives us a list of booleans. To determine what fraction are correct, we cast to floating point numbers and then take the mean.
# For example, [True, False, True, True] would become [1,0,1,1] which would become 0.75.

print(sess.run(accuracy, feed_dict={x: mnist_data.test.images, y_: mnist_data.test.labels}))
# Finally, we ask for our accuracy on our test data.
# This should be about 95%