# coding: utf-8

# imports
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# download and store MNIST data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# create Tensorflow interactive session
sess = tf.InteractiveSession()

learning_rate = 1e-3
batch_size = 100
num_epochs = 15
num_batches = mnist.train.num_examples / batch_size

# create input, excpected output placeholders (can replace with variable number of input / output vectors)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# create weights, biases variables
W1 = tf.Variable(tf.random_normal([784, 512]))
b1 = tf.Variable(tf.random_normal([512]))

W2 = tf.Variable(tf.random_normal([512, 256]))
b2 = tf.Variable(tf.random_normal([256]))

W3 = tf.Variable(tf.random_normal([256, 10]))
b3 = tf.Variable(tf.random_normal([10]))

# define y in terms of placeholder x (input) and variables W, b (weights, biases)
h1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
h2 = tf.nn.relu(tf.add(tf.matmul(h1, W2), b2))
y = tf.add(tf.matmul(h2, W3), b3)

# define cross-entropy loss (mean of softmax cross entropy with logits / stable version of cross-entropy)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# define a single training step using the gradient descent optimization, minimizing the cross-entropy loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# initialize variables with parameters passed as arguments
sess.run(tf.global_variables_initializer())

print '\n'
for i in xrange(num_epochs):
    print '- Epoch :', i + 1
    average_cost = 0.0
    
    for j in xrange(num_batches):
        # get a minibatch of training data (100 samples)
        batch = mnist.train.next_batch(batch_size)
        # do training step on the sampled minibatch
        _, c = sess.run([optimizer, cross_entropy], feed_dict={x : batch[0], y_ : batch[1]})

        average_cost += c / num_batches

    print '\tcost :', '{:.9f}'.format(average_cost)

print '\n'

# define a boolean function which tells us if we've made correct predictions or not
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# create accuracy function (mean value of numeric-cast output of correct_prediction function)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# print the accuracy of the model on the test labels
print '\n', 'Single layer fully-connected network MNIST test data accuracy:', accuracy.eval(feed_dict={x : mnist.test.images, y_ : mnist.test.labels}), '\n'

#########################
# CONVOLUTIONAL NETWORK #
#########################

# define function to create weight, bias variables and do 2D convolution, 2x2 max-pooling

def weight_variable(shape):
    '''
    Creates a variable representing a network layer's weight parameters.
    
    input:
        shape: the dimensionality of the parameters of the network layer
        
    output:
        tensorflow.Variable vector where are entries are sampled from a truncated
        normal (> 0) distrubition with mean 0 and standard deviation 0.1
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial) 

def bias_variable(shape):
    '''
    Creates a variable representing the bias added to the outputs of each of the
    nodes in the network layer (changes the output from a linear transfrom to an
    affine transform).
    
    input:
        shape: the dimensionality of the output of the network layer
        
    output:
        tensorflow.Variable vector with all entries constant = 0.1
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    '''
    2D convolution operation. We use a (uniform) stride and padding of size 1.
    
    input:
        x: input to the convolutional layer
        W: weight parameters of the convolutional layer
        
    output:
        tensorflow.nn.conv2d of the input x, weights w, and stride and padding
        settings mentioned above.
    '''
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
def max_pool_2x2(x):
    '''
    2x3 max-pooling operation.
    
    input:
        x: input to the max-pooling layer
        
    output:
        tf.nn.max_pool of the input x... this downsamples the layers input by taking
        the max value of each 2x2 block, striding by 2x2 over the image
    '''
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    
# creating weights, biases for first convolutional layer (32 filters) 
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# reshaping the image input (28x28 instead of flattened 784
x_image = tf.reshape(x, [-1, 28, 28, 1])

# first hidden layer convolution and max-pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# creating weights, biases for second convolutional layer (64 filters)
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

# second hidden layer convolution and max-pooling
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# creating weights, biases for first fully-connected layer after conv / max-pool
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

# flattening the output from the second hidden layer max-pool
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# first fully-connected hidden layer after conv / max-pool
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# for dropout, creating a placeholder for the probability of keep a node in the network
keep_prob = tf.placeholder(tf.float32)

# creating the dropout function itself, to use on the first fully connected layer
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# variables for the second fully-connected layer (outputs to 10, the number of MNIST classes)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# the unnormalized probabilities output by the convolutional network
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# redefining the cross-entropy function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

# redefining the training step (use Adam optimization method with initial step size of 1*10^-4
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# redefining the correct prediction function
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))

# redefining the accuracy function
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all variables in session
sess.run(tf.global_variables_initializer())

print '...training the convolutional network', '\n'

# do the training! we use 2,000 iterations, print every 100 steps, and use dropout
# with probability 0.5 during training
for i in range(2000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print '\n'

# evaluate the accuracy of the trained convolutional network on the MNIST training data
print("MNIST test dataset accuracy: %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    
    
