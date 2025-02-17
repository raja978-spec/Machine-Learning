#                   WHEN TO USE TENSORFLOW
'''
When working with deep learning (e.g., image classification, NLP, reinforcement learning).
When you need GPU acceleration for training large models.
When you need flexibility in model design (custom layers, loss functions, etc.).

For simple problems like fruit classification, scikit-learn is enough. 
But if you were training a deep learning model on a large dataset 
(e.g., images of fruits instead of just numbers), PyTorch/TensorFlow would be 
the right choice.
'''


#                   SIMPLE TENSOR CODE
'''
import tensorflow as tf


with tf.compat.v1.Session() as sess:
    # Creates a placeholder for a float tensor.
    tensor = tf.compat.v1.placeholder(tf.float32, shape=(None,))
    # Creates a square of a tensor
    squared = tf.square(tensor)
    # feeds the tensor to the placeholder
    output = sess.run(squared, feed_dict={tensor: [1, 2, 3, 4, 5]})
    print(output)
'''

#                    SIMPLE KERAS ML MODEL
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential([
    Dense(units=64, activation='relu', input_shape=(784,)),
    Dense(units=10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''

#               TENSORS
'''
 It is a data structure that used to store various type of data, but in
 ML most of the type of tensors type would be int and float

 There can be 1D, 2D, 3D and ND tensors.

 Common functions in tensor:

 1. dtype
 2. shape
 3. size
 4. ndim
'''

#      TENSOR OPERATIONS
'''
import tensorflow as tf

a = tf.constant([1,2,3])
b= tf.constant([4,5,6])

print(tf.add(a,b))
# tf.Tensor([5 7 9], shape=(3,), dtype=int32)

print(tf.subtract(a,b))
# tf.Tensor([-3 -3 -3], shape=(3,), dtype=int32)

print(tf.multiply(a,b))

print(tf.divide(b,a))

#tf.Tensor([ 4 10 18], shape=(3,), dtype=int32)
#tf.Tensor([4.  2.5 2. ], shape=(3,), dtype=float64)
'''

#        TENSOR MATHEMATICAL OPERATIONS
''' 
  All these mathematcal operations are done in element wise means
  each and every element

  For sqrt operation tensor's data type should be float

import tensorflow as tf
a = tf.constant([[1,2],[3,4]], dtype=tf.float32)
b= tf.constant([[4,5],[6,7]])

print(tf.sqrt(a))
print(tf.square(b))
print(tf.math.log(a))

# tf.Tensor(
# [[1.        1.4142135]
#  [1.7320508 2.       ]], shape=(2, 2), dtype=float32)
# tf.Tensor(
# [[16 25]
#  [36 49]], shape=(2, 2), dtype=int32)
# tf.Tensor(
# [[0.        0.6931472]
#  [1.0986123 1.3862944]], shape=(2, 2), dtype=float32)

'''

#    REDUCTION OPERATIONS IN TENSOR
'''
 Helps to reduce the tensor values

import tensorflow as tf
a = tf.constant([[1,2],[3,4]], dtype=tf.float32)
b= tf.constant([[4,5],[6,7]])

print(tf.reduce_sum(a))
print(tf.reduce_mean(b))
print(tf.reduce_min(a))
print(tf.reduce_max(b))

# OUTPUT:
# tf.Tensor(10.0, shape=(), dtype=float32)
# tf.Tensor(5, shape=(), dtype=int32)
# tf.Tensor(1.0, shape=(), dtype=float32)
# tf.Tensor(7, shape=(), dtype=int32)
'''

#    TENSOR MATRIX OPERATIONS, INDXING, SLICING
'''

 What is inverse matrix ?
 
 The matrix inverse of a square matrix A is another matrix, denoted as A power of -1 ,
 such that when A is multiplied with A to the power of -1 it will give I(it is also
 matrix which has only 0 and 1 in it's column)

import tensorflow as tf
a = tf.constant([[1,2],[3,4]], dtype=tf.float32)
b= tf.constant([[4,5],[6,7]])
c= tf.constant([[4,5],[6,7]])

print(tf.matmul(c,b))
print(tf.transpose(a))
print(tf.linalg.inv(a))

#  INDXING AND SLICING
print(a[0])
print(a[:])


OUTPUT:

tf.Tensor(
[[46 55]
 [66 79]], shape=(2, 2), dtype=int32)
tf.Tensor(
[[1. 3.]
 [2. 4.]], shape=(2, 2), dtype=float32)
tf.Tensor(
[[-2.0000002   1.0000001 ]
 [ 1.5000001  -0.50000006]], shape=(2, 2), dtype=float32)
tf.Tensor([1. 2.], shape=(2,), dtype=float32)
tf.Tensor(
[[1. 2.]
 [3. 4.]], shape=(2, 2), dtype=float32)

'''

#            BROADCASTING
'''
 diamensons of two tensors is not matched to do arthimetic operation then
 it will automatically picks the values from the same dim

import tensorflow as tf
a = tf.constant([[1,2],[3,4]], dtype=tf.float32)
b= tf.constant([[4,5],[6,7]])
c= tf.constant([[4,5]])

print(b+c)

OUTPUT:
tf.Tensor(
[[ 8 10]
 [10 12]], shape=(2, 2), dtype=int32)
'''

#   CONSTANTS, VARIABLES, PLACEHOLDERS
'''
 All are helps to define and manipulate tensors in tensorflow

 1.Constant - not changed, created fixed tensor
 2.Variable - wegihts and baises are stored here that 
              can be modified.

import tensorflow as tf
a = tf.constant([[1,2],[3,4]], dtype=tf.float32)
b= tf.random.normal(shape=(1,3))
vt = tf.Variable(b)
print(vt.value())
c= tf.random.normal(shape=(1,3))
vt.assign(c)
print(vt.value())

OUTPUT:

tf.Tensor([[1.0626148 0.8104946 1.1755217]], shape=(1, 3), dtype=float32)
tf.Tensor([[ 1.9705979  0.5451773 -1.0278941]], shape=(1, 3), dtype=float32)

 
 3. Placeholders - helps to give values to computational graph on run time 
                   it is deprecated in tensorflow 2.x, because it is executed
                   eargly. The code will give error.

plc = tf.compat.v1.placeholder(shape=(1,3),dtype=tf.float16)
x= tf.constant([[1,2],[3,4]])
y = tf.square(x)
print(y)

 In tensorflow values can be directly feed

import tensorflow as tf

x= tf.constant([[1,2],[3,4]])
y = tf.square(x)
print(y)

OUTPUT:

tf.Tensor(
[[ 1  4]
 [ 9 16]], shape=(2, 2), dtype=int32)
'''

#      COMPUTATIONAL GRAPHS CREATING AND RUNNING TENSORFLOW SESSION
'''
 It is the back bone of tensor model, TensorFlow can optimize the 
 computational graph for performance, memory usage, and 
 distributed execution.

 It helps to view the flow of data, provided good 

 1. What is CG ?

 it's a computational graph is a directed graph that represents a 
 series of mathematical operations, which are called nodes, and the 
 flow of data, which are called tensors between each other.

 Node is also a graph representation operations that performed on 
 tensors 

 Edges in the graph represent tensors flowing between operations, carrying the 
 output of one operation to the input of another.

 2. How to build computation graph

 define operations like matmul, loss function

x= tf.constant([[1,2],[3,4]])
y=tf.constant([5,6],[7,8])
c= tf.matmul(x,y)

  connect it for graph visulaization

print(tf.compat.v1.get_default_graph().as_graph_def())

  View thw create computational graph using tensorboard, filewriter
  writes the graph on board

with tf.compat.v1.Session() as sess:
    writer = tf.compat.v1.summary.FileWriter(logdir='logdir', graph=tf.compat.v1.get_default_graph())
    writer.close()

with tf.compat.v1.Session() as sess:
    result = sess.run(c)
    print(result)


            CREATING AN RUNNING TENSORFLOW SESSION IN tensorflow 1

     
    Creating and running TensorFlow sessions is essential for executing 
    operations and computing the output

FULL CODE OF ABOVE: 

import tensorflow as tf
# Disable eager execution to use graph-based execution (TensorFlow 1 style)
tf.compat.v1.disable_eager_execution()

x= tf.constant([[1,2],[3,4]])
y=tf.constant([[5,6],[7,8]])
    
c= tf.matmul(x,y)

with tf.compat.v1.Session() as sess:
    writer = tf.compat.v1.summary.FileWriter(logdir='logdir', graph=tf.compat.v1.get_default_graph())
    writer.close()
with tf.compat.v1.Session() as sess:
    result = sess.run(c)
    print(result)

    
     CREATING AN RUNNING TENSORFLOW SESSION IN tensorflow 2.x
LATEST CODE:

import tensorflow as tf

x= tf.constant([[1,2],[3,4]])
y=tf.constant([[5,6],[7,8]])
    
c= tf.matmul(x,y)
writer = tf.summary.create_file_writer(logdir='logs')
   
with writer.as_default():
    tf.summary.write("Mat mul", tf.constant('G logged'), step=0)

print(c.numpy())

OUTPUT:
[[19 22]
 [43 50]]
'''


#                MANAGING GRAPHS AND SESSIONS
'''
 It is much needed to know about this to built the model in more efficient
 way

 1. TensorFlow Graph - A TensorFlow graph or a computational graph is a data 
                       structure that represents a series of TensorFlow
                       operations(nodes) and flows of the data between
                       the(edges)

                       the default graph TensorFlow automatically creates a default
                       graph when you define operations.

EX:

import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    a = tf.constant(2)
    b= tf.constant(3)
    c= tf.add(a,b)

 2. TensorFlow Session - TensorFlow session provides an execution 
                         environment for TensorFlow operations within a 
                         specific graph.

 3. TensorFlow Options - Various options such as specifying the device to execute
                        operations GPU memory growth, etc

EX: in this example we passed graph which we want to execute our operations
    if the graph is not specified the operations are automatically moved
    to default graph

import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    a = tf.constant(2)
    b= tf.constant(3)
    c= tf.add(a,b)

with tf.compat.v1.Session(graph=graph) as sess:
    print(sess.run(c))


    WITHOUT SESSION OPTION

import tensorflow as tf
with tf.compat.v1.Session() as sess:
    a = tf.constant(2)
    b= tf.constant(3)
    c= tf.add(a,b)
    print(sess.run(c))

OUTPUT:
5

  3. Closing Session - We need to close session manually to release the
                       resources

                       If it is not closed then below problem occurs

                       Resource Leakage:

                       Problem: A session allocates resources such as 
                       memory (for variables, tensors, and computation graphs) 
                       and sometimes GPU memory. Not closing the session can 
                       lead to resource leakage, where these resources remain 
                       allocated until the Python process terminates.

                       Impact: This can cause memory exhaustion, especially in 
                       systems with limited resources or when running multiple 
                       sessions in the same program.

EX:

import tensorflow as tf
with tf.compat.v1.Session() as sess:
    a = tf.constant(2)
    b= tf.constant(3)
    c= tf.add(a,b)
    print(sess.run(c))
    sess.close() # closes session for this block

print('Hello') # session will be automatically closed if execution process
               # comes to this even if the close() is not defined


               MORE CONFIG FOR SESSION

import tensorflow as tf
config = tf.compat.v1.ConfigProto() # Used to create more config
config.log_device_placement = True
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    a = tf.constant(2)
    b= tf.constant(3)
    c= tf.add(a,b)
    print(sess.run(c))
            

'''

#       BASIC NEURAL NETWORK WITH TENSORFLOW
'''
  Example of feed forward neural network 

  Note: Below code will not run

import tensorflow as tf

# Define architecture like input features, no of hidden layers,
# the no of neurons in each hidden layer, number of output classes for classification tasks.

input_size = 784 
hidden_size = 128 # no of neuron in hidden layer
output_size = 10 # no of output classes

# Specify the input and output layer
X = tf.compat.v1.placeholder(tf.float32, shape=[None, input_size])
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, output_size])


# DEPRECATED EXAMPLE USED IN TENSOR 1

# Config hidden layer
hidden_layer = tf.compat.v1.layers.dense(inputs=X, units=hidden_size, activation=tf.nn.relu)
output_layer = tf.compat.v1.layers.dense(inputs=hidden_layer, units=output_size, activation=None)

# Define loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))

# Define optimization algorithm to update weights and baises
# values which are passed to loss function to minimize it
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)

# Initialize variables

init = tf.compat.v1.global_variables_initializer();

# Starting Session
with tf.compat.v1.Session() as sess:
    sess.run(init)
    
    #Train model (example feed forward neural network)
    for epoch in range(num_epochs):
        # Perform feed forward and backward pass
        _,curr_loss = sess.run([train_op, loss], feed_dict={X: input_data, Y: target_labels})

        # print training loss
        print(f'Epoch {epoch+1}, loos: {curr_loss}')

    # Example evaluation
    accuracy = sess.run(accuracy_op, feed_dict={X: test_data, Y: test_labels})
    print('Test accuray', accuracy)


        WORKING CODE

import tensorflow as tf

# Define architecture parameters
input_size = 784  # Number of input features
hidden_size = 128  # Number of neurons in the hidden layer
output_size = 10  # Number of output classes

# Input and labels (placeholders)
X = tf.keras.Input(shape=(input_size,), name="X")
Y = tf.keras.Input(shape=(output_size,), name="Y")

# Define the model architecture
hidden_layer = tf.keras.layers.Dense(units=hidden_size, activation=tf.nn.relu, name="hidden_layer")(X)
output_layer = tf.keras.layers.Dense(units=output_size, activation=None, name="output_layer")(hidden_layer)

# Define the model
model = tf.keras.Model(inputs=X, outputs=output_layer)

# Compile the model with a loss function, optimizer, and metrics
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Placeholder for dummy data (replace with actual dataset)
import numpy as np
num_samples = 100
input_data = np.random.rand(num_samples, input_size)  # Replace with real input data
target_labels = tf.keras.utils.to_categorical(np.random.randint(0, output_size, num_samples), num_classes=output_size)

# Training the model
num_epochs = 10
batch_size = 32
model.fit(input_data, target_labels, epochs=num_epochs, batch_size=batch_size)

# Placeholder for test data (replace with actual test dataset)
test_data = np.random.rand(num_samples, input_size)  # Replace with real test data
test_labels = tf.keras.utils.to_categorical(np.random.randint(0, output_size, num_samples), num_classes=output_size)

# Evaluating the model
test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


'''

#     ACTIVATION FUNCTIONS IN TENSORFLOW
'''
 Allowing neural networks to learn complex patterns and 
 relationships in data.

 They are applied to the output of each neuron in a neural network, 
 transforming the input signal into, the output signal that is passed 
 to the next layer.
 
 1. Linear Activation Function - Used for regression tasks, where the
                                 output number will be real numbers.

                                 But it is rarely use due to its
                                 limited representation.

                                 In mathematical terms, f(x) = x

EX:

import tensorflow as tf
input=32
output=tf.keras.activations.linear(input)
print(output)

OUTPUT: 32

 2. Sigmoid - Squeezes the input values between 0 and 1, used for binary
              classification where the output represents possibilities.

EX:

import tensorflow as tf
input=32.00
output=tf.keras.activations.sigmoid(input)
print(output)

OUTPUT:
tf.Tensor(1.0, shape=(), dtype=float32)

3. Hyperbolic Tangent - Squeezes the input values between -1 and 1, used
                         in hidden layer


EX:

import tensorflow as tf
input=32.00
output=tf.keras.activations.tanh(input)
print(output)

OUTPUT:
tf.Tensor(1.0, shape=(), dtype=float32)


4. Rectified Linear Unit - The rectified linear unit activation 
                           function introduces non-linearity by 
                           outputting the input value if it is positive

                           If it is negative it returns 0 

                           Widely used in hidden layers due to it's simplicity

EX:

import tensorflow as tf
input=-32.00
output=tf.keras.activations.relu(input)
print(output)

OUTPUT:
tf.Tensor(0.0, shape=(), dtype=float32)


5. Leaky Rectified Linear Unit - The leaky ReLU activation function 
                                 addresses the dying ReLU problem by 
                                 allowing a small positive gradient
                                 when the input is negative.It 
                                 helps prevent the issues of neurons 
                                 becoming inactive during training.

                                 The "dying ReLU" problem occurs when 
                                 the Rectified Linear Unit (ReLU) 
                                 activation outputs zero for all 
                                 negative inputs, causing some neurons 
                                 to stop learning as their gradients 
                                 become zero. For example, if a neuron 
                                 consistently has negative inputs, ReLU 
                                 will always output 0, and its weights 
                                 won't update during training

EX:

import tensorflow as tf
input=-32.00 # Here it returns always zero as it is negative
output=tf.keras.activations.relu(input)
print(output)

OUTPUT:
tf.Tensor(0.0, shape=(), dtype=float32)


EX: To avoid the above problem leaky multiplies small gardient value
    which is 0.1 to all the tensor values in tensor

import tensorflow as tf
input=tf.constant([-32.00, 32.000])
output=tf.keras.layers.LeakyReLU(alpha=0.2)(input)
print(output)

OUTPUT:
tf.Tensor([-6.4 32. ], shape=(2,), dtype=float32)
 
 6. Softmax - Used to multiclassification, converts input values into
              probability distribution over multiple classes ensuring the
              output sum to one

EX:
import tensorflow as tf
input=tf.constant([24.00, 32.000])
output=tf.keras.activations.softmax(input)
print(output)

OUTPUT:
tf.Tensor([3.3535017e-04 9.9966466e-01], shape=(2,), dtype=float32)
'''

#   LOSS FUNCTIONS AND OPTIMIZERS
'''

 Used to measuring the model's performance and updating its 
 parameters to minimize the loss.

 Loss functions finds the difference between the model's predicted output
 (labels) and the actual output (labels).

 1. Mean Squared Error - Used to find the average value of predicted and
                         actual labels.


EX:

import tensorflow as tf
print(tf.keras.losses.MeanSquaredError())

DEPRECATED:

import tensorflow as tf
print(tf.keras.losses.mean_squared_error())


2. Binary Cross-Entropy - helps to measure the binary classification's predicted and 
                          and actual labels.


EX: tf.keras.losses.binary_crossentropy(true_labels, predicted_labels)
 
 3. Categorical Cross-Entropy - helps to measure the multiclass classification's predicted and 
                          and actual labels.


EX: tf.keras.losses.categorical_crossentropy(true_labels, predicted_labels)

  4. Sparse Categorical Cross-Entropy - helps to measure the multiclass 
                                       classification's predicted 
                                       and actual labels. But accepts 
                                       only integer values as its labels.


EX: tf.keras.losses.sparse_categorical_crossentropy(true_labels, predicted_labels)


                            OPTIMIZERS

Optimizers are algorithms used to minimize the loss function by 
updating the models parameters, weights and biases during training.

1. Stochastic Gradient Descent (SGD) - The most basic optimizer, 
                                      updates the weights and biases 
                                      using the gradient of the loss 
                                      function.

EX:

import tensorflow as tf
print(tf.keras.optimizers.SGD())

2. Adam - A variant of SGD that uses adaptive learning rates for 
          different parameters. Combines the advantages of both Adagrad and 
          Rmsprop algorithm

EX:

import tensorflow as tf
print(tf.keras.optimizers.Adam())

3. RMPSProp - Root mean square propagation

              This adjusts the learning rate for each parameter 
              based on the magnitude of recent gradients.
              
              It prevents the learning rate from decreasing too 
              fast for frequently occurring parameters.

              The magnitude is a measure of the "size" of the gradient.

EX:
import tensorflow as tf
print(tf.keras.optimizers.RMSprop(learning_rate=0.001))


4. Adagrad - Adaptive gradient algorithm
            
             adapts the learning rate for each parameter based on the historical 
             gradients of that parameter.

             It allocates more learning to less frequently occurring parameters

EX:

import tensorflow as tf
print(tf.keras.optimizers.Adagrad(learning_rate=0.01))


                CUSTOM LOSS FUNCTION ANS OPTIMIZERS

import tensorflow as tf
class CustomLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(CustomLoss, self).__init__()
    
    def call(self, y_true, y_pred):
        return tf.math.square(y_true - y_pred)

custom_loss = CustomLoss()
custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

class CustomOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01):
        super(CustomOptimizer, self).__init__(learning_rate)

    def apply_gradients(self, grads_and_vars):
        for grad, var in grads_and_vars:
            var.assign_sub(self.learning_rate * grad)
        return []

custom_optimizer = CustomOptimizer(learning_rate=0.01)
print(custom_optimizer)
'''

#  BUILD MODEL USING KERAS API
'''
 * Working on the top of tensorflow
 * Has modular approach to develop the model 
   in more efficient way.

 Components of keras API:

    1. Layers - used of building neural networks, including dense, 
                fully connected, convolutional, recurrent, 
                dropout etc.

    2. Activations - Keras supports various activation functions
                     such as sigmoid, tanh, ReLU, softmax, etc.
                    which introduce non-linearity into the network

    3. Loss Functions 

    4. Optimizers

    5. Metrics - Keras provides built in metrics to evaluate model 
                 performance during training, such as accuracy, 
                 precision.

    6. Callbacks - Keras provides callbacks to monitor and 
                   control training process, such as early stopping,
                   model saving, learning rate scheduling, etc.


EXAMPLE NEURAL NETWORK IN KERAS

import tensorflow as tf
print(tf.__version__)

# Load sample dataset
# Mnist is the collection of handwritten digits
# All the digits are represents images

mins = tf.keras.datasets.mnist

# x and y train will have the handwritten digits(images) that are 
# with their respective lables, x and y test have the same
# values which used to evaluate model's prediction

(x_train,y_train), (x_test,y_test) = mins.load_data()

# Normalizes the pixels values in train and test 
# By divding the each pixel values in train and test
# the value will be in between 0 and 1, 0 represents black
# and 1 represents white.

x_train, x_test = x_train / 255.0, x_test / 255.0

# Sequential model is a linear stack of layers where you can simply 
# add one layer at a time.

# The firt layer of the model is flatten which helps to change
# the 2D array (which has the shape of 28*28) to 1D array 
# like 28 * 28 = 784, because Dense layer performs operations on 
# a flat vector of input values. 
# It requires a 1D array for each input instance. 
# This is because each neuron in the Dense layer is connected 
# to every input value, and these connections are represented 
# as a vector of weights.

# Second layer is the dense layer where each neuron is connected
# other layers, so it is called fully connected layer

# Thirs is dropout used to avoid overfitting
# 20% of the neurons will be randomly "dropped"
# The dropout rate 0.2 means that 20% of the neurons 
# in the layer where the Dropout is applied will be 
# deactivated (ignored) in each forward pass during training.

# The fourth layer is the output layer which has no activations
# it will give 10 neurons as output each represents the probabilty
# of the input belonging to a specific class

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# We want output of the model should ne probability that is 0 and 1 
# for that from_logits=True is used here, it applies softmax function
# internally to get the probabilities in 0's and 1's if it is false
# the output will be in positive and negative integers.

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# before trainig the model we have to compile the model
# using compile, if we are not compile it then the TensorFlow 
# won't know what loss function to 
# minimize or which optimizer to use for updating the weights.
# hence the fit() method for training the model will raises error.


# In this model the loss_fn will be reduced using adam optim
# The 'accuracy' metric computes the percentage of correct predictions 
# out of all predictions.
# During Training After each batch or epoch, TensorFlow 
# calculates the specified metric(s) based on the predictions 
# and true labels.
# It provides feedback (e.g., accuracy) so you can monitor 
# how well the model is learning.
# Metrics are just for monitoring and are not used to compute 
# gradients or update the model's weights. Only the loss function 
# is used for optimization.

model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

# Different Types of Metrics:

# The metrics you specify depend on the type of problem you're solving:

# For Classification Problems:

# 'accuracy': Compares predicted labels to true labels.
# 'sparse_categorical_accuracy': Similar to 'accuracy' but used when true labels are integers (e.g., [0, 1, 2]).
# 'categorical_accuracy': Used when true labels are one-hot encoded (e.g., [[1,0,0], [0,1,0]]).
# For Regression Problems:

# 'mean_absolute_error' (MAE): Average of the absolute differences between predictions and true values.
# 'mean_squared_error' (MSE): Average of the squared differences between predictions and true values.

# Custom Metrics:

# You can define your own metrics by writing a custom function. For example:
# def custom_metric(y_true, y_pred):
#     return tf.reduce_mean(tf.abs(y_true - y_pred))  # Example: MAE
# 

# model.compile(optimizer='adam', loss='mse', metrics=[custom_metric])

# Why Use Metrics?
# Training Monitoring: Metrics give you insights into how the model is performing during training (e.g., accuracy or loss trends).
# Validation Performance: They help you compare the performance on the training and validation datasets to detect overfitting or underfitting.
# Evaluation: After training, metrics show how well the model performs on unseen test data.
# Example:
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy', 'sparse_categorical_accuracy']
# )
# accuracy: General accuracy metric.
# sparse_categorical_accuracy: Equivalent to accuracy when using integer labels (for datasets like MNIST).

# x_train is the input data which typically consists of 
# a set of features or images.
# y_train These are the corresponding labels for the training data.
# The model learns to associate the input data
# x_train with these labels (y_train) during training.
# An epoch is one complete pass through the entire training data set.
# The epochs parameter specifies how many times the model will 
# iterate over the entire data set during the training.

model.fit(x_train, y_train, epochs=5)

# Now the model is trained, we have to evaluate the
# model by parsing the test data set to know
# how it predicts the data

model.evaluate(x_test,y_test, verbose=2)
'''

#       UNDERSTAND WHAT INPUT AND TARGET DATA MEANS
'''
 fit(input_data, target_labels) and evaluate(input_data, target_labels) gets this 
 parameters, lets understand what they mean

 fit learns features from input and learns what to predict with that 
 feature with target

 EXAMPLE:

 For a simple regression model where you're predicting house prices:

Input Data (Features): Information about houses (e.g., size, number of rooms).
Target Data: Actual prices of those houses.
When you run model.fit(), the model tries to learn the 
relationship between the features (input data) and the target 
(output label). It adjusts itself to minimize the prediction error.

Process:

The model sees an input (e.g., house size).
It makes a prediction (e.g., predicted price).
It compares that prediction with the actual price (target).
The model adjusts its parameters to reduce the error, 
and this process is repeated until the model is good at predicting.

SUMMARY:

Input Data: Features the model uses to make predictions.
Target Data: Correct outputs the model aims to predict.
fit(): Trains the model by learning the relationship between input and target data.


 EVALUATE:

 The evaluate() method is used to assess how well 
 the trained model performs on unseen data, typically 
 the test set (data that the model hasn't been trained on).

 So the above sample code When you pass x_test to the model:

 * The model applies the learned patterns (from training) to 
   the new data in x_test.

 * It uses the learned weights and biases (which were adjusted 
   during training) to generate predictions.

 * During evaluation (via evaluate()), the model is not 
   learning from the test data; it is simply making predictions 
   and comparing them to the actual values (y_test)

 * The model compares its predictions with these actual house prices which is
   on y_test to calculate the loss or error.

 * It calculates how far off its predictions are from the 
   actual values (using a loss function like Mean Squared Error or Mean Absolute Error)
   
 * It calculates how far off its predictions are from 
   the actual values (using a loss function like Mean Squared Error or Mean Absolute Error)
   A lower loss means the model is more accurate.
'''

#          BUILDING CONVOLUTION NEURAL NETWORK IN TENSORFLOW KERAS

'''

These are considered as complex model, because it will have multiple
input and output layers. 

In the below code have two types of input layers, that are
Conv2D, Dense

We can create custom layers.

We have to do model sub classing 


OUTPUT:

Epoch 1/5
Epoch 3/5
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 14s 9ms/step - accuracy: 0.9918 - loss: 0.0271 - val_accuracy: 0.9875 - val_loss: 0.0405
Epoch 4/5
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 14s 9ms/step - accuracy: 0.9938 - loss: 0.0191 - val_accuracy: 0.9864 - val_loss: 0.0567
Epoch 5/5
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 14s 9ms/step - accuracy: 0.9960 - loss: 0.0119 - val_accuracy: 0.9879 - val_loss: 0.0444
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9864 - loss: 0.0482 (evalivation phase)
* 0.03873628377914429
* 0.9891999959945679

'''


#           STATERGIES IN CNN TRANSFORM LEARNING
'''
There are 2 statergies in CNN

1. feature extraction - here the pre-trained model will be freezed
                        and new small part of dense layer added to
                        train the model for new task.

2. Fine tuning -  pre-trained CNN model is used as an initialization 
                  for training on the new dataset. So pre-trained
                  model is not freezed here. This pre trained
                  model will adapts with new features in new dataset.

CHOSSING THE RIGHT PRE TRAINED MODEL:

It depends on the size and nature of target label

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to a range of [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Split validation data from the training data
val_images = train_images[:10000]
val_labels = train_labels[:10000]
train_images = train_images[10000:]
train_labels = train_labels[10000:]

# One-hot encode the labels
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

# Load pre-trained model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Feature extraction
base_model.trainable = False # Freeze the pre-trained model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(10, activation = 'softmax')
])

base_model.trainable = True
fine_tune_at = 120 # specifies lower training rate on pre-trained model
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Define the model
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optiSmizer='adam', loss='spares_categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
print(model.summary)

history = model.fit(train_images,train_labels, epochs=5, batch_size=32, validation_data=(val_images, val_labels))

test_loss , test_accuracy = model.evaluate(test_images,test_labels)
print('*',test_loss)
print('*',test_accuracy)


'''

#                  WHAT IS RNN
'''
Works well on sequential based data like text recognization, It has
hidden state which stores the previous input until the current input
is arrived.

The gradient which are going to find on this model will be too small
so the training non linearity will raise here, to avoid such sutivation
LSTM (long short time memory) and GRU (Gated Recurrent Unit) are used here.

LSTM It uses a memory cell with gates (input, forget, and output gates) 
to control the flow of information.


GRU has two gates (reset and update gates) instead of three.

When to Use:

Use LSTM when you need to remember long sequences 
(e.g., speech recognition, time-series forecasting).

Time Series prediction like weather prediction.

Use GRU when you need a faster and less complex model.

Refer rnn.docx for more
'''

#             BUILDING RZZ MODEL IN TENSORFLOW

'''
 SimpleRNN(64, input_shape=(sequence_length,input_dim))

 this is the syntax for LSTM, GRU

 EXAMPLE FOR BOTH SIMPLERNN AND LSTM

 from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to a range of [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Split validation data from the training data
val_images = train_images[:10000]
val_labels = train_labels[:10000]
train_images = train_images[10000:]
train_labels = train_labels[10000:]

# One-hot encode the labels
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

# Define the model
model = Sequential([
    LSTM(64, input_shape=(28, 28)),  # Fixed input shape
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Fixed loss function

# Display the model summary
model.summary()  # Fixed function call

# Train the model
history = model.fit(train_images, train_labels, epochs=2, batch_size=32, validation_data=(val_images, val_labels))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print('* Test Loss:', test_loss)
print('* Test Accuracy:', test_accuracy)

'''


#              DEPLOYING TENSORFLOW MODELS(Save Load)
'''
 While saving a trained model the parameters used for weights, optimizer
 and its architecture  are
 stored, multiples ways are available for saving the model.

 model.save('filename.h5) # stores all info

 For saving model and architecture on different file we can use json
 format

 with open('model.json', 'w') as file:
    file.write(model.to_json())

model.save_weights('model.weights.h5')


EX:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to a range of [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Split validation data from the training data
val_images = train_images[:10000]
val_labels = train_labels[:10000]
train_images = train_images[10000:]
train_labels = train_labels[10000:]

# One-hot encode the labels
train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

model = Sequential([
    LSTM(64, input_shape=(28, 28)),  # Fixed input shape
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Fixed loss function
history = model.fit(train_images, train_labels, epochs=2, batch_size=32, validation_data=(val_images, val_labels))

model.save('model.h5')

with open('model.json', 'w') as file:
    file.write(model.to_json())

model.save_weights('model.weights.h5')




                    LOAD SAVED MODEL

from tensorflow.keras.models import load_model, model_from_json

# Load model from HDF5 file (architecture + weights)
loaded_model = load_model('model.h5')

# Load architecture from JSON file
with open('model.json', 'r') as file:
    loaded_model_json = file.read()

lm = model_from_json(loaded_model_json)

# # Load weights separately
# lm.load_weights('model_weights.h5')  # Ensure this file exists

print(lm.summary())  # Print model summary


OUTPUT:

Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓    
┃ Layer (type)                     ┃ Output Shape            ┃       Param # ┃    
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩    
│ lstm (LSTM)                      │ (None, 64)              │        23,808 │    
├──────────────────────────────────┼─────────────────────────┼───────────────┤    
│ dense (Dense)                    │ (None, 10)              │           650 │    
└──────────────────────────────────┴─────────────────────────┴───────────────┘    
 Total params: 48,918 (191.09 KB)
 Trainable params: 24,458 (95.54 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 24,460 (95.55 KB)

NOTE: Model saved in one version will not be worked on other
      newer version of model. Based on deploy environment we have
      save the model in correct format, like hdf5, json


                WHAT IS h5 AND WHY MODEL ARE SAVED IN THA FORMAT

The .h5 (HDF5) format is used in TensorFlow/Keras for saving deep learning models because it is efficient and structured.

What is HDF5 (.h5)?
HDF5 (Hierarchical Data Format version 5) is a binary data format designed to store large amounts of data efficiently. It supports:
✅ Structured data storage (like a file system inside a file)
✅ Compression (reduces file size without losing information)
✅ Fast read/write access
✅ Support for large datasets

Why is .h5 used for saving models?
Stores everything in a single file:

Model architecture
Model weights
Training configuration (loss, optimizer, metrics)
Optimizer state (for resuming training)

'''
