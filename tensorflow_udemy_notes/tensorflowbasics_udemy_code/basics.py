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
                       then(edges)

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