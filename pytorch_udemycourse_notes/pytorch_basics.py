#                   WHEN TO USE PYTORCH
'''
When working with deep learning (e.g., image classification, NLP, reinforcement learning).
When you need GPU acceleration for training large models.
When you need flexibility in model design (custom layers, loss functions, etc.).

For simple problems like fruit classification, scikit-learn is enough. 
But if you were training a deep learning model on a large dataset 
(e.g., images of fruits instead of just numbers), PyTorch/TensorFlow would be 
the right choice.
'''

#            CUDA
'''
 
Compare to normal cpu device cuda will execute the large tensors
operations in more faster way

import torch

a = torch.rand(3,3)
b= torch.rand(3,3)
if torch.cuda.is_available():
    a.to('cuda')
    b.to('cuda')    
    print(torch.matmul(a,b))
else:
    print('cpu', torch.matmul(a,b))

'''


#               TENSOR
'''
 Refer tensor.docx for theory part

import torch
import numpy as np

a = torch.tensor([1,2,3])

#                       Numpy tensor
b = np.array([1,2,3])
c = torch.tensor(b)
#print(a, c)

#                        Tensor with 0,1
a = torch.zeros(3,3)
b = torch.ones(3,3)
#print(a,b)

#                    Random tensor
# Used to initilize weights for the model
a = torch.rand(2,2)
#print(a) 


#                    DTYPE DEVICES
#a = torch.tensor([1,2,3], dtype=torch.float32, device='cuda')
#print(a)


#               ELEMENT WISE ADDITION
a = torch.tensor([1,2,3])
b = torch.tensor([1,2,3])
s = a+b
#print(s)

#                    MATMUL
#print(torch.matmul(a,b))


#               IN PLACE OPERATIONS
a = torch.tensor([1,2,3]) 
print(a.add_(5)) # adds 5 to all the tensor element without creating
                 # new memory.
'''

#        AUTOGRAD AND DYNAMIC COMPUTAION GRAPH
'''

 * Auto grad is a process of monitoring the operations which are
   performed on tensors to create dynamic computational graph.

 * DCG is the one of the feature that make pytorch more stronger.

 * Gradient are computed in pytorch that is used to build the
   DCG.

 * Gradient provide the necessary information to update the model
   weight's parameter. 

HOW AUTOGRAD WORKS

When we perform computations on tensors pytorch will automatically  generates
DCG, which shows the dependency between one tensor to another tensor
when we perform computations on tensor.

By default the gradient are not calculated in pytorch tensor, we
have to set requires_grad = True.

EX:

import torch

x = torch.tensor([1,2,3], requires_grad=True, dtype=torch.float16)

x.sum().backward() # computes gradients, computes the changes of each
                   # tensor element while performing sum()
            
print(x.grad) # this is the attribute which stores the

# value of x.grad is used to update the model's parameter with the
# algorithm called gradient descent.

learning_rate = 0.1
with torch.no_grad():    # pytorch automatically monitors the 
                         # operations performed tensor when we set
                         # requires_grad = True, this time 
                         # are updating the model's parameter for
                         # we don need to monitor this operations
                         # because the DCG grows unnecessarily.
    x -= learning_rate



                              DETACH

Is the method helps to create new tensor, which is detach from monitoring
it's operations.

x.detach() only creates a new detached tensor but does not reset x.grad.

To clear x.grad, you should explicitly set x.grad = None 
before calling .backward() again, to avoid accumulating gradients.

import torch

x = torch.tensor([1,2,3], requires_grad=True, dtype=torch.float32)
x.sum().backward()

a=x.detach() # returns detached tensor 
#a.sum().backward() # this will raise error, the
                   # backward() should be used when 
                   # we find the gradient(requires_grad=True).
                   # but here detach() will sets the requires_grad
                   # as False, so we can't use backward() to find
                   # gradient.

print(a.grad, 2**3)

 
                WHAT IS COMPUTATIONAL GRAPH AND DCG

What is a Computational Graph?
A computational graph is like a flowchart that shows how numbers (tensors) are transformed through operations (like addition, multiplication, etc.).

Imagine you have this equation:

z=(x+y)×2

The computational graph would look like this:

   x      y
    \    /
     +  (Addition)
      \
       * (Multiplication by 2)
        \
         z

Each node represents an operation, and edges represent how data flows.


                   What is a Dynamic Computational Graph?

A dynamic computational graph means the graph is built in real-time as operations happen.

In PyTorch, every time you run a tensor operation, the graph 
updates automatically.

This makes debugging and experimenting easier because you don't 
need to define the whole graph beforehand.

You can actually visualize the computational graph using TorchViz! 


import torch
from torchviz import make_dot

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = (x + y) * 2

# Generate and display the computational graph
make_dot(z, params={"x": x, "y": y}).render("graph", format="png", view=True)


'''

#                  BUILDING SIMPLE NEURAL NETWORK
'''
 Neural network are the key components in any model which learns patters
 from real world data and make prediction.

 Neural network consist of inter connected neurons like human brain, each
 neuron will takes input as produces output.

 A neuron will calculate the weighted sum from it's input which we will
 used on activation function.

 All the neuron will have three layers

 1. input layer which take input

 2. hidden layer which performs computations to get pattern.
    size of this neuron will vary from it's complexity of the
    task.

 3. output neuron which produces output, like prediction score.



                        FORWARD PASS


 Is the process of given input to neuron to neuron to obtain output,
 this output will be used to reduce loss function.

 
                        BACKWARD PASS

computing gradient using backpropogation, in pytorch this is done by
backward()


Both of the above process are helps to update the model's parameters.

                        Activation function

Helps to learn the exact patterns from input to a model.


                       Loss function

Finds the value by comparing the predicted output and actual output.
Common loss functions in pytorch are nn.MSELoss() (Mean Squared Error) it
is used when we want to predict continuos value.

nn.CrossEntropy() used when prediction is categorical.


                          OPTIMIZERS

updates the model parameters based on computed gradient.

1. torch.optim.Adam() - adapts learning rate for each parameter.

2. SGD() - stochastic gradient descent, updates model parameter in negative
            direction.


                               Example neural network with pytorch

                    
torch.nn.Module is the class helps to create neurons

import torch
import torch.nn as nn
import torch.optim as optim

# Step 1: Prepare the data (Features and Labels)
X_train = torch.tensor([[150, 0], [130, 0], [180, 1], [170, 1]], dtype=torch.float32)  # Features
y_train = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Labels (0 = Apple, 1 = Orange)

# Step 2: Define a simple Neural Network model
class FruitClassifier(nn.Module):
    def __init__(self):
        super(FruitClassifier, self).__init__()
        self.layer = nn.Linear(2, 2)  # Input: 2 features, Output: 2 classes (Apple, Orange)

    def forward(self, x):
        return self.layer(x)

# Step 3: Initialize the model, loss function, and optimizer
model = FruitClassifier()
loss_function = nn.CrossEntropyLoss()  # For classification
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Step 4: Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()  # Clear gradients
    output = model(X_train)  # Forward pass
    loss = loss_function(output, y_train)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

# Step 5: Make a prediction on new data
X_test = torch.tensor([[160, 1]], dtype=torch.float32)  # New fruit: 160g, rough texture
prediction = model(X_test)
predicted_class = torch.argmax(prediction).item()  # Get the class with the highest score

print("Predicted:", "Apple 🍏" if predicted_class == 0 else "Orange 🍊")


      Diagrammatic Explanation of the Training Process in PyTorch

The training loop consists of multiple steps that help the neural network learn from 
the data. Below is a visual breakdown of each step.

Step 1: Forward Pass
🔹 Inputs (X_train) pass through the model to generate predictions (output).
🔹 Neurons apply weights and biases, followed by an activation function.

[ X_train ]  --->  [ Layer (Weights & Biases) ]  --->  [ Output (Predictions) ]


Step 2: Compute Loss
🔹 Compare the predicted output (output) with the actual labels (y_train).
🔹 Calculate the loss using loss_function.


Prediction (output) vs. Actual (y_train)
       ↓
    Compute Loss
       ↓
  Loss Function (CrossEntropyLoss)


Step 3: Backpropagation
🔹 Calculate the gradients of the loss with respect to weights using loss.backward().
🔹 This step helps in finding which weights contributed to the error.


      Loss
       ↓
 Backpropagation
       ↓
Calculate Gradients (dL/dW)

Step 4: Optimizer Step (Weight Update)
🔹 The optimizer (optimizer.step()) updates the weights based on gradients.
🔹 Gradients are scaled using the learning rate to adjust model parameters.

New Weight = Old Weight - Learning Rate * Gradient


Step 5: Clear Gradients
🔹 Before the next iteration, we reset gradients using optimizer.zero_grad() to avoid accumulation.

Reset Gradients to Zero → Prevent accumulation from previous iterations
📌 Overall Flow of One Epoch

1. Forward Pass  --->  2. Compute Loss  --->  3. Backpropagation  --->  4. Update Weights  --->  5. Clear Gradients

This process repeats for multiple epochs (e.g., 100 times) to improve model accuracy. 🚀

'''
