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

#            LOADING AND PREPROCESSING DATA IN PYTORCH
'''
 pytorch provides various pre processing techniques that used to 
 give correct data to the model without any noise.

 we'll work with torchvision.dataset and torch.utils.Dataset 
 to pre processing

 torchvision.dataset - for standard dataset
 torch.utils.Dataset - custom data set

 ToTensor from transforms one of the data pre processing function helps
 to change images to tensor and normalization etc.


 EX:

import torchvision.datasets as dataset
from torchvision.transforms import ToTensor

minst_train = dataset.MNIST(root='data', train=True, transform=ToTensor(), download=True)
minst_test = dataset.MNIST(root='data', train=False, transform=ToTensor(), download=True)


        EXAMPLE FOR CUSTOM DATASET


This are often used to load data from different sources.

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import numpy as np

class CustomDataSet(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sample = self.data[index]
        label = self.labels[index]

        if(self.transform):
            sample = self.transform(sample)
        return sample, label

data = CustomDataSet(np.ndarray([1,1,2,5]),np.ndarray([0,2,3,1]),ToTensor())
print(data.__getitem__(0))


                 EXAMPLE FOR NORMALIZATION

use of this normalize is to scaling the input data to given range

EX:

from torchvision.transforms import Normalize
import torch

# Define a sample tensor (grayscale image with one channel)
x = torch.tensor([[[1, 3, 3], [5, 3, 3], [4, 6, 3]]], dtype=torch.float32)


# Define normalization
normalize = Normalize(mean=[0.5], std=[0.5])

# Apply normalization
x = normalize(x)


print(x)# new_value = each_tensor_value - mean / std

OUTPUT:
tensor([[[ 1.,  5.,  5.],
         [ 9.,  5.,  5.],
         [ 7., 11.,  5.]]])


                  EXAMPLE FOR IMAGE NORMALIZATION

from torchvision.transforms import Normalize, ToTensor, Resize, Compose, RandomHorizontalFlip
import torch

# transform for image
transforms = Compose([
    Resize((28,28)),
    ToTensor(),
    RandomHorizontalFlip()
])


                       EXAMPLE FOR DATALOADER

Helps to load data in batch manner, shuffles data, allows parallel loading
Improve efficiency of training with large data

Large no of data are getting into splitted in different specified batch
with batch_size attribute.

If your system have more than one GPU then the training data will be
parallely loaded to all GPU devices to speeding up the training process
of the model. For this num_worker in dataloader are used.

from torchvision.transforms import Normalize, ToTensor, Resize, Compose, RandomHorizontalFlip
from torch.utils.data import Dataset, DataLoader
import os
import PIL

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
    
    def __len__(self):
        self.size = len(self.image_files)

    def __getitem__(self, index):
        image = self.image_files[index]
        if(self.transform):
            image_path = os.path.join(self.image_dir,image)
            self.transform(image_path)
            PIL.Image.open(image_path)


# transform for image
ImageTransforms = Compose([
    Resize((28,28)),
    ToTensor(),
    RandomHorizontalFlip()
])

custom_data = ImageDataset(image_dir='path to image', transform=ImageTransforms)

# num_worker for parallel loading
data = DataLoader(custom_data, batch_size=32, shuffle=True, num_worker=3)



                ITERATE IN BATCH

for batch in data:
    inputs, lables = batch
    # set no grad
    # model output (forward)
    # pass output to loss
    # backward
    # step optimizer


''' 

#                     MODEL EVALUATION AND VALIDATION
'''
By evaluating this we can see if the model has overfitted and underfitted.

                           MODEL EVALUATION

Models performance are evaluated using following metrics

1. Accuracy                - sum of correct prediction / total no of prediction
                             
                              Accuracy can be misleading because of
                              imbalanced dataset.

2. Precision and Recall     - Precision measure accuracy of the model's positive prediction.
                              It is mainly used when false positive increases

                              True positive / true positive + false positive 

                              Recall ensures that can a model predicts all
                              true predictions. It is used when False negative
                              is in high.

                              True positive / True positive + False negative

                              In a cancer based system a model doesn't predicts
                              the person who have cancer will make huge risk. 

                              
3. F1 Score   -   Used when both precision and recall is important

                  f1-score = 2 * (precision * recall / precision + recall)

                  It is particularly useful in imbalanced dataset.




                       MODEL  VALIDATION

Used to see how well the model performed on unseen data, or well it 
generalize in unseen data.

1. Train val split - validation set of data are helps to measure how
                     well model predicts unseen data.

                     EX: train_test_split()

                     It helps to prevent over fitting.


2. k- fold cross validation - all the data will become train dataset and
                              validation set. 

                              It splits the data into k fold from that
                              fold some part is used for training and
                              remaining part is used for testing, and finally
                              it calculate mean of all k folds that is
                              the value given by this function.

                              This computation is not good for bigger
                              dataset or complex model.

3. Stratified k-Fold - varation of k-fold that ensure model are trained
                       with all classes.

                       It is particularlly used in classification 
                       imbalanced dataset.


            MONITORING THE MODEL PERFORMANCE DURING TRAINING

It can be done by plotting the loss values of each epoch in training.

EX:

import matplotlib.pyplot as pylt

pylt.plot(train_loss)
pylt.plot(val_loss)
pylt.show()

It is mainly used to detect overfitting and underfitting earlier.

Both train,val loss are decreased in same manner - model learning well
trains loss decreasing, validation loss increasing - overfitting
train loss increase, validation loss increase - underfitting

SOLUTION FOR OVERFITTING:

Accuracy curves, Early stopping(prevent overfitting)

Following are the more solutions to prevent ovefitting

import torch
import torchvision

model = torch.nn.Linear(in_features=10, out_features=5)

#apply L2 regularization
optimizer = torch.optim.SGD(model.parameters, lr=0.01, weight_decay=0.01)
drop_out = torch.nn.Dropout(p=0.5)
data_augmantation = torchvision.transforms.RandomHorizontalFlip()


SOLUTIONS FOR UNDERFITTING

1. Create more neural layers
2. Increase training epoch
3. Reduce regularization
'''

#                     ADVANCE NEURAL NETWORK WITH PYTORCH(CNN, RNN)
'''
 Traditional feedforward network is not efficient to train a data like
 image and text, for this CNN and RNN,  deep neural networks are used

 1. CNN - takes input image and apply filters(kernal) to take the main
          part of the images like edges, horizontal edges. This process
          is called feature mapping.

          Filters: It is small metrics used to detect specific feature of
                   input data, values of this filters found during training
                   process.

          stride: Determines how filter moves each step

          padding: used to maintain original dim of data

          Pooling: It is a layer which reduces the spatial dim for more speed computations
                   It can be max and average pooling.

          
                      Common CNN architecture

1. LeNet - used to recognize handwritten digit in minst dataset.
           Has 2 layers, pooling and fully connected dense layer.

2. AlexNet - Used for large scale image classification, has more layers
             like ReLu activations and DropOut for regularization.

3. VGG - Widely used in image classification, feature extraction, has
         deep layer with small 3X3 filter, improves model performance

4. ResNet - Used for object detection and image classification, provides
            more much more deeper layer, finds shortcut connection and skips
            one or more layer to find better one.


                     RECURRENT NEURAL NETWORK

Handel sequential data like time series, natural language processing
speech recognization, unlike normal independant dataset it depends on
it's previous data, Has memory to maintian previuos data until the current
data gets arrived.

EX: meaning of a word is depending on memorizing each next letters.


                            LSTM 

long short time memory Overcomes the limitation of RNN, RNN is not used for
long dependencies in RNN it stores each letter while this stores each word or
sentence, paragraph

EX: it predicts the next letter of the senetence by memorizing large sentence.


                         APPLICATION OF LSTM

Used in sentiment analysis, language translation for text, stock price prediction,
weather forecasting for time series analysis

EX: LSTM predicts next days stock price by the previous data.

Has input, forgot, output gates.

                      TRANSFORMER NETWORK

This network models captures long dependencies more effectively than RNNs
it will predict same word with different meaning that is called self attention.

In transformer architecture it has encoder to process input, decoder to generate
output. Positional indexing are used to put the indexes on the give text
data to understand the different relation on a word.

EX: website like gpt(generative pretrained transformer) uses this 
    architecture it process input text and generates output texts.
'''

#            TRANSFER LEARNING AND FINE TUNING
'''
 Pre trained model learned patterns from large dataset, by transfering
 this learning to new task we can reduce learning time of a model.

 The approach is used when we have small dataset for new task

 It improves performance when target data is small.

 Pre trained model available in pytorch in torchvision resnet50, it
 has pre trained model for image classification and object detection

 EX:

import torchvision.models as models
resnet = models.resnet50(pretrained=True)

You can use this model as it is, or you can fine tuning the model to
adapt small new task.


             PRE-TRAINED MODEL ON HUGGING FACE

Provides pre-trained model for NLP, text classification, text generation.
sentiment analysis.

BERT is the pre-trained model.


from tranformers import BertModel

model = BertModel.from_pretrained('bert-base-uncased')


                  FEATURE EXTRACTION FOR SPECIFIC DOMAIN

Involves further training the pre trained model for specific task

It freezes the earlier layer of pre trained model and replace 
last layer by adding new layer for new task

While training the pre trained model it is important to select good
hyper parameter tuning like learning rate, batch size, no of epochs
to avoid overfitting, especially 


import torchvision.models as models
resnet = models.resnet50(pretrained=True)
import torch

for params in resnet.parameters():
    params.requires_grad = False

num_classes = 10
resnet.fc = torch.nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes)


                      FINE TUNING

In this we will not frozen the earlier layers of pre trained model we used
the pertrained model directly on dataset, It it mainly used if the new task is
similar to the model trained task.

for params in resnet.parameters():
    params.requires_grad = True

model.train()

EX: use a resnet model which learn the image edges already are further
    trained to recognize medical images.
'''


#                   HANDLING COMPLEX DATA
'''

                   IMAGE HANDING

Data Augmentation for image dataset improves generalization
  
   EX:

   Randomly selecting the subset of the image crop it.

   from torchvision.transforms import RandomCrop
   trand = RandomCrop(size=(24,24))

   Like this RandomHorizontalFlip and RandomRotation are used.

   RandomHorizontalFlip(p=0.5)

   RandomRotation(degree=45)

   ColorJitter(brighness=0.5, contrast, saturation, hue)

   AdjustBrightness(brightness_factor=0.3)
   

                     TEXT HANDLING

Text data preprocessing - Tokenization represents smallest unit of meaning

   Word Level Tokenization  - split text into words, each word considered as
                              seperate token, used on text classification and
                              sentiment analysis

from nltk.tokenize import word_tokenize

trand = word_tokenize("This is simple text")

    
   Character level tokenization - splits text into char, each char considered
                                  as token, used for language translation
                                  txt generation

                                trand = list("This is simple text")


                HANDLING SEQUENCE OF VARIABLE LENGTH

1. Padding - helps to change the short sequence to match neural network
             input length

from keras.preprocessing.sequence import pad_sequences

trand = pad_sequences("This is simple text", maxlen=100, padding='post')


2. truncation - truncat sequence to match input lenght if sequence is too long

from keras.preprocessing.sequence import pad_sequences

trand = pad_sequences("This is simple text", maxlen=100, truncating='post')




                        TIME SERIES DATA HANDLING


1. Temporal Convolution Neural Network - TCN adapted to handle time series
                                         data by applying filter

2. RNN             

'''

#                MODEL DEPLOYMENT
'''
                   SAVING MODEL

import torch

model = torch.nn.Sequential([
    torch.nn.Linear(in_features=100, out_features=50),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=50, out_features=10)
])

# model.state_dict() contains model weights and biases. 
# it is recommended
torch.save(model.state_dict(), 'model.pth')

# saving entire model
# this is not recommended especially when dealing with
# dynamic computational graph
torch.save(model, 'model.pth')


                        LOAD MODEL

import torch
from sompath import SampleModelClass

# To state model we need to initial the
# model archintecture first
model = SampleModelClass()
model.load_state_dict(torch.load('model.pth'))


# Load entier model
model.load('model.pth')

                           
                           DEPLOY MODEL

For deploy model we need to serialize them in a format that can run 
in any platform. For this jiit script used here, that converts our
pytorch model to script that can be run in any platform without 
python run time.

import torch

model = torch.nn.Sequential([
    torch.nn.Linear(in_features=100, out_features=50),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=50, out_features=10)
])

scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model, 'model_script.pt')

This can be deployed and can be runnable on any devices like mobile
devices and edge devices.

import torch

model = torch.nn.Sequential([
    torch.nn.Linear(in_features=100, out_features=50),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features=50, out_features=10)
])

scripted_model = torch.jit.script(model)
torch.jit.save(scripted_model)


                      SERVING A MODEL TO FLASK, FASTAPI and
                                AWS Lambda

            REFER severing_model_to_python_backendframewors.docx
'''

#                 DEBUG DEPLOYED MODEL
'''
 Before going to build a model we have to keep in mind one thing that
 our model should capable to process NAN and infinite like values

 And also by seeing real world data after deploying model's performance might
 be reduced, so we have to debug it to make the correct prediction on 
 real world data.

 REFER maintaining deployed model.docx to get the idea of how to debug
 and overcome from unwanted common pytorch errors
'''

#         DISTRIBUTED TRAINING AND PERFORMANCE OPTIMIZATION
'''
 Distributed training uses multiple GPU to train your model faster

EX:

import torch

import torch.distributed as dist

# initilize the communication between GPUs
dist.init_process_group(backend='nccl',init_method='env://')

# wrap your model to distributed parallel
# this function breaks all dataset to different GPU devices and
# the weights and biases of each GPU device will be synchronized.

distributed_training = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

# ensure each process are assigned to each GPU
torch.cuda.set_device(local_rank)
model.cuda(local_rank)



                       DISTRIBUTED SAMPLER

Ensure all GPUs has different sub sets of data by using this distributed
sampler.

import torch

train_sampler = torch.utils.data.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler)


     REFER model_perfomance_optimizer.docx for more optimization
     techniques


'''      

#                     CUSTOM LAYERS, LOSS FUNCTIONS
'''
 custom layers allows to do unique operation which are not present
 in pytorch

EX FOR CUSTOM LAYERS:

import torch.nn as nn
import torch
class CustomLayer(nn.Module):
    def __init(self, in_features, out_features):
        super(CustomLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

        # custom the initlization weight parameter using this
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self,x):
        x = self.linear(x)
        x = self.relu(x)
        return x


model = nn.Sequential([
    CustomLayer(100,50),
    nn.Linear(50,10)
])




EXAMPLE FOR CUSTOM LOSS FUNCTION:


import torch.nn as nn
import torch

import torch.nn as nn
import torch

class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        loss = torch.nn.functional.mse_loss(output, target) + 0.1 * torch.mean(output)
        return loss

criterion = CustomLoss()
loss = criterion(output, target)


And you can create custom metrix function like normal python functions
that evaluate the performance of model on test data or validation data.

And also we can use this same nn.Module to create custom generalization
like drop 

                    
                    ADVANCED ACTIVATION FUNCTION(SWISH, MISH, GELU)

import torch.nn as nn
import torch

class Swish(nn.Module):

    def forward(self, x): # normal relu activation function has not
                          # this implementation, it iw work will
                          # on some deep learning nerual network
        return x * torch.sigmoid(x)

model = nn.Sequential([
    nn.Linear(50,10),
    Swish(),
    nn.Linear(10,5)
])
    

MISH - used in CNN network
GELU - used in RNN like model, it is used in BERT like model
'''


#            REPRODUCIBILITY OF MODEL PREDICTION
'''
 It allows other users to verify your model's prediction, it is very
 crucial because various users have various device capability but
 still some of the techniques that can be used on torch to give the
 same good prediction result to all users.

 REFER reproducibility_model.docx for such methods
'''

#            OTHER LIBRARY INTEGRATION WITH PYTORCH
'''
 pytorch already allows to integrate other libraries like OpenCV,
 NLTK, spaCY

 And also we can use tensorflow models on pytorch anf pytorch models
 to tensorflow by converting the models into a common format with
 ONNX(open neural network exchange).

 ONNX is the open format that allows model to communicate different
 frameworks.

 onnxkeras is the library helps to load keras models.

 REFER other_library_iteration_with_pytorch.docx for code
'''
