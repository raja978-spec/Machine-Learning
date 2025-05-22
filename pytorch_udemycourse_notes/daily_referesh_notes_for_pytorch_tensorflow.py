#              USE OF PYTORCH
'''
Framework used to create and train neural network model

Used when we need to train a model with image, text

Advantage:
Dynamic computation graph easy to debug models unlike tensorflow
Allows tensor operation, tensor is like numpy with more
feature to graphical processing unit(used to train
neural network model)

Has various libraries for computer vision(torchvision)
,NLP(torch text) and Torch serve for model deployment

'''

#                       USE OF TENSORFLOW
'''
Same as pytorch, but it has lite version which is used to develop model 
for mobile applications and it has .js framework too.
'''

#                        TENSOR CREATION IN PYTORCH
'''
* Important built in block in DL
* It is like numpy so we can do all the numpy
  functions and operations but it can't be moved to
  GPU devices to train DL like models.
* Tensor have to_cuda() method that moves tensors to GPU

                   IMPORTANT METHODS IN TENSOR

1. shape - gives the shape of each dimension
2. dtype - tensor can hold int32,64 float32,64 bool32,64
3. ndim - gives no of dimensions the tensor have

                   IMPORTANT ATTRIBUTES IN TENSOR

1. device - helps to specify the device which the tensor
            going to be processed.

                  CREATING TENSOR

import torch
import numpy as np

n_array = np.array([
    [
        [1,2,3],
        [4,5,6]
    ],
    [
        [4,5,6],
        [9,5,7]
    ]
])

a= torch.tensor(n_array)
b= torch.tensor([
    [
        [1,2,4],
        [3,4,5]
    ],
    [
        [5,6,7],
        [7,8,9]
    ]
], device='cpu')

print(b.shape)
print(b.ndim)
print(a.dtype)
print(np.where(b>5,0,b))

OUTPUT:

torch.Size([2, 2, 3])
3
torch.int32
[[[1 2 4]
  [3 4 5]]

 [[5 0 0]
  [0 0 0]]]

  
'''

#              VARIOUS WAY OF CREATING TENSORS IN PYTORCH
'''

                     ZEROS AND ONES TENSOR

import torch
a = torch.zeros(3,3)
b = torch.ones(3,3)
print(a,b)

                          RANDOM TENSOR
import torch
a = torch.rand(3,3)
b= torch.randn(3,3)
c=torch.arange(start=12, end=20,step=2) # 1D tensor
d=torch.linspace(start=0, end=0.5, steps=3) # 1D tensor
print(a)
print(b)
print(c)
print(d)


OUTPUT:
tensor([[0.4832, 0.7571, 0.4526],
        [0.2056, 0.2431, 0.4439],
        [0.9242, 0.7031, 0.0547]])
tensor([[-0.9875,  0.5152,  0.6413],
        [-0.7969, -1.4287,  1.2963],
        [ 0.8523, -2.1810,  0.2610]])
tensor([12, 14, 16, 18])
tensor([0.0000, 0.2500, 0.5000])
'''

#                 TENSOR CREATION IN TENSORFLOW
'''
import tensorflow as tf
import numpy as np

np_a = np.array([1,3,5])
a= tf.constant(np_a)
b=tf.range(start=10, limit=15)
print(a)
print(type(b),b)

##########OUTPUT################
tf.Tensor([1 3 5], shape=(3,), dtype=int32)
<class 'tensorflow.python.framework.ops.EagerTensor'> tf.Tensor([10 11 12 13 14], shape=(5,), dtype=int32)
'''

#         TENSORFLOW ARITHMETIC OPERATION IN TENSORFLOW
'''

add,subtract,multiply,divide(t1 or only one value, t1 or only one value)

import tensorflow as tf
import numpy as np

with tf.device('cpu'):
    np_a = np.array([1,3,5])
    a= tf.constant(np_a)
    b=tf.range(start=10, limit=13)
    print(tf.add(a,b))
    print(tf.subtract(a,b))
    print(tf.multiply(a,2))
    print(tf.divide(a,2))

###############OUTPUT####################
tf.Tensor([11 14 17], shape=(3,), dtype=int32)
tf.Tensor([-9 -8 -7], shape=(3,), dtype=int32)
tf.Tensor([ 2  6 10], shape=(3,), dtype=int32)
tf.Tensor([0.5 1.5 2.5], shape=(3,), dtype=float64)
'''

#              TENSOR ARITHMETIC OPERATIONS ON PYTORCH
'''
import torch as tf
a=tf.Tensor([1,2,4])
b=tf.Tensor([5,6,7])
print(tf.add(a,b))
print(tf.subtract(a,b))
print(tf.mul(a,b))
print(a/b)

OUTPUT:
tensor([ 6.,  8., 11.])
tensor([-4., -4., -3.])
tensor([ 5., 12., 28.])
tensor([0.2000, 0.3333, 0.5714])
'''


#            OTHER MATRIX OPERATIONS ON TENSORFLOW TENSOR ANS SORT
'''
import tensorflow as tf

a = tf.constant([[1,3,4],[5,4,3],[5,3,2]], dtype=tf.float16) # Nd tensors will give error

print(tf.square(a))
print(tf.sqrt(a))
print(tf.linalg.inv(a))
print("Before sort")
print(a)
print('After sort')
print(tf.sort(a, axis=0)) # column sort
'''

#      OTHER MATHEMATICAL OPERATIONS AND CONDITIONAL MASK IN PYTORCH TENSOR
'''
import torch as tf
import numpy as np

a = tf.Tensor([[12,4,3],[1234,131,13],[14,324,2],[434,24,13]])

print(tf.mean(a))
print(tf.std(a))
print(a[a>324])
print(np.where(a>324,0,a))

OUTPUT:
tensor(184.)
tensor(360.3357)
tensor([1234.,  434.])
[[ 12.   4.   3.]
 [  0. 131.  13.]
 [ 14. 324.   2.]
 [  0.  24.  13.]]
'''