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

#               USE OF TENSORFLOW
'''
Same as pytorch, but it has lite version which is used to develop model 
for mobile applications and it has .js framework too.
'''

#                        TENSOR
'''
* Important built in block in DL
* It is like numpy so we can do all the numpy
  functions and operations but it can't be moved to
  GPU devices to train DL like models.
* Tensor have to_cuda() method that moves tensors to GPU

                   ATTRIBUTES IN TENSOR

1. shape - gives the shape of each dimension
2. dtype - tensor can hold int32,64 float32,64 bool32,64
3. ndim - gives no of dimensions the tensor have

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
])

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
])

print(b.shape)
print(b.ndim)
print(a.dtype)
print(np.where(b>5,0,b))
