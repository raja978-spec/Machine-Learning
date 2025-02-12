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

'''
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
