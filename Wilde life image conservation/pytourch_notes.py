#  WHAT IS PYTORCH ?

'''
 The term tensor comes from mathematics. 
 It refers to an array of values organized into one or more dimensions.

 In Python, there are several libraries for creating and 
 manipulating tensors. In this program, we'll use PyTorch, which is 
 built for deep learning. We'll build our computer visions with 
 PyTorch.

 Example creation of tensors in python:

 import os
 import sys

 import matplotlib
 import matplotlib.pyplot as plt
 import pandas as pd
 import PIL
 import torch
 import torchvision
 from PIL import Image
 from torchvision import transforms

 my_values = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
 my_tensor = torch.Tensor(my_values)

 print("my_tensor class:", type(my_tensor))
 print(my_tensor)

 OUTPUT:

 my_tensor class: <class 'torch.Tensor'>
 tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]])

'''

# SHAPE ,DTYPE, DEVICE ATTRIBUTE IN TENSOR

'''
 my_tensor = torch.Tensor(my_values)
 my_tensor.shape() # OUTPUT:  torch.Size([4, 3])
 my_tensor.dtype() #  torch.float32        

 CPU will sets for smaller training data set, but GPU(Graphics processing
 unit) are deals with large image matrix and which required to check
 if the system has GPU to do deep learning neural network in ML.

 For this device attribute are helped

 print("my_tensor device:", my_tensor.device)
 my_tensor device: cpu

'''

#  ACCESSING GPU ON LINUX, WINDOWS

'''

 To do deep learning we want to access the GPU of the system
 that can be done by the following way.

 The cuda package is used to access GPUs on Linux and Windows machines; 
 mps is used on Macs. 

 The following code checks the availability of cuda

 # Check if GPUs available via `cuda`
 cuda_gpus_available = torch.cuda.is_available()

 # Check if GPUs available via `mps`
 mps_gpus_available = torch.backends.mps.is_available()

 print("cuda GPUs available:", cuda_gpus_available)
 print("mps GPUs available:", mps_gpus_available)

'''

#   MOVING TENSORS TO CUDA TO DO DL
'''            
 my_tensor = torch.Tensor(my_values)
 my_tensors = my_tensors.to('cuda')

 print("my_tensor device:", my_tensor.device)

 OUTPUT: my_tensor device: cuda:0
'''

#      SLICING IN TENSOR   

'''

 SYNTAX: tensor_name[startrowindex:endrowindex,startcolumnindex:endcolumnindex]

 tensor tensor([[ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 7.,  8.,  9.],
        [10., 11., 12.]], device='cuda:0')


 left_tensor = my_tensor[:2, :]

 OUTPUT:
 tensor([[1., 2., 3.],
        [4., 5., 6.]], device='cuda:0')

 right_tensor = my_tensor[2:, :]

 OUTPUT:
 tensor([[ 7.,  8.,  9.],
        [10., 11., 12.]], device='cuda:0')

'''

#          ADD IN TENSOR
'''
 summed_tensor_operator = left_tensor + right_tensor
 summed_tensor_method = left_tensor.add(right_tensor)

 print("summed_tensor_operator class:", type(summed_tensor_operator))
 print("summed_tensor_operator shape:", summed_tensor_operator.shape) 
 print("summed_tensor_operator data type:", summed_tensor_operator.dtype)
 print("summed_tensor_operator device:", summed_tensor_operator.device)
 print(summed_tensor_operator)
 print()
 print("summed_tensor_method class:", type(summed_tensor_method))
 print("summed_tensor_method shape:", summed_tensor_method.shape)
 print("summed_tensor_method data type:", summed_tensor_method.dtype)
 print("summed_tensor_method device:", summed_tensor_method.device)
 print(summed_tensor_method)


 OUTPUT:

 summed_tensor_operator class: <class 'torch.Tensor'>
 summed_tensor_operator shape: torch.Size([2, 3])
 summed_tensor_operator data type: torch.float32
 summed_tensor_operator device: cuda:0
 tensor([[ 8., 10., 12.],
        [14., 16., 18.]], device='cuda:0')

 summed_tensor_method class: <class 'torch.Tensor'>
 summed_tensor_method shape: torch.Size([2, 3])
 summed_tensor_method data type: torch.float32
 summed_tensor_method device: cuda:0
 tensor([[ 8., 10., 12.],
        [14., 16., 18.]], device='cuda:0')
'''

#   MUL IN TENSOR (ELEMENT WISE MUL)

'''
 element wise multiplication 


 ew_tensor_operator = left_tensor * right_tensor
 ew_tensor_method = left_tensor.mul(right_tensor)

 # Note that element-wise multiplication is commutative
 # The below will be true

 left_tensor * right_tensor == right_tensor * left_tensor 

 print("ew_tensor_operator class:", type(ew_tensor_operator))
 print("ew_tensor_operator shape:", ew_tensor_operator.shape)
 print("ew_tensor_operator data type:", ew_tensor_operator.dtype)
 print("ew_tensor_operator device:", ew_tensor_operator.device)
 print(ew_tensor_operator)
 print()
 print("ew_tensor_method class:", type(ew_tensor_method))
 print("ew_tensor_method shape:", ew_tensor_method.shape)
 print("ew_tensor_method data type:", ew_tensor_method.dtype)
 print("ew_tensor_method device:", ew_tensor_method.device)
 print(ew_tensor_method)

 OUTPUT:

 ew_tensor_operator class: <class 'torch.Tensor'>
 ew_tensor_operator shape: torch.Size([2, 3])
 ew_tensor_operator data type: torch.float32
 ew_tensor_operator device: cuda:0
 tensor([[ 7., 16., 27.],
        [40., 55., 72.]], device='cuda:0')

 ew_tensor_method class: <class 'torch.Tensor'>
 ew_tensor_method shape: torch.Size([2, 3])
 ew_tensor_method data type: torch.float32
 ew_tensor_method device: cuda:0
 tensor([[ 7., 16., 27.],
        [40., 55., 72.]], device='cuda:0')     
'''

#  MATMUL IN TENSOR (NOT ELEMENT WISE)

'''
 This will helpful to do mul if no of rows and columns are not equal
 with two tensors, matmul() is the method does it, and symbol @ also
 used for this

 EX:

 new_left_tensor = torch.Tensor([[2, 5], [7, 3]])
 new_right_tensor = torch.Tensor([[8], [9]])

 mm_tensor_operator = new_left_tensor @ new_right_tensor
 mm_tensor_method = new_left_tensor.matmul(new_right_tensor)

 OUTPUT FOR BOTH:

 tensor([[61.],
        [83.]])

 CALCULATION:

 First Row of new_left_tensor ([2, 5]) × new_right_tensor ([8, 9]):
 (2 * 8) + (5 * 9) = 16 + 45 = 61
 
 Second Row of new_left_tensor ([7, 3]) × new_right_tensor ([8, 9]):
 (7 * 8) + (3 * 9) = 56 + 27 = 83
 
 One very important thing to remember: matrix multiplication 
 is **not commutative**. The number of columns in the tensor on 
 the left must equal the number of rows in the tensor on the right. 
 If these two dimensions don't match, your code will throw a 
 `RunTimeError`.


                       NOTE

  Matrix multiplication is the way your models will train and make 
  predictions, and dimension mismatches will be a common source of 
  bugs when you start building models. For that reason, it's always 
  important to check the shape of your tensors.

'''

#     MEAN IN TENSOR
'''
 my_tensor_mean = my_tensor.mean()

 print("my_tensor_mean class:", type(my_tensor_mean))
 print("my_tensor_mean shape:", my_tensor_mean.shape)
 print("my_tensor_mean data type:", my_tensor_mean.dtype)
 print("my_tensor_mean device:", my_tensor_mean.device)
 print("my_tensor mean:", my_tensor_mean)

 OUTPUT:

 my_tensor_mean class: <class 'torch.Tensor'>
 my_tensor_mean shape: torch.Size([])
 my_tensor_mean data type: torch.float32
 my_tensor_mean device: cuda:0
 my_tensor mean: tensor(6.5000, device='cuda:0')


 MEAN:

 Mean= Number of elements in the tensor / 
       Sum of all elements in the tensor
​  

  FIND MEAN FOR ONLY ROW OR ONLY COLUMN

  my_tensor_column_means = my_tensor.mean(dim=[0])

 print("my_tensor_column_means class:", type(my_tensor_column_means))
 print("my_tensor_column_means shape:", my_tensor_column_means.shape)
 print("my_tensor_column_means data type:", my_tensor_column_means.dtype)
 print("my_tensor_column_means device:", my_tensor_column_means.device)
 print("my_tensor column means:", my_tensor_column_means) 

 OUTPUT:

 my_tensor_column_means class: <class 'torch.Tensor'>
 my_tensor_column_means shape: torch.Size([3])
 my_tensor_column_means data type: torch.float32
 my_tensor_column_means device: cuda:0
 my_tensor column means: tensor([5.5000, 6.5000, 7.5000], device='cuda:0')

'''

#     EXPLORING TRAINING DATA (OS MODULE)
'''

 Before that we want to know os

 import os

 # Joining two components
 result = os.path.join("folder", "file.txt")
 print(result)

 OUTPUT: folder/file.txt

 Why Use os.path.join() Instead of String Concatenation?

 Platform Independence:

 path = "folder" + "/" + "file.txt"  # May fail on Windows
 Using os.path.join() ensures correct separators for the OS.

'''

#   LISTDIR IN OS

'''
 # gives list of folders in that dir
 class_directories = os.listdir(train_dir)

 print("class_directories type:", type(class_directories))
 print("class_directories length:", len(class_directories))
 print(class_directories)


 OUTPUT:

 class_directories type: <class 'list'>
 class_directories length: 8
 ['hog', 'blank', 'monkey_prosimian', 'antelope_duiker', 'leopard', 'civet_genet', 'bird', 'rodent']

'''

#     SERIES IN PD
'''
 A Series in pandas is a one-dimensional labeled array capable of 
 holding any data type, such as integers, floats, strings, 
 or even objects. It is similar to a list or array, but with an 
 associated index that labels each element.

 EX:

 import pandas as pd

 data = [10, 20, 30, 40]
 s = pd.Series(data)
 print(s)

 OUTPUT:

 0    10
 1    20
 2    30
 3    40
 dtype: int64

 _______________________________________
 
 data = {'a': 10, 'b': 20, 'c': 30}
 s = pd.Series(data)
 print(s)

 OUTPUT:

 a    10
 b    20
 c    30
 dtype: int64

 _________________________________________

 data = [100, 200, 300]
 s = pd.Series(data, index=['x', 'y', 'z'])
 print(s)

 OUTPUT:

 x    100
 y    200
 z    300
 dtype: int64


'''

#    MARPLOTLIN SUBPLOTS

'''
 In Matplotlib, subplots allow you to create multiple plots 
 (charts, graphs, or visualizations) within a single figure. 
 This is especially useful when you want to compare multiple datasets 
 or present related visualizations side-by-side.

 Key Concepts

 Figure: The overall container for all plots.
 Axes: The individual plots within the figure.
 Using subplots, you can arrange multiple axes in a grid-like 
 layout within a figure


 plt.subplot()
 This creates a single subplot within a grid.
 Syntax: plt.subplot(nrows, ncols, index)
 nrows: Number of rows in the grid.
 ncols: Number of columns in the grid.
 index: The position of the current subplot (1-based indexing).

 import matplotlib.pyplot as plt

 # Create a figure with 2 rows and 1 column of subplots
 plt.subplot(2, 1, 1)  # First subplot
 plt.plot([1, 2, 3], [4, 5, 6])
 plt.title("First Plot")

 plt.subplot(2, 1, 2)  # Second subplot
 plt.bar([1, 2, 3], [3, 2, 1])
 plt.title("Second Plot")

 plt.tight_layout()  # Adjust spacing
 plt.show()

'''


#       MATPLOT LIB SUBPLOTS FIGSIZE

'''
 The figsize parameter in Matplotlib's plt.subplots() 
 (and other related functions like plt.figure()) is used 
 to set the size of the figure, which is the entire drawing 
 area containing all the subplots, axes, and decorations 
 (titles, labels, etc.).

 Syntax of figsize
 fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))

 width: The width of the figure in inches.
 height: The height of the figure in inches.

 Why Use figsize?
 
 To ensure the plots are properly sized and not too cramped.
 To improve readability of complex visualizations with many subplots.
 To adjust the figure dimensions for display or saving purposes.

 EX:

 import matplotlib.pyplot as plt

 # Create a figure with a specific size
 fig, axes = plt.subplots(2, 2, figsize=(10, 6))  # Width: 10 inches, Height: 6 inches

 # Add some plots
 axes[0, 0].plot([1, 2, 3], [4, 5, 6])
 axes[0, 0].set_title("Line Plot")

 axes[0, 1].bar([1, 2, 3], [3, 2, 1])
 axes[0, 1].set_title("Bar Chart")

 axes[1, 0].scatter([1, 2, 3], [3, 2, 1])
 axes[1, 0].set_title("Scatter Plot")

 axes[1, 1].hist([1, 1, 2, 3, 3, 3], bins=3)
 axes[1, 1].set_title("Histogram")

 # Adjust spacing
 plt.tight_layout()

 # Display the plots
 plt.show()

'''


#   CREATING BAR CHART WITH SERIES, SUNPLOTS FIGSIE

'''
 # Create a bar plot of class distributions
 fig, ax = plt.subplots(figsize=(10, 5))

 # Plot the data
 ax.bar(class_distributions.index, class_distributions.values)  # Write your code here
 ax.set_xlabel("Class Label")
 ax.set_ylabel("Frequency [count]")
 ax.set_title("Class Distribution, Multiclass Training Set")
 plt.xticks(rotation=45)
 plt.tight_layout()
 plt.show()

'''

#     PIL LIBRARY 

'''
 Used to process and view image based data

 from PIL import Image

 # Define path for hog image
 hog_image_path = os.path.join(train_dir, "hog", "ZJ000072.jpg")

 # Define path for antelope image
 antelope_image_path = os.path.join(train_dir, "antelope_duiker", "ZJ002533.jpg")

 print("hog_image_path type:", type(hog_image_path))
 print(hog_image_path)
 print()
 print("antelope_image_path type:", type(antelope_image_path))
 print(antelope_image_path)

 hog_image_pil = Image.open(hog_image_path)

 print("hog_image_pil type:", type(hog_image_pil))

 OUTPUT:

 hog_image_pil type: <class 'PIL.JpegImagePlugin.JpegImageFile'>
 hog_image_pil #OPENS IMAGE
'''

#   SIZE AND MODE IN PIL LIBRARY

'''
 size - used to view the pixel size of the image
 mode - used to view the color mode of the image where there 
        it is RGB or RGBA or grayscale (mode="L").

 RGB image will have more size than grayscale image.

 EX:

 antelope_image_pil = Image.open(antelope_image_path)
 
 # Get image size 
 antelope_image_pil_size = antelope_image_pil.size

 # Get image mode
 antelope_image_pil_mode = antelope_image_pil.mode

 # Get image mode
 print("antelope_image_pil_size class:", type(antelope_image_pil_size))
 print("antelope_image_pil_size length:", len(antelope_image_pil_size))
 print("Antelope image size:", antelope_image_pil_size)
 print()
 print("antelope_image_pil_mode class:", type(antelope_image_pil_mode))
 print("Antelope image mode:", antelope_image_pil_mode)


 OUTPUT:

 antelope_image_pil_size class: <class 'tuple'>
 antelope_image_pil_size length: 2
 Antelope image size: (960, 540)

 antelope_image_pil_mode class: <class 'str'>
 Antelope image mode: RGB
'''

#                         LOAD TENSORS

#  TRANSFORM MODULE IN TORCHVISION

'''
  This is a module in Torch vision that helps to transform actual images
  to tensors, we need tensors to work with deep learning neural
  networks.

 antelope_image_pil = Image.open(antelope_image_path)
 antelope_tensor = transforms.ToTensor()(antelope_image_pil)

 print("antelope_tensor type:", type(antelope_tensor))
 print("antelope_tensor shape:", antelope_tensor.shape)
 print("antelope_tensor dtype:", antelope_tensor.dtype)
 print("antelope_tensor device:", antelope_tensor.device)

 OUTPUT:

 antelope_tensor type: <class 'torch.Tensor'>
 antelope_tensor shape: torch.Size([3, 540, 960])
 antelope_tensor dtype: torch.float32
 antelope_tensor device: cpu
'''

#  LOAD TRANSFORMED DATA TO IMSHOW
'''
 imshow() is the function helps to see the tensor
 images

 The transform ToTensor() will return 3 dim array as
 it's shape

 antelope_image_pil = Image.open(antelope_image_path)
 antelope_tensor = transforms.ToTensor()(antelope_image_pil)

 antelope_tensor.shape
 antelope_tensor shape: torch.Size([3, 540, 960])

 1st dim represents the color mode of the image, here we have
 3 so image will have 3 colors Red, green, blue

 2nd dim represents the rows and 3rd dim represents columns of the
 image pixel.

 We can access each color and pass it to imshow() to see the
 images, cmap() is used here

 In Matplotlib's imshow() function, the cmap parameter stands 
 for colormap. A colormap is a mapping of data values to colors, 
 used to visually represent numerical data in an image. 
 It is especially useful for displaying 2D arrays as images, 
 where the array values are represented by different colors.

 EX:

 # Create figure with 3 subplots
 fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5))

 # Plot red channel
 red_channel = antelope_tensor[0, :, :]
 ax0.imshow(red_channel, cmap="Reds")
 ax0.set_title("Antelope, Red Channel")
 ax0.axis("off")

 # Plot green channel
 green_channel = antelope_tensor[1, :, :]
 ax1.imshow(red_channel, cmap="Greens")
 ax1.set_title("Antelope, Green Channel")
 ax1.axis("off")


 # Plot blue channel
 blue_channel = antelope_tensor[2, :, :]
 ax2.imshow(red_channel, cmap="Blues")
 ax2.set_title("Antelope, Blue Channel")
 ax2.axis("off")

 plt.tight_layout()
 plt.show()

'''

#  AMIN, MAX  IN TOTENSOR
'''
 Helps to find the min and max value of the converted
 image tensor

 max_channel_values = antelope_tensor.amax()
 min_channel_values = antelope_tensor.amin()


 print("max_channel_values class:", type(max_channel_values))
 print("max_channel_values shape:", max_channel_values.shape)
 print("max_channel_values data type:", max_channel_values.dtype)
 print("max_channel_values device:", max_channel_values.device)
 print("Max values in antelope_tensor:", max_channel_values)
 print()
 print("min_channel_values class:", type(min_channel_values))
 print("min_channel_values shape:", min_channel_values.shape)
 print("min_channel_values data type:", min_channel_values.dtype)
 print("min_channel_values device:", min_channel_values.device)
 print("Min values in antelope_tensor:", min_channel_values)

 OUTPUT:

 max_channel_values class: <class 'torch.Tensor'>
 max_channel_values shape: torch.Size([])
 max_channel_values data type: torch.float32
 max_channel_values device: cpu
 Max values in antelope_tensor: tensor(1.)

 min_channel_values class: <class 'torch.Tensor'>
 min_channel_values shape: torch.Size([])
 min_channel_values data type: torch.float32
 min_channel_values device: cpu
 Min values in antelope_tensor: tensor(0.)

 Actually this min max are stored as the value from
 [0, 255], ToTensor() automatically converts this
 colors to 0 and 1.

 0 means no color
 1 means 100% color

 So it's always a good idea to double-check image tensor values 
 before building a model.
'''


#  MM, RANDN, MANUAL SEED METHODS IN TORCH 

'''

 # Important! Don't change this!
 torch.manual_seed(42)
 torch.cuda.manual_seed(42)

 matrix_1 = torch.randn(3, 4).cuda()

 matrix_2 = torch.randn(4, 2).cuda()
 result = torch.mm(matrix_1, matrix_2)
 print(result)

 1. torch.manual_seed(42)
 
 This function sets the seed for the random number generator (RNG) on the CPU. By setting a seed, you ensure that the random numbers generated (e.g., by torch.randn) are reproducible, making your experiments deterministic and easier to debug.

 Why set a seed? It ensures that every time you run the script, you get the same random values.


 2. torch.cuda.manual_seed(42)

 This sets the seed for the random number generator on the GPU. If you're using CUDA tensors (e.g., via .cuda()), the RNG on the GPU needs to be explicitly seeded to ensure reproducibility for GPU operations.

 Why is this separate? CPU and GPU use different RNGs. Setting both seeds ensures consistent behavior across devices.

 3. torch.randn(3, 4)

 This function generates a tensor of shape (3, 4) filled with random numbers sampled from a standard normal distribution (mean = 0, standard deviation = 1).

 Input (3, 4) specifies that the tensor has 3 rows and 4 columns.
 Output is a 2D tensor with random floating-point values.


 4. torch.mm(matrix_1, matrix_2)
 This performs a matrix multiplication of matrix_1 and matrix_2:

 matrix_1 shape: (3, 4)
 matrix_2 shape: (4, 2)
 The result will be a tensor of shape (3, 2) because the number of columns in matrix_1 (4) matches the number of rows in matrix_2 (4). Matrix multiplication follows the rule:

'''







