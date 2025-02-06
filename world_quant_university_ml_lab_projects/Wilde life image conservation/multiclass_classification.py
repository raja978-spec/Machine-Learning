#    MULTICLASS CLASSIFICATION

'''

 We'll need to read in our data. Since we'll be using images 
 once again, we'll need to convert them to something our network 
 can understand. To start with, we'll use the same set of 
 transformations we used in the previous notebook.

 These transformations are
 
 * Convert any grayscale images to RGB format with a custom class
 * Resize the image, so that they're all the same size 
   (we chose $224$ x $224$, but other sizes would work as well)
 * Convert the image to a Tensor of pixel values

 This should result in each image becoming a Tensor of size 
 3 x 224 x 224. We'll check this once we read in the data.


 import os
 import sys
 from collections import Counter

 import matplotlib
 import matplotlib.pyplot as plt
 import numpy as np
 import pandas as pd
 import PIL
 import torch
 import torch.nn as nn
 import torch.optim as optim
 import torchvision
 from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
 from torch.utils.data import DataLoader, random_split
 from torchinfo import summary
 from torchvision import datasets, transforms
 from tqdm.notebook import tqdm

 torch.backends.cudnn.deterministic = True

 if torch.cuda.is_available():
    device = "cuda"
 elif torch.backends.mps.is_available():
    device = "mps"
 else:
    device = "cpu"

 print(f"Using {device} device.")

 class ConvertToRGB:
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

 transform = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
 )
'''
 
 #  SETTING IMAGE PATH AND CREATE DATASET TO TRAIN MULTI CLASS MODEL

'''
 data_dir = os.path.join('data_p1','data_multiclass')
 train_dir = os.path.join(data_dir,'train')

 print("Will read data from", train_dir)

 dataset = datasets.ImageFolder(root=train_dir, transform=transform)
'''

 #  SETTING THE DATASET AS BATCH
'''
 
 This prevents PyTorch from trying to load all of the files into 
 memory at once, which would cause our notebook to crash. 
 Instead, it loads just a few (the batch_size), The batch size to 
 work with will depend on our system, but something in the to 
 range is usually fine. We'll pick 32

 batch_size = 32
 dataset_loader = DataLoader(dataset, batch_size=batch_size)

 # Get one batch
 first_batch = next(iter(dataset_loader))

 print(f"Shape of one batch: {first_batch[0].shape}")
 print(f"Shape of labels: {first_batch[1].shape}")
'''

#   FIND MEAN AND STANDARD DEVIATION

'''
 finding the mean and standard deviation of all of the pixels in 
 all of the images will helps for normalization.

 
 def get_mean_std(loader):
    """Computes the mean and standard deviation of image data.

    Input: a `DataLoader` producing tensors of shape [batch_size, channels, pixels_x, pixels_y]
    Output: the mean of each channel as a tensor, the standard deviation of each channel as a tensor
            formatted as a tuple (means[channels], std[channels])"""

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in tqdm(loader, desc="Computing mean and std", leave=False):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std

 
 Run the get_mean_std function. on the training data, and save 
 the means and standard deviations to variables mean and std. 
 There should be a value for each color channel(r,g,b), giving us vectors 
 of length 3.

 mean, std = get_mean_std(dataset_loader)

 print(f"Mean: {mean}")
 print(f"Standard deviation: {std}")

 OUTPUT:

 Mean: tensor([0.4788, 0.4925, 0.4833])
 Standard deviation: tensor([0.2541, 0.2470, 0.2501])

 we need to normalize this each channels
'''

#                   NORMALIZATION

'''

 Activation functions like sigmoids and ReLUs are uses only 0 and 1
 as it's value, that's why we need to normalize the data.

 Normalization is a data preprocessing technique used to adjust the 
 scale or distribution of data so it fits within a specific range or 
 has specific statistical properties. ensuring all input features 
 contribute equally to the learning process.

 To get mean 0 we need to subtract our measured mean from every pixel. 
 To get standard deviation 1 we divide every pixel by the std.

 We can perform these calculations using the Normalize 
 transformation that torchvision gives us.

 transform_norm = transforms.Compose(
    [
        ConvertToRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
 )

 norm_dataset = datasets.ImageFolder(root=train_dir, transform=transform_norm)

 norm_loader = DataLoader(norm_dataset, batch_size=batch_size)

 norm_mean, norm_std = get_mean_std(norm_loader)

 print(f"Mean: {norm_mean}")
 print(f"Standard deviation: {norm_std}")

 OUTPUT:

 Mean: tensor([-2.2570e-07, -7.9987e-07, -1.4907e-07])
 Standard deviation: tensor([1.0000, 1.0000, 1.0000])

 The means may not be exactly zero due to machine precision. 
 But they should be extremely small.

 This sort of "rounding error" is extremely common when 
 working with floating point numbers on a computer. 
 The computer only stores a certain number of digits 
 after the decimal point. This rounding means that math 
 operations sometimes don't get the last few digits right. 
 This becomes very obvious when you subtract two numbers that 
 should be the same. If the last digits in the two numbers 
 are different because of this rounding, you won't get zero, 
 you'll get that last digit leftover.
 
 As an example, 
 1 / 3 - 1 / 5 - 2 / 15 

 Output: -2.7755575615628914e-17

'''

#    DIVIDE THE DATA INTO TRAIN AND TES

'''

 # Important, don't change this!
 g = torch.Generator()
 g.manual_seed(42)

 train_dataset, val_dataset = random_split(norm_dataset,[0.8,0.2])

 length_train = len(train_dataset)
 length_val = len(val_dataset)
 length_dataset = len(norm_dataset)
 percent_train = np.round(100 * length_train / length_dataset, 2)
 percent_val = np.round(100 * length_val / length_dataset, 2)

 print(f"Train data is {percent_train}% of full data")
 print(f"Validation data is {percent_val}% of full data")

 OUTPUT:

 Train data is 80.0% of full data
 Validation data is 20.0% of full data
'''

#     CHECKING IF TRAIN AND TEST HAS COVERED ALL THE CLASSES

'''

 We need to test is the splited data for train and test has covered
 all the classes to train the model efficiently, if train lefts
 some portion of class then model performance will not be good.

 EX: train has the image set of animal, but test set has animal
     and bird, when we do model prediction with test then the
     bird images also will be given to the model which is not
     as part of the trained data.

 To check this we need to create a bar plot of the class distribution

 def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    class_to_index = dataset.dataset.class_to_idx
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()})

 train_class_distributions = class_counts(train_dataset)

 train_class_distributions
 
 # Create a bar plot from train_class_distribution

 train_class_distributions.sort_values().plot(kind='bar')
 # Add axis labels and title
 plt.xlabel("Class Label")
 plt.ylabel("Frequency [count]")
 plt.title("Class Distribution in Training Set")

 # Get the class distribution
 validation_class_distributions = class_counts(val_dataset)

 # Create a bar plot from train_class_distribution
 validation_class_distributions.sort_values().plot(kind='bar')
 # Add axis labels and title
 plt.xlabel("Class Label")
 plt.ylabel("Frequency [count]")
 plt.title("Class Distribution in Validation Set");

 NOTE:
 The two graphs should look similar, though they won't be identical. 
 The random process always produces some differences. If they are _too_ different, 
 you can run your train-validation split again to get a better balance. 
 If you do this, remake the graphs to make sure they're actually better.
'''

#   LOAD NORMALIZED TRAINING AND TEST DATA USING DATA LOADER

'''
 # Important, don't change this!
 g = torch.Generator()
 g.manual_seed(42)


 batch_size = 32

 train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

 # we dont need to shuffle the validation data
 val_loader = DataLoader(val_dataset, batch_size)

 single_batch = next(iter(train_loader))[0]
 print(f"Shape of one batch: {single_batch.shape}")
'''

#       FIND MAX VALUES USING ARGMAX
'''
 sample_confidence = torch.tensor([0.13, 0.01, 0.02, 0.12, 0.10, 0.34, 0.16, 0.12])
 classes = norm_dataset.classes

 class_number = torch.argmax(sample_confidence)
 prediction = classes[class_number]

 print(f"This image is a {prediction}")
'''

#  CREATING CONVOLUTIONAL NEURAL NETWORK
'''
 Conv2d is the 2D convolution layer that is used to extract features from the input image.
 
 This will have four parameters:

 1. in_channels: the number of channels in the input image
 2. out_channels: the number of channels in the output image
 3. kernel_size: the size of the convolutional kernel
 4. padding: the number of pixels to add to the edges of the input image, without padding
             The edges of the input matrix are not included in the output because the kernel 
             cannot fully fit over them. (REFER MULTICAST CLASSIFICATION DOC)

 model_seq = torch.nn.Sequential()
 conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
 model_seq.append(conv1)
'''

#   PASSING THE BATCH TO THE CONVOLUTIONAL LAYER
'''
 
 Before passing the batch we'll check the shape of the batch is that has
 3 channels, because CCN accepts 3 input channels

 test_batch = next(iter(train_loader))[0]
 batch_shape = test_batch.shape

 print(f"Batch shape: {batch_shape}")

 OUTPUT: Batch shape: torch.Size([32, 3, 224, 224])

 first_step_out = model_seq(test_batch)

 first_step_shape = first_step_out.shape

 print(f"Shape after first convolution layer: {first_step_shape}")

 After passing the data shape will be below

 OUTPUT: Shape after first convolution layer: torch.Size([32, 16, 112, 112])

 The 3 input channels are transformed into 16 output channels 
 because the convolutional layer has 16 filters, and each filter 
 processes all input channels to produce one output channel.

 The spatial dimensions (224 → 112) reduce because of the convolution operation,
 depending on kernel size, padding, and stride.
'''

#  APPEND ACTIVATION FUNCTION AND POOLING
'''
 We have three CCN layers appended with activation function and pooling

 model_seq.append(torch.nn.ReLU())
 max_pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
 model_seq.append(max_pool1)

 Now the model look like

 Sequential(
  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
 )

 second_con = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)

 # OUTPUT: output of second_con will be 32 x 112 x 112
 second_poll = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
 model_seq.append(second_con)
 model_seq.append(torch.nn.ReLU())
 model_seq.append(second_poll)
 
 conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
 max_pool3 = torch.nn.MaxPool2d(2)
 model_seq.append(conv3)
 model_seq.append(torch.nn.ReLU())
 model_seq.append(max_pool3)

 # Now we passing the data to the model

 third_set_out = model_seq(test_batch)
 third_set_shape = third_set_out.shape

 print(f"Shape after third max pool: {third_set_shape}")

 We should now have a 32 x 64 x 28 x 28 We could keep adding more of these sets of layers, 
 but this should be plenty. Now we need to move toward getting our final 
 classes.
'''

#    FLATTING 
'''

 After adding the flatting we should again pass the
 test batch to get the flatted data. Flatten will be 
 done by multiplying 3D values from the previous layer.

 On previous layer we have 32 x 64 x 28 x 28, so it takes
 64 x 28 x 28

 model_seq.append(torch.nn.Flatten())

 OUTPUT:

 Sequential(
  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): ReLU()
  (5): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU()
  (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
   (9): Flatten(start_dim=1, end_dim=-1)
 )

 flat_out = model_seq(test_batch)
 flat_shape = flat_out.shape

 print(f"Shape after flattening: {flat_shape}")

 OUTPUT: torch.Size(32, 50176)
'''

#    MOVE DATA TO DENSE LAYER AND ADDING OUTPUT LAYER

'''
 At this point we have a flat input, and can build a normal set of dense layers.
 You can think of the convolution/max pool layers as having done the image processing. 
 Now we need to do the actual classification. It turns out that dense layers are good 
 at that task.

 We could add a single layer and just go straight to our output 
 classes. But we'll get better performance by adding a few dense layers, 
 Linear in PyTorch's terminology, first. For these layers, we need to tell
 it the size of the input, and how many neurons we want in the layer. 
 
 Since the input is our previous layer, we tell it that size. We'll add a 500 layer of 
 neurons.

 linear1 = torch.nn.Linear(in_features=50176, out_features=500)

 model_seq.append(linear1)
 model_seq.append(torch.nn.ReLU())

 OUTPUT:

 Sequential(
  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): ReLU()
  (5): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU()
  (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (9): Flatten(start_dim=1, end_dim=-1)
  (10): Linear(in_features=50176, out_features=500, bias=True)
 )

 linear_out = model_seq(test_batch)
 linear_shape = linear_out.shape

 print(f"Shape after linear layer: {linear_shape}") 

 And now we should be getting an output shape from the 500
 neurons.

 OUTPUT: Shape after linear layer: torch.Size([32, 500])

 output_layer = torch.nn.Linear(in_features=500, out_features=8)

 model_seq.append(output_layer)

 OUTPUT:

 Sequential(
  (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): ReLU()
  (5): MaxPool2d(kernel_size=(2, 2), stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU()
  (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (9): Flatten(start_dim=1, end_dim=-1)
  (10): Linear(in_features=50176, out_features=500, bias=True)
  (11): ReLU()
  (12): Linear(in_features=500, out_features=8, bias=True)
 )

 model_seq(test_batch).shape

 OUTPUT: torch.Size([32, 8]) 
'''

#  TRAINING THE MODEL
'''
 # Important! Don't change this
 torch.manual_seed(42)
 torch.cuda.manual_seed(42)

 model = torch.nn.Sequential()

 conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
 max_pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
 model.append(conv1)
 model.append(torch.nn.ReLU())
 model.append(max_pool1)

 conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
 max_pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
 model.append(conv2)
 model.append(torch.nn.ReLU())
 model.append(max_pool2)

 conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
 max_pool3 = torch.nn.MaxPool2d(2)
 model.append(conv3)
 model.append(torch.nn.ReLU())
 model.append(max_pool3)

 model.append(torch.nn.Flatten())
 model.append(torch.nn.Dropout()) # each forward pass during training, a randomly selected subset of neurons is temporarily 
                                  # ignored which might cause overfitting, that will be fixed with dropout 

 linear1 = torch.nn.Linear(in_features=50176, out_features=500)
 model.append(linear1)
 model.append(torch.nn.ReLU())
 model.append(torch.nn.Dropout())

 output_layer = torch.nn.Linear(500, 8)
 model.append(output_layer)

'''

#   VIEW MODEL PREDICTION USING SKLEARN CONFUSION MATRIX
'''
 A confusion matrix is a performance evaluation tool from sklearn used to measure the 
 accuracy of a classification model by comparing the predicted labels to 
 the actual labels. 
 
 # Using the module from binary_classification_with_pytorch.py

 from training import predict, train

 from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
 from torch.utils.data import DataLoader, random_split
 from torchinfo import summary


 loss_fn = nn.CrossEntropyLoss()
 optimizer = optim.Adam(model.parameters(), lr=0.001)
 model.to(device)

 train(model,optimizer,loss_fn,train_loader,val_loader,epochs=20,device='cpu')
  
 # OR

 model = torch.load("model/trained_model.pth", weights_only=False)


 probabilities = predict(model,val_loader,device)
 predictions = torch.argmax(probabilities, dim=1)

 targets = []
 
 # The value returned by prediction will not human readable, so we need to convert it to human readable
 # by using confusion matrix

 for _, labels in tqdm(val_loader):
    targets.extend(labels.tolist())

 cm = confusion_matrix(targets, predictions.cpu())

 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

 disp.plot(cmap=plt.cm.Blues, xticks_rotation="vertical")
 plt.show();

 OUTPUT: Refer CNN.doc
'''


#                  STRIDE
'''
 1. Stride in Convolution Layers

 The stride defines how far the convolutional kernel moves after each 
 step horizontally and vertically.
 
 A stride of 1 (default) means the kernel slides one pixel at a time, 
 resulting in overlapping regions and output dimensions that are close to the 
 input dimensions (depending on padding).
 
 A stride > 1 means the kernel skips some pixels, which:

 Reduces the spatial dimensions of the output.
 Extracts features over a broader area.
 Makes the operation faster but with less spatial detail.

'''