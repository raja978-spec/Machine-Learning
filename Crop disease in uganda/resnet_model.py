#         LOADING IMAGES
'''
 import os

 import matplotlib
 import matplotlib.pyplot as plt
 import numpy as np
 import sklearn.model_selection
 import torch
 import torch.nn as nn
 import torch.optim as optim
 import torchinfo
 import torchvision
 from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
 from torch.utils.data import DataLoader
 from torchinfo import summary
 from torchvision import datasets, transforms
 from tqdm import tqdm

 if torch.cuda.is_available():
    device = "cuda"
 elif torch.backends.mps.is_available():
    device = "mps"
 else:
    device = "cpu"

 print(f"Using {device} device.")

 data_dir = os.path.join('data_p2','data_undersampled','train')

 print("Data directory:", data_dir)
 
 class ConvertToRGB(object):
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

 mean = [0.4326, 0.4952, 0.3120]
 std=[0.2179, 0.2214, 0.2091]

 transform_normalized = transforms.Compose(
    [
    ConvertToRGB(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])
 transform_normalized

 dataset = datasets.ImageFolder(root=data_dir, transform=transform_normalized)

 batch_size = 32
 dataset_loader = DataLoader(dataset, batch_size=batch_size)

 print(f"Batch shape: {next(iter(dataset_loader))[0].shape}")
'''

       # PYTORCH RESNET MODEL
'''
 Classifying images is a very common task, many people have already done it. 
 Those people have already spent the time and computing resources to design 
 and train a model. If we can get their architecture and weights, we can use theirs!

 These are called pre-trained models. PyTorch comes with some included. 
 Here we'll load a model called resnet.

 Every model's weight has be initilized with random value, at the time of
 training the weights are updated here those weights are initilized with
 torchvision.models.ResNet50_Weights.DEFAULT

 EX:

 model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)

 This model is very large, and took a long time to train. To get a full summary, 
 we'll need to provide the model with the shape of our data.

 test_batch = next(iter(dataset_loader))[0]
 batch_shape = test_batch.shape

 # Create the model summary
 summary(model, input_size=batch_shape)

 We don't want to train the Pre-trained model by adjusting the weights,
 so we'll set requires_grad = False on all of the weights. In this model's
 weight will not be updated on backpropagation. we use gradient descent to 
 optimize a model's parameters(to reduce loss function). Weights of each layer
 can be iterative by model.parameters(). For 

 for params in model.parameters():
    params.requires_grad = False
 
 This model was trained for a different purpose than we need it for. 
 We can see this by looking at the shape of the output. 
 But our model is very large, so before we run it we'll want to make 
 sure both the model and the test_batch are on the GPU.

 # Move the model to device
 model.to(device)

 # Move our test_batch to device
 test_batch_cuda = test_batch.to(device)

 print("Test batch is running on:", test_batch_cuda.device)

 model_test_out = model(test_batch_cuda)
 model_test_shape = model_test_out.shape

 print("Output shape:", model_test_shape)

 Output shape: torch.Size([32, 1000])
'''

    # MODIFYING THE MODEL
'''
 This model was meant for a task with classes. We only have, 
 so that's not going to work for us. Even if they were the 
 same number of classes, it wouldn't work, since it was trained 
 for a different task.

 But we can replace the final layer with our own network. 
 The rest of the network will still do the image processing, and 
 provide our layer with good inputs. Our network will do the final 
 classification. This process of using most of an already trained model 
 is called transfer learning.

 Which layer is the last one? We can access the list of layers with 
 the named_modules method. It returns a generator, which we can convert
 to a list to get the last element.

 list(model.named_modules())[-1]

 OUTPUT:
 ('fc', Linear(in_features=2048, out_features=1000, bias=True))

 This looks right — it's a linear layer with 
 neurons (and hence outputs). The thing we really wanted to know was its name — fc. 
 Now we can access it with model.fc. We'll need to know how many inputs it takes 
 to be able to replace it. It's a Linear layer, so the number of inputs it 
 takes is recorded in the in_features attribute.

 in_features = model.fc.in_features
 in_features

 OUTPUT:
 2048

 Let's build a network to replace it. It will need to take the same inputs, 
 but produce our outputs.

 We'll make a small network to do our classification. As before, 
 we'll build it with the Sequential container.

 classifier = torch.nn.Sequential()

 classification_layer = torch.nn.Linear(in_features=in_features, out_features=256)
 classifier.append(classification_layer)
 classifier.append(torch.nn.ReLU())
 classifier.append(torch.nn.Dropout())

 output_layer = torch.nn.Linear(in_features=256, out_features=5)
 # Add the layer to our classifier
 classifier.append(output_layer)

 And now we want to do two things: remove the output layer in 
 ResNet that's wrong for us, and add our classifier. 
 We can do both at the same time by replacing fc with 
 our classifier network.

 model.fc = classifier

 # Create the model summary
 summary(model,input_size=batch_shape)

 The last layer will have this

 OUTPUT:

 │    │    └─BatchNorm2d: 3-147           [32, 2048, 7, 7]          (4,096)
 │    │    └─ReLU: 3-148                  [32, 2048, 7, 7]          --
 ├─AdaptiveAvgPool2d: 1-9                 [32, 2048, 1, 1]          --
 ├─Sequential: 1-10                       [32, 5]                   --
 │    └─Linear: 2-17                      [32, 256]                 524,544
 │    └─ReLU: 2-18                        [32, 256]                 --
 │    └─Dropout: 2-19                     [32, 256]                 --
 │    └─Linear: 2-20                      [32, 5]                   1,285
 ==========================================================================

'''