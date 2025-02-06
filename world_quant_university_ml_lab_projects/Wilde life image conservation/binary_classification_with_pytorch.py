#     BINARY CLASSIFICATION IN PYTORCH
'''

 All the images will not have same mode and type so we have to convert
 them to RGB mode

 Following are the steps will should do before classification

 We'll use transforms.Compose to create a "pipeline" of 
 preprocessing sets. The steps defined (in order) will be:

  1. Convert image (if needed) to RGB
  2. Resize the image
  3. Convert the images to PyTorch tensors
  
  Afterwards, we'll load the data and apply the transformation pipeline.

 For this we will use the module transforms.Compose() method from the
 following module 

 from torchvision import datasets, transforms

 class ConvertToRGB:

    # call is one the dunder method in python
    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

 # Define transformation to apply to the images
 transform = transforms.Compose(
    [
        ConvertToRGB(),  # Convert images to RGB format if not already
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor()

    ]
 )

 print(type(transform))
 print(transform)

 OUTPUT:

 <class 'torchvision.transforms.transforms.Compose'>
 Compose(
     (ConvertToRGB): ConvertToRGB()
     (Resize): Resize(size=(224, 224), interpolation=PIL.Image.BILINEAR)
     (ToTensor): ToTensor()
 )

'''

#   BINARY VALUES (METHODS IN dataset module IMAGE_FOLDER, CLASSES, 
#   IMGS)
# PROVE THAT BINARY VALUES
'''
 Now that transformed data we have to create dataset with the
 help of the following module

 from torchvision import datasets, transforms

 # Load the dataset using `ImageFolder`
 dataset = datasets.ImageFolder(root=train_dir, transform=transform)
 print(dataset)

 OUTPUT:

 Dataset ImageFolder
    Number of datapoints: 3191
    Root location: data_p1/data_binary/train
    StandardTransform
 Transform: Compose(
               <__main__.ConvertToRGB object at 0x7d2372a97fd0>
               Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=True)
               ToTensor()
           )

 The dataset object has the attribute .classes which returns a 
 list of distinct classes.

             CLASSES

 print(dataset.classes)

 OUTPUT:
 ['blank', 'hog']

                  IMGS

 The .imgs attribute is a list of tuples of path to image and label. 
 The label value will be a number. Since we're doing binary 
 classification, there will only be two distinct numbers. 
 Those numbers are 0 and 1.

 The labels are usually assigned based on the directory structure of your dataset. For instance:

  data_binary/
  ├── train/
  │   ├── blank/      # Class 0
  │   ├── filled/     # Class 1

 In this structure, images under the blank/ directory are assigned the label 0, 
 and images under the filled/ directory are assigned the label 1.
 The mapping between directories and labels is done when the dataset is 
 loaded using tools like PyTorch's ImageFolder. 
 
 This tool:
 Automatically assigns labels based on the alphabetical order of the folder 
 names (e.g., blank comes before filled). Stores the image paths and corresponding 
 labels as tuples.

 EX:

 im = dataset.imgs
 print(im[0])

 OUTPUT:
 ('data_p1/data_binary/train/blank/ZJ000013.jpg', 0)

 Here 0 is the label

'''

#     SUPERVISED LEARNING WITH RANDOM_SPLIT()

'''

 random_split splits the dataset as training and test data in some
 ratio

 EX:

 from torch.utils.data import DataLoader, random_split

 train_data, test_data = random_split(dataset, [0.8, 0.2])

 print(len(train_data))

 OUTPUT: 2553
'''

#       TQDM, COUNTERS, CREATING A BAR CHART FROM PD SERIES
'''
 * tqdm is a progress bar that shows the progress of a loop
 * Counter is used to count the number of elements in a list

 With the help of this we will visualize the number of two classified
 images (blank and hog) in a bar chart


 EX:

 def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    class_to_index = dataset.dataset.class_to_idx
    print('c',c)
    print('cti', class_to_index)
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()})

 val_counts = class_counts(val_dataset)
 print(val_counts)

 OUTPUT:

 c Counter({0: 2553, 1: 319})
 cti {'blank': 0, 'hog': 1}
 0    2553
 1    319
 dtype: int64


 # Create a bar chart from the function output
 val_counts.sort_values().plot(kind="bar");


      MORE DETAILS ON COUNTER FUNCTION

 from collections import Counter

 # Example 1: Count elements in a list
 fruits = ['apple', 'banana', 'orange', 'apple', 'banana', 'apple']
 fruit_counter = Counter(fruits)

 print(fruit_counter)
 # Output: Counter({'apple': 3, 'banana': 2, 'orange': 1})

 # Example 2: Count characters in a string
 text = "hello world"
 char_counter = Counter(text)

 print(char_counter)
 # Output: Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, 'w': 1, 'r': 1, 'd': 1})

 # Example 3: Using Counter with most_common
 print(fruit_counter.most_common(2))  
 # Output: [('apple', 3), ('banana', 2)] -> The 2 most common elements

 # Example 4: Update counts with another iterable
 more_fruits = ['apple', 'grape', 'banana']
 fruit_counter.update(more_fruits)

 print(fruit_counter)
 # Output: Counter({'apple': 4, 'banana': 3, 'orange': 1, 'grape': 1})

 # Example 5: Subtract counts
 fruit_counter.subtract({'apple': 2, 'banana': 1})
 print(fruit_counter)
 # Output: Counter({'apple': 2, 'banana': 2, 'orange': 1, 'grape': 1})
 
'''

#           DATALOADER
'''
 When we work with pytorch we have to load the data in batches. Before going to 
 build and train the model we have to load the data.

 EX:

 from torch.utils.data import DataLoader, random_split

 # Important, don't change this!
 g = torch.Generator()
 g.manual_seed(42)

 batch_size = 32
 train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

 val_loader = DataLoader(val_dataset, batch_size)

 print(type(val_loader))

 OUTPUT:

 <class 'torch.utils.data.dataloader.DataLoader'>

'''

#   CREATING ITERATOR FROM DATALOADER WITH ITER AND NEXT

'''

 * iter create iterator from dataloader
 * next will return the data one by one as
   32 batch.

 data_iter = iter(train_loader)
 images, labels = next(data_iter)

 # This gives you [batch_size, channels, height, width] for images
 image_shape = images.shape
 print("Shape of batch of images", image_shape)

 # This gives you [batch_size] for labels
 label_shape = labels.shape
 print("Shape of batch of labels:", label_shape)

 OUTPUT:

 Shape of batch of images torch.Size([32, 3, 224, 224])
 Shape of batch of labels: torch.Size([32])

 The create tensor using iter have 4 dimensions, 32 images
 that 32 images have 3 channels (red, green, blue) and 224
 rows and 224 columns.
'''


#     FLATTENING THE TENSORS

'''
 * To build nerual network we have to flatten the tensors from 4 dimensions
   to 1 dimensions for that we use this method

 * That will be done by multiplying the last two dimensions with color
   3 * 224 * 224

 EX:

 flatten = nn.Flatten()
 tensor_flatten = flatten(images)

 # Print the shape of the flattened tensor
 print(tensor_flatten.shape)

 OUTPUT:

 torch.Size([32, 150528])

'''

#    CREATING NEURAL NETWORK WITH NN SEQUENTIAL

'''
 * We'll create a neural network with one input layer
   two hidden layer and one output layer

 * Sequential is used to create a sequence of layers

 * Linear is used to create a fully connected layer
 
 * nn.Linear is a standard dense, or fully-connected, layer. 
   It takes two arguments, the number of inputs coming into this 
   layer and the number of outputs produced by this layer.

 * nn.ReLU performs the rectified linear unit activation. 
   Activation functions are necessary for neural networks to work, 
   and ReLU is a popular choice.

 EX:

 # Image size from our transformer
 height = 224
 width = 224

 model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * height * width, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
 )

 print("model type:", type(model))
 print("model structure:")
 print(model)

 OUTPUT:

 model type: <class 'torch.nn.modules.container.Sequential'>
 model structure:
 Sequential(
   (0): Flatten(start_dim=1, end_dim=-1)                        # INPUT LAYER
   (1): Linear(in_features=150528, out_features=512, bias=True) # FIRST HIDDEN LAYER
   (2): ReLU()                                                  # FIRST ACTIVATION FUNCTION
   (3): Linear(in_features=512, out_features=128, bias=True)    # SECOND HIDDEN LAYER
   (4): ReLU()                                                  # SECOND ACTIVATION FUNCTION
 )

 Sequential can be appended like array, now well
 add output layer

 EX:

 output_layer = nn.Linear(128, 2)
 model.append(output_layer)

 print(model)

 OUTPUT:

 Sequential(
   (0): Flatten(start_dim=1, end_dim=-1)                        # INPUT LAYER
   (1): Linear(in_features=150528, out_features=512, bias=True) # FIRST HIDDEN LAYER
   (2): ReLU()                                                  # FIRST ACTIVATION FUNCTION
   (3): Linear(in_features=512, out_features=128, bias=True)    # SECOND HIDDEN LAYER
   (4): ReLU()                                                  # SECOND ACTIVATION FUNCTION
   (5): Linear(in_features=128, out_features=2, bias=True)      # OUTPUT LAYER
 )

 NOTE: out_features of one layer will becomes in_features
       of next layer
'''


#      TRAINING OUR MODEL USING CrossEntropyLoss, ADAM OPTIMIZER

'''

 * First step of this process is to define a loss function
   (cost function), which helps the measure how our model
   preforms wrongly on a training dataset.

 * With the use of loss function only model can be trained.

 * We'll use nn.CrossEntropyLoss() loss function from pytroch

 * We also need an optimizer. This will adjust the model's parameters 
   to try to minimize the loss function. Reducing the loss function 
   is the goal of training.

 *  The optim.Adam class is initialized with the model parameters through model.parameters. 
  
 * An optional argument is the learning rate lr. This controls how large the step sizes 
   are in gradient descent

 When training a model, the data is typically too large to feed into the 
 model all at once. The dataset is split into batches (or mini-batches), 
 and the model is trained on these smaller chunks of data.One epoch is completed when the 
 model has seen all batches of the training data exactly once.


 * The below code is used to calculate the loss of training current
   batch dataset/tensor.

 
 EX:

 import torch.nn as nn

 model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * height * width, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
 )

 output_layer = nn.Layer(128,2)
 model.append(output_layer)
 
 loss_fn = nn.CrossEntropyLoss()
 optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

 height = 224
 width = 224

 
 def train_epoch(model, optimizer, loss_fn, data_loader, device="cpu"):
    # We'll report the loss function's average value at the end of the epoch.
    training_loss = 0.0

    # The train method simply sets the model in training mode. No training
    # has happened.
    model.train()

    # We iterate over all batches in the training set to complete one epoch
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        # Sets the gradients to zero. We need to do this every time.
        optimizer.zero_grad()

        # Unpack images (X) and labels (y) from the batch and add those
        # tensors to the specified device.
        inputs = inputs.to(device)
        targets = targets.to(device)

        # We make a forward pass through the network and obtain the logits.
        # With the logits, we can calculate our loss.
        output = model(inputs)
        loss = loss_fn(output, targets)

        # After calculating our loss, we calculate the numerical value of
        # the derivative of our loss function with respect to all the
        # trainable model weights. Once we have the gradients calculated,
        # we let the optimizer take a "step", in other words, update or
        # adjust the model weights.
        loss.backward()
        optimizer.step()

        # We increment the training loss for the current batch
        training_loss += loss.data.item() * inputs.size(0)

    # We calculate the training loss over the completed epoch
    return training_loss / len(data_loader.dataset)

 loss_value = train_epoch(model, optimizer, loss_fn, train_loader, device)
 print(f"The average loss during the training epoch was {loss_value:.2f}.")

 OUTPUT:
 The average loss during the training epoch was 0.74.

 A nice feature of neural networks is that when we do another training 
 run, it doesn't start training from scratch. In other words, training 
 resumes from the current weights and continues to adjust the weights to 
 improve the model's performance.

 EX:

 loss_value = train_epoch(model, optimizer, loss_fn, train_loader, device)
 print(f"The average loss during the training epoch was {loss_value:.2f}.")

 OUTPUT:
 The average loss during the training epoch was 0.52.
'''

#          PREDICT MODEL uSING SOFTMAX FUNCTION

'''
 
 The loss function we're using in this problem, the cross-entropy, 
 is difficult for humans to interpret. We know that lower is better, 
 but is that number good or bad? To help us humans judge, we'll 
 calculate the accuracy of the model, the fraction of predictions 
 it gets right. To do that, we need the model to make predictions.

 The following function will make a prediction for each row of data 
 in data_loader, using model. As with the train_epoch function 
 above, it's a bit daunting. The comments explain what 
 each section does.

 def predict(model, data_loader, device="cpu"):
    # This tensor will store all of the predictions.
    all_probs = torch.tensor([]).to(device)

    # We set the model to evaluation mode. This mode is the opposite of
    # train mode we set in the train_epoch function.
    model.eval()

    # Since we're not training, we don't need any gradient calculations.
    # This tells PyTorch not to calculate any gradients, which speeds up
    # some calculations.
    with torch.no_grad():

        # Again, we iterate over the batches in the data loader and feed
        # them into the model for the forward pass.
        for inputs, targets in tqdm(data_loader, desc="Predicting", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)

            # The model produces the logits.  This softmax function turns the
            # logits into probabilities.  These probabilities are concatenated
            # into the `all_probs` tensor.
            probs = F.softmax(output, dim=1)
            all_probs = torch.cat((all_probs, probs), dim=0)

    return all_probs

 probabilities_train = predict(model, train_loader, device)
 print(probabilities_train.shape)

 OUTPUT:

 torch.Size([2553, 2])

 2553 - number of images in the 32 batch
 2 - number of classes or probability values
'''

#          WHAT PREDICTION WILL ACTUALLY RETURN

'''
 print(probabilities_train[0])

 OUTPUT:
 tensor([0.1377, 0.8623], device='cuda:0')


 0.1377 - probability of image belonging to class 0
 0.8623 - probability of image belonging to class 1

 In this we have two classes which is blank and hog. 
 Because these are the only two possibilities, they should 
 add up to one.

 print(probabilities_train[0].sum())

 OUTPUT:
 tensor(1., device='cuda:0')
'''

#       FIND MAXIMUM PROBABILITY USING ARGMAX

'''

 The argmax function in PyTorch is used to find the index of the maximum 
 value along a specified dimension of a tensor.

 SYNTAX:

 torch.argmax(input, dim=None, keepdim=False)

 input: The input tensor.
 dim (optional): The dimension along which to compute the argmax. 
                 If None, the function returns the index of the maximum value 
                in the flattened tensor.
 keepdim (optional): If True, retains the reduced dimension(s) with size 1. 
                     Default is False.

 RETURNS: 

 A tensor containing the indices of the maximum values along the 
 specified dimension. If dim is None, it returns a single value as 
 the index in the flattened tensor.

 EX:

 import torch

 x = torch.tensor([1, 3, 2, 7, 4])
 index = torch.argmax(x)
 print(index)  # Output: 3 (index of the maximum value 7)

 probabilities_train is a 2D tensor with shape (N, 2), 
 where N is the number of rows (data points), and 2 is the number 
 of columns (typically representing two classes in a binary 
 classification problem).

 predictions_train = torch.argmax(probabilities_train, dim=1)

 dim=1
 When dim=1 is specified, torch.argmax looks for the maximum value 
 along the second dimension (columns) for each row. This results in 
 a 1D tensor containing the indices of the maximum values for 
 each row.

 print(f"Predictions shape: {predictions_train.shape}")
 print(f"First 10 predictions: {predictions_train[:10]}")

 OUTPUT:
 
 Predictions shape: torch.Size([2553])
 First 10 predictions: tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], device='cuda:0')
'''

#   COMPARE PREDICTIONS TO TARGETS USING TORCH.EQ, TORCH.SUM, AND TORCH.CAT

'''
 # Taking all the actual labels from the data loader and 
 # concatenating them into a single tensor.
 targets_train = torch.cat([labels for _, labels in train_loader]).to(device)

 # Comparing the predictions to the actual labels.
 is_correct_train = torch.eq(predictions_train, targets_train)

 # Counting the number of correct predictions.
 total_correct_train = torch.sum(is_correct_train).item()

 # Calculating the accuracy. train_loader.dataset is the no of
 # images in the dataset
 accuracy_train = total_correct_train / len(train_loader.dataset)

 print(f"Accuracy on the training data: {accuracy_train}")


 There was a lot of work we did to get the accuracy. It's more efficient 
 to do this calculation batch by batch. It's best we wrap that code 
 into a function. The score function, below, does just that.
 
 It's very similar to the prediction function, but instead of gathering 
 all the predictions, it just calculates the number of correct predictions in 
 each batch. It also calculates the loss function over the whole batch. 

 We'll use this to compare the loss function value on the training set 
 and the validation set, to understand if there's any overfitting.

 def score(model, data_loader, loss_fn, device="cpu"):
    # Initialize the total loss (cross entropy) and the number of correct
    # predictions. We'll increment these values as we loop through the
    # data.
    total_loss = 0
    total_correct = 0

    # We set the model to evaluation mode. This mode is the opposite of
    # train mode we set in the train_epoch function.
    model.eval()

    # Since we're not training, we don't need any gradient calculations.
    # This tells PyTorch not to calculate any gradients, which speeds up
    # some calculations.
    with torch.no_grad():
        # We iterate over the batches in the data loader and feed
        # them into the model for the forward pass.
        for inputs, targets in tqdm(data_loader, desc="Scoring", leave=False):
            inputs = inputs.to(device)
            output = model(inputs)

            # Calculating the loss function for this batch
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            total_loss += loss.data.item() * inputs.size(0)

            # Calculating the correct predictions for this batch
            correct = torch.eq(torch.argmax(output, dim=1), targets)
            total_correct += torch.sum(correct).item()

    return total_loss / len(data_loader.dataset), total_correct / len(
        data_loader.dataset
    )

  loss_train, accuracy_train = score(model, train_loader, loss_fn, device)
  print(f"Training accuracy from score function: {accuracy_train}")

  OUTPUT:

  Training accuracy from score function: 0.81786133960047

'''


#        SAVING MODEL

'''
 We can either save the entire model using save or just the parameters using 
 state_dict. Using the latter is normally preferable, as it allows you to reuse 
 parameters even if the model's structure changes (or apply parameters from one
 model to another). Saving the model avoids having to retrain the model the next 
 time we want to use it.

 torch.save(model, os.path.join("model", "shallownet"))
 model = torch.load(os.path.join("model", "shallownet"))

 torch.save(model.state_dict(), os.path.join("model", "shallownet"))
 new_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * height * width, 512),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
 )
 model_state_dict = torch.load(os.path.join("model", "shallownet"))
 new_model.load_state_dict(model_state_dict)
'''

