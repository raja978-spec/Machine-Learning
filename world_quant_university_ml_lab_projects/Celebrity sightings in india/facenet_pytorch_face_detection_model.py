#               INTRO
'''
 Before going to recognize the face we have to detect
 faces, for that we uses facenet_pytorch pre-trained model.

 Objectives:

Initialize a pre-trained MTCNN model from facenet_pytorch
Detect faces in an image using MTCNN model
Display the resulting bounding boxes of faces detected by the model
Crop out detected faces for further analysis
Determine facial landmarks such as eyes, nose, and mouth using the MTCNN model
Select a subset of images for face recognition tasks in the next lesson
'''

#           INITILZED THE MODEL
'''
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision.utils import make_grid

VERSION OUTPUT:
Platform: linux
Python version: 3.11.0 (main, Nov 15 2022, 20:12:54) [GCC 10.2.1 20210110]
---
torch version :  2.2.2+cu121
torchvision version :  0.17.2+cu121
PIL version :  10.2.0

# We'll perform face detection using a MTCNN network from 
# facenet_pytorch library. This model is able to simultaneously 
# propose bounding boxes of faces, determine detection probabilities, 
# and detect facial landmarks like eyes, nose and mouth.

# Let's start by initializing the model. Here are a couple of arguments we get to set:

# device: The device on which to run the model.

# keep_all: A boolean determining if all detected faces are returned or not.

# min_face_size: Minimum face size (in pixels) to search for in the image.

# post_process: A boolean determining if we want image standardization of
# detected faces. This is advised before proceeding with face recognition models, 
# but if we want face images that are returned to us to look normal to the human eye, 
# we can set post_process=False.

# Task 4.3.1: Initialize a MTCNN model. Make sure to use a GPU, 
# keep all detected faces and set minimum face size to search for to be 60.

mtcnn = MTCNN(device=device, keep_all=True, min_face_size=60, post_process=False)

print(mtcnn)

OUTPUT:
MTCNN(
  (pnet): PNet(
    (conv1): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))
    (prelu1): PReLU(num_parameters=10)
    (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
    (conv2): Conv2d(10, 16, kernel_size=(3, 3), stride=(1, 1))
    (prelu2): PReLU(num_parameters=16)
    (conv3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
    (prelu3): PReLU(num_parameters=32)
    (conv4_1): Conv2d(32, 2, kernel_size=(1, 1), stride=(1, 1))
    (softmax4_1): Softmax(dim=1)
    (conv4_2): Conv2d(32, 4, kernel_size=(1, 1), stride=(1, 1))
  )
  (rnet): RNet(
    (conv1): Conv2d(3, 28, kernel_size=(3, 3), stride=(1, 1))
    (prelu1): PReLU(num_parameters=28)
    (pool1): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (conv2): Conv2d(28, 48, kernel_size=(3, 3), stride=(1, 1))
    (prelu2): PReLU(num_parameters=48)
    (pool2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
    (conv3): Conv2d(48, 64, kernel_size=(2, 2), stride=(1, 1))
    (prelu3): PReLU(num_parameters=64)
    (dense4): Linear(in_features=576, out_features=128, bias=True)
    (prelu4):
'''

#         SET PATH FOR FRAME OR IMAGES

'''
curr_work_dir = Path.cwd()

print(curr_work_dir)

extracted_frames_dir = curr_work_dir / 'project4' / 'data' / 'extracted_frames'

print(extracted_frames_dir)

# Bounding Boxes of Detected Faces

# If we want to detect faces and obtain their bounding boxes, 
# we need to use the detect method on the MTCNN model and pass in the 
# sample image. This returns both the bounding boxes of detected faces 
# as well as the predicted probability that the object in a given bounding 
# box is indeed a face.

boxes, probs = mtcnn.detect(sample_image)

print("boxes type:", type(boxes))
print("probs type:", type(probs))

OUTPUT:
boxes type: <class 'numpy.ndarray'>
probs type: <class 'numpy.ndarray'>

# Great! We now have two arrays. Array boxes contains the bounding boxes 
# of the detected faces and probs contains the probabilities.

# Let's look at the boxes array first.

print(boxes)
print(boxes.shape)
OUTPUT:
[[211.4107208251953 46.25676345825195 285.15838623046875
  146.4738311767578]
 [346.31689453125 40.236629486083984 392.3767395019531 99.2223892211914]
 [458.0262756347656 55.83146667480469 502.13848876953125
  112.0572280883789]]
(3, 4)

number_of_detected_faces = len(boxes)

print(number_of_detected_faces)

OUTPUT: 3

# So we have three faces in the given image

# Using probs, determine for how many of the faces detected 
# did the model predict with at least 99% probability that it's a face.

num_faces = probs[probs > 0.99]

print(num_faces)

OUTPUT:
[0.9999923706054688 0.9997478127479553 0.9999895095825195]

#Now let's plot the bounding boxes together with the sample image.
#Fill in the missing code below to iterate over all of the 
# bounding boxes and plot them on top of the sample image.

fig, ax = plt.subplots()
ax.imshow(sample_image)

for box in boxes:
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    ax.add_patch(rect)
plt.axis("off");
'''

#  Extracting Facial Landmarks
'''
# MTCNN not only detects faces but can also mark facial landmarks such as 
# eyes, nose, and mouth in each detected face.

# The way to obtain the facial landmarks together with bounding boxes 
# and probabilities is to again use the detect method on the MTCNN model. 
# But this time together with the sample image, we need to pass in landmarks=True.

boxes, probs, landmarks = mtcnn.detect(sample_image,landmarks=True)

print("boxes type:", type(boxes))
print("probs type:", type(probs))
print("landmarks type:", type(landmarks))

OUTPUT:

boxes type: <class 'numpy.ndarray'>
probs type: <class 'numpy.ndarray'>
landmarks type: <class 'numpy.ndarray'>

# The facial landmarks detected by the model on each face are:

# left eye,
# right eye,
# nose,
# left mouth corner,
# right mouth corner.
# Let's make sure that the shape of the landmarks array 
# matches what we'd expect given that six faces were detected.
print(landmarks.shape)

OUTPUT:
(3, 5, 2)
Great! We have 3 faces detected and on each face, we 
have 5 facial landmarks and 2 coordinates(x,y) locating each landmark.

# Fill in the missing code to plot the bounding boxes as well as the facial 
# landmarks on top of the sample image. We recommend using zip on boxes and 
# landmarks in the for loop that you need to fill in.

fig, ax = plt.subplots()
ax.imshow(sample_image)

for box, landmark in zip(boxes, landmarks):
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    ax.add_patch(rect)
    for point in landmark:
        ax.plot(point[0], point[1], marker="o", color="red")
plt.axis("off");

REFER wha is mutlit-taks convolution neural network.docx for output

'''

#            Cropping out Detected Faces
'''
# If we wanted to proceed with further face analysis like for 
# example perform face recognition, it's a good idea to crop out 
# the detected faces. That way further analysis can focus only on 
# the relevant parts of the image.

# So let's learn how we can crop out the detected faces!

# In order to get the PyTorch tensors of the detected faces 
# instead of the bounding boxes, we need to call the MTCNN object 
# directly and just pass in the image we're working with.

faces = mtcnn(sample_image)

print(faces.shape)
OUTPUT: torch.Size([3, 3, 160, 160])

# Looks like this returned three small images, each with 3 color 
# channels and 160 width and 160 height. Let's plot these 3 images!

#  Create a grid of these three images by using make_grid from 
#  torchvision.utils and passing in faces. Use nrow=3 so we'll 
#  have all 3 images in one row.

Grid = make_grid(faces, nrow=3)

print(Grid.shape)

plt.imshow(Grid.permute(1, 2, 0).int())
plt.axis("off");

'''

# Prepare a Subset of Images for Face recognition task

'''
images_dir = curr_work_dir / "project4" / "data" / "images"
images_dir.mkdir(exist_ok=True)

# Make a subdirectory in the images directory and call it mary_kom. 
# Again make sure you do it such that no error is raised even if the 
# directory already exists.

mary_kom_dir = images_dir / 'mary_kom'

# Now Create `mary_kom` directory
mary_kom_dir.mkdir(exist_ok=True)

# Good job! The directory you just created will be the 
# directory into which we'll put the selected images.

# Let's make a list of frames that we want to use.

mary_kom_imgs = [
    "frame_80.jpg",
    "frame_115.jpg",
    "frame_120.jpg",
    "frame_125.jpg",
    "frame_135.jpg",
]

# Iterate over mary_kom_imgs list of image filenames and 
# create a list of absolute paths to each image using pathlib 
# syntax. Remember that the images are in the extracted_frames 
# directory.

mary_kom_img_paths = [extracted_frame_dir / i for i in mary_kom_imgs]

print("Number of images we'll use:", len(mary_kom_img_paths))

OUTPUT: Number of images we'll use: 5

# Before we copy these images over to mary_kom directory, 
# let's just look at them.

fig, axs = plt.subplots(1, 5, figsize=(10, 8))

for i, ax in enumerate(axs):
    ax.imshow(Image.open(mary_kom_img_paths[i]))
    ax.axis("off")

# Iterate over mary_kom_img_paths in order to copy these 
# selected images into mary_kom directory.

for image_path in mary_kom_img_paths:
    shutil.copy(image_path, mary_kom_dir)

print("Number of files in mary_kom directory:", len(list(mary_kom_dir.iterdir())))

OUPUT:
Number of files in mary_kom directory: 5

# We'll also get some images of the interviewer, so we'll have more 
# than one face we can potentially identify. We'll call that directory 
# ranveer, since that's the interviewer's first name.

# So we can do the same above code to do this
'''
