'''
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import PIL
import torch
import torchvision
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---")
print("PIL version : ", PIL.__version__)
print("torch version : ", torch.__version__)
print("torchvision version : ", torchvision.__version__)

OUTPUT:
Platform: linux
Python version: 3.11.0 (main, Nov 15 2022, 20:12:54) [GCC 10.2.1 20210110]
---
PIL version :  10.2.0
torch version :  2.2.2+cu121
torchvision version :  0.17.2+cu121

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device} device.")
'''

# Initializing MTCNN and Inception-ResNet V1¶

'''
# Let's start things by initializing MTCNN as we've done 
# in the past lesson. We'll want an MTCNN model that just detects one face.

# Create a MTCNN model. Make sure to set the image size as 240, 
# keep_all to False, andmin_face_size to 40.

mtcnn0 = MTCNN(image_size=240, keep_all=False, min_face_size=40)
print(mtcnn0)

# We'll also initialize the Inception-ResNet V1 model. This model 
# is used for facial recognition. Particularly, we'll be using a model 
# that has been pre-trained on the VGGFace2 dataset. This is a massive 
# dataset of over 3 million images and over 9000 identities.

resnet = InceptionResnetV1(pretrained="vggface2").eval()

# We need to prepare our data to make it easy to use the
# Inception-ResNet V1 model. We'll ultimately create a DataLoader object.
# Create a path object for the path project4/data/images/. Use the Path class from pathlib.

images_folder = Path('project4','data','images')

print(f"Path to images: {images_folder}")

The next step is to create an ImageFolder object for the 
path we just created. ImageFolder is provided to us by 
torchvision and makes it easier when working with PyTorch.

dataset = datasets.ImageFolder(images_folder)

print(dataset)

OUTPUT:
Dataset ImageFolder
    Number of datapoints: 10
    Root location: project4/data/images

# With the ImageFolder object, each subdirectory of the input 
# path is considered a separate class. The class label is just 
# the name of the subdirectory. In the previous lesson, we pulled 
# out frames for Mary Kom and Ranveer, the interviewer.

# Print out all subdirectories in images_folder. 
# You should use the method iterdir of the path object.

for subdirectory in images_folder.iterdir():
    print(subdirectory)

OUTPUT:
project4/data/images/mary_kom
project4/data/images/ranveer

# With a ImageFolder object, the .class_to_idx is a mapping 
# between class label to class integer.

# However, we'd like to create the reverse mapping. In other words, 
# integer to class label.

idx_to_class = {j: i for i,j in dataset.class_to_idx.items()}

print(idx_to_class)

OUTPUT:
{0: 'mary_kom', 1: 'ranveer'}

# The next step is to create a DataLoader object with our ImageFolder. 
# These objects are iterables that work well with PyTorch. 
# Remember how we worked with DataLoader in past projects. 
# One thing we'll do differently here is provide the DataLoader a 
# collate function. In our case, the collate function is returning the 
# first element of a tuple, which is the image object.

# Construct a DataLoader object with the previously created dataset. 
# Make sure to define the collate function defined above with the keyword 
# argument collate_fn.

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

print(loader.dataset)

OUTPUT:
Dataset ImageFolder
    Number of datapoints: 10
    Root location: project4/data/images

# We are ready to start using our facial recognition model. 
# We have a DataLoader object that we can iterate over all the images 
# when using the model but let's first see how it's done with just 
# a single image.

img, _ = iter(loader).__next__()
img

# The first step to our facial recognition model is to first 
# detect the faces in the image. We can use MTCNN model like 
# in the previous lesson. From img, we only see one person, 
# Mary Kom. So it's fine to use mtcnn0, which will only 
# return one detected face.

#Use mtcnn0 to detect the face and probability.
face, prob = mtcnn0(img, return_prob=True)

print(type(face))
print(f"Probability of detected face: {prob}")

OUTPUT:
<class 'torch.Tensor'>
Probability of detected face: 0.9999592304229736

#Looks like we are really certain we have detected a face.
'''

#   Running the Inception-ResNet V1 model
'''
# The facial recognition model needs a 4D PyTorch tensor of 
# faces. At the moment, we have just one 3D tensor because 
# we only have one face. We can't use the model right away!
# Let's see what happens if we try.

# Change the shape of the tensor from [3, 240, 240] to [1, 3, 240, 240]. 
# You'll need to use the .unsqueeze method of the PyTorch tensor.
# squeeze will be reduces the dim

face_4d = face.unsqueeze(0)

print(face_4d.shape)

OUTPUT:
torch.Size([1, 3, 240, 240])

# Now we can use Inception-ResNet.
embedding = resnet(face_4d)
print(f"Shape of face embedding: {embedding.shape}")

OUTPUT: Shape of face embedding: torch.Size([1, 512])

# The model returns an embedding for the face using 512 dimensions. 
# The embedding is a vector that represents the features the model 
# extracted. Now we are ready to run our model on all the extracted 
# frames. The process will be:

# * Iterate over all images
# * Run the face detection model
# * If the model returns a result and with a probability of 
#   at least 90%, add the embeddings to a dictionary that 
#   keeps the embeddings for each person separately.

# Filter out results in which face is None or 
# where the probability is less than 0.90.

# Dictionary that maps name to list of their embeddings
name_to_embeddings = {name: [] for name in idx_to_class.values()}

for img, idx in loader:
    face, prob = mtcnn0(img, return_prob=True)
    if face is not None and prob >= 0.9:
        emb = resnet(face.unsqueeze(0))
        name_to_embeddings[idx_to_class[idx]].append(emb)

# Since we had several images of Mary Kom and Ranveer, we have 
# several face embeddings for each of them. We'll want to create 
# a single face embedding for each person by taking the average 
# across their embeddings. This average face embedding will be our 
# reference for a person, often called a faceprint in analogy with 
# fingerprints.

# The first step is to take our list of face embeddings for a person 
# and create one 2D PyTorch tensor. This can be done with torch.
# stack given our name_to_embedings dictionary.

# Take the list of embeddings for both Mary and Ranveer and convert 
# each list to a 2D PyTorch tensor by using torch.stack. Then print the 
# shapes of the two PyTorch tensors resulting from the stacking operation.

# Now we can take the average using torch.mean.

avg_embedding_mary = torch.mean(embeddings_mary, dim=0)
avg_embedding_ranveer = torch.mean(embeddings_ranveer, dim=0)

print(f"Shape of avg_embedding: {avg_embedding_mary.shape}")
print(f"Shape of avg_embedding: {avg_embedding_ranveer.shape}")

OUTPUT:
Shape of avg_embedding: torch.Size([1, 512])
Shape of avg_embedding: torch.Size([1, 512])

# We'll save the embeddings as a list of tuples.
# The saved face embeddings will be used on future, we
# don't need to create new emdeddings for mary com
# The tuples 
# will be in the form of (average embedding, name). Since 
# we have two different people, our list will have length two.

# Create the list to save the embedding. Use the string 
# "mary_kom" and "ranveer" for Mary Kom and Ranveer, respectively.

embeddings_to_save = [
    (avg_embedding_mary, "mary_kom"),
    (avg_embedding_ranveer, "ranveer"),
]


torch.save(embeddings_to_save, "embeddings.pt")

# In the previous code cell, we saved the average embeddings to file. 
# If we need the embedding, we can load it up with torch.load. 
# It accepts as its input the path to the saved embedding. We saved it to 
# the current directory with the name "embeddings.pt".

embedding_data = torch.load("embeddings.pt")

names = [name for _, name in embedding_data]
print(f"Loaded the embedding for: {names}")
'''

#   Inception-ResNet V1 on an Image with One Person¶
'''
# We can consider the embeddings of Mary Kom and Ranveer as our 
# database of known faces. If we now take a new image, we can check 
# if we can recognize whether Mary Kom or Ranveer appear in that image! 
# We'll first focus on one image with only Mary Kom.

#Create a path object for the image in project4/data/extracted_frames/frame_100.jpg

test_img_path = Path('project4','data','extracted_frames','frame_100.jpg')

test_img = Image.open(test_img_path)
test_img

# Earlier, we had used mtcnn0 to just detect one face but let's practice with 
# one that will detect all faces in an image.

mtcnn = MTCNN(image_size=240, keep_all=True, min_face_size=40)
print(f"MTCNN image size: {mtcnn.image_size}")
print(f"MTCNN keeping all faces: {mtcnn.keep_all}")

OUTPUT:
MTCNN image size: 240
MTCNN keeping all faces: True

# Run the face detection model `mtcnn` on the selected frame from 
# above. Recall how `mtcnn` was set to find all faces.

img_cropped_list, prob_list = mtcnn(test_img, return_prob=True )

print(f"Number of detected faces: {len(prob_list)}")
print(f"Probability of detected face: {prob_list[0]}")

OUTPUT:
Number of detected faces: 1
Probability of detected face: 0.9999531507492065

#The code below will then run the face recognition model to get the face embedding.
for i, prob in enumerate(prob_list):
    if prob > 0.90:
        emb = resnet(img_cropped_list[i].unsqueeze(0))

# With the embedding we created above, emb, we can see how different it's 
# from the average embedding for Mary Kom. Let's compute the distance between 
# emb and the two saved embeddings of Mary and Ranveer.

# Use torch.dist to calculate the distance between emb and known_emb. 
# Make sure that dist is a float and not a PyTorch tensor. 
# The calculated distances will be stored in a dictionary.

distances = {}
for known_emb, name in embedding_data:
    dist = float(torch.dist(emb, known_emb))
    distances[name] = dist

closest, min_dist = min(distances.items(), key=lambda x: x[1])
print(f"Closest match: {closest}")
print(f"Calculated distance: {min_dist :.2f}")

OUTPUT:
Closest match: mary_kom
Calculated distance: 0.58

# The smallest distance corresponds to Mary Kom. It makes sense 
# the distance is closer for Mary Kom as that's who is in the image.

#We can use mtcnn.detect to get the bounding boxes for the faces.
boxes, _ = mtcnn.detect(test_img)
print(f"Shape of boxes tensor: {boxes.shape}")

OUTPUT: Shape of boxes tensor: (1, 4)

# Now let's draw a box over a face together with the name of its closest 
# known embedding. We'll use a threshold of 0.8. Any detected face with 
# an embedding that is further than this threshold from a known face 
# will remain unrecognized.

threshold = 0.8

# Recognize Mary Kom's face by drawing a box with her name next to it. 
# Only do so for faces where min_dist is smaller than the threshold. 
# The code below displays the original image and plots the bounding boxes 
# for the detected faces.

# This sets the image size and draws the original image
width, height = test_img.size
dpi = 96
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
axis = fig.subplots()
axis.imshow(test_img)
plt.axis("off")

for box in boxes:
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    axis.add_patch(rect)

    closest, min_dist = min(distances.items(), key=lambda x: x[1])

    # Drawing the box with recognition results

    if min_dist < threshold:
        name = closest
        color = "blue"
    else:
        name = "Unrecognized"
        color = "red"

    plt.text(
        box[0],
        box[1],
        f"{name} {min_dist:.2f}",
        fontsize=12,
        color=color,
        ha="left",
        va="bottom",
    )

plt.axis("off")
plt.show()

REFER face_embeding_with_resent.docx for output

#Notice how we were able to recognize Mary Kom given our specified threshold!


NOTE:     FOR plt.Rectangle ans axis.text

# plt.Rectangle((x, y), width, height, fill=False, color="blue"):

# Creates a rectangle using Matplotlib.
# (box[0], box[1]) specifies the bottom-left corner of the rectangle (bounding box).
# box[2] - box[0] is the width of the rectangle.
# box[3] - box[1] is the height of the rectangle.
# fill=False ensures the rectangle is just an outline.
# color="blue" sets the rectangle color to blue.
# axis.add_patch(rect):

# Adds the rectangle (bounding box) to the plot (axis).

# axis.text(x, y, text, fontsize="large", color=color):
# Adds a text label at position (box[0], box[1]), which is the top-left of the bounding box.
# label = f"{name} {dist:.2f}":
# Combines the name and distance (formatted to 2 decimal places) as text.
# fontsize="large": Makes the font size large.
# color=color:
# Red if the face is "Undetected".
# Blue otherwise.
'''

# Inception-ResNet V1 on an Image With More Than One Person¶
'''
# Now we are ready to run our facial recognition model when 
# there are more than one person in the image. We have chosen 
# the following image.

img_multiple_people_path = Path("project4", "data", "extracted_frames", "frame_210.jpg")
img_multiple_people = Image.open(img_multiple_people_path)

img_multiple_people

# There are six peoples is this image
# The function below encapsulates the code we have already written. 
# It accepts the path of an image and it will draw boxes over each 
# detected face and indicate where it was able to recognize any of the 
# two faces from our embeddings we saved earlier. If any face is recognized, 
# the person's name is displayed along with the distance.

def recognize_faces(img_path, embedding_data, mtcnn, resnet, threshold=0.7):
    # Generating the bounding boxes, faces tensors, and probabilities
    image = Image.open(img_path)
    boxes, probs = mtcnn.detect(image)
    cropped_images = mtcnn(image)

    if boxes is None:
        return

    # This sets the image size and draws the original image
    width, height = image.size
    dpi = 96
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = fig.subplots()
    axis.imshow(image)
    plt.axis("off")

    # Iterating over each face and comparing it against the pre-calculated embeddings
    # from our "database"
    for box, prob, face in zip(boxes, probs, cropped_images):
        if prob < 0.90:
            continue

        # Draw bounding boxes for all detected faces
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            color="blue",
        )
        axis.add_patch(rect)

        # Find the closest face from our database of faces
        emb = resnet(face.unsqueeze(0))
        distances = {}
        for known_emb, name in embedding_data:
            dist = torch.dist(emb, known_emb).item()
            distances[name] = dist

        closest, min_dist = min(distances.items(), key=lambda x: x[1])

        # Drawing the box with recognition results
        name = closest if min_dist < threshold else "Unrecognized"
        color = "red" if name == "Unrecognized" else "blue"
        label = f"{name} {min_dist:.2f}"

        axis.text(box[0], box[1], label, fontsize=8, color=color)

    plt.axis("off")
    plt.show()
recognize_faces(img_multiple_people_path, embedding_data, mtcnn, resnet)

# the above code will get failed to recoginize mary com, we have to adjust the
# threshold

recognize_faces(img_multiple_people_path, embedding_data, mtcnn, resnet, 0.9)


NOTE:   For Figure method in matplot lib

In this code, fig = plt.figure(...) is used to create a new figure in 
Matplotlib, which serves as the container for the entire plot.

fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
plt.figure(...):

Creates a figure object (fig), which acts as a blank canvas where plots (images, charts, etc.) can be drawn.
figsize=(width / dpi, height / dpi):

Specifies the size of the figure in inches.
Since width and height are in pixels, dividing them by dpi converts them into inches.
This ensures that the figure maintains the correct resolution when displayed.
dpi=dpi:

DPI (Dots Per Inch) determines the resolution of the figure.
A higher DPI results in a sharper image.
How It Fits in the Code:
The figure is created with the same dimensions as the image (sample_multiple).
axis = fig.subplots():
Adds a subplot (which is essentially an axis to plot data).
axis.imshow(sample_multiple): Displays the original image on this axis.
label_face(...): Draws a bounding box and labels on the detected face.
Why Use a Figure?
Ensures the image size matches the original.
Provides a container for adding elements like the image, bounding boxes, and labels.
Helps in fine-tuning resolution and layout.
'''