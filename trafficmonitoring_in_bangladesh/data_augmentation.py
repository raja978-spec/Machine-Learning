#                   WHAT IS DATA AUGMENTATION
'''
 * There will be an situation when we can't give large amount ot
   input to build model.

 * It will complicated to memorize all the inputs in model.

 * Here the data augmentation play's it's role.

 * It creates new data based on existing data

 * data augmentation enhances a model's ability to generalize. This will increase their 
   performance on unseen data, which is the ultimate goal of a machine learning system.
'''

#               DA CODE IMPORT STATEMENTS
'''

 We will need version 2 of the torchvision.transforms 
 module here. The API is slightly different than that 
 of version 1 that we've used previously, but it's pretty 
 similar.
 
import pathlib
import sys

import matplotlib.pyplot as plt
import torch
import torchinfo
import torchvision
import ultralytics
from PIL import Image
from torchvision.transforms import v2
from ultralytics import YOLO

print("Platform:", sys.platform)
print("Python version:", sys.version)
print("---")
print("matplotlib version:", plt.matplotlib.__version__)
print("PIL version:", Image.__version__)
print("torch version:", torch.__version__)
print("torchvision version:", torchvision.__version__)
print("ultralytics version:", ultralytics.__version__)

OUTPUT:

Platform: linux
Python version: 3.11.0 (main, Nov 15 2022, 20:12:54) [GCC 10.2.1 20210110]
---
matplotlib version: 3.9.2
PIL version: 10.2.0
torch version: 2.2.2+cu121
torchvision version: 0.17.2+cu121
ultralytics version: 8.3.27

CLASS_DICT = dict(
    enumerate(
        [
            "ambulance",
            "army vehicle",
            "auto rickshaw",
            "bicycle",
            "bus",
            "car",
            "garbagevan",
            "human hauler",
            "minibus",
            "minivan",
            "motorbike",
            "pickup",
            "policecar",
            "rickshaw",
            "scooter",
            "suv",
            "taxi",
            "three wheelers (CNG)",
            "truck",
            "van",
            "wheelbarrow",
        ]
    )
)

print("CLASS_DICT type,", type(CLASS_DICT))
CLASS_DICT

OUTPUT:

CLASS_DICT type, <class 'dict'>
{0: 'ambulance',
 1: 'army vehicle',
 2: 'auto rickshaw',
 3: 'bicycle',
 4: 'bus',
 5: 'car',
 6: 'garbagevan',
 7: 'human hauler',
 8: 'minibus',
 ....}
'''

#        LOAD PATH
'''
In the previous notebook, we passed our training images to the YOLO model 
and let it do its thing. The obvious assumption to make is 
that these images would be used as is, but it turns out not 
to be so. To demonstrate what was happening, we'll load the 
model back up and poke around inside of it a bit.

Let's start by finding a saved version of the model. 
This cell should show all of the training runs that 
have been completed.

run_dir = runs_dir / 'train'
weights_file = run_dir / 'weights' / 'best.pt'

print("Weights file exists?", weights_file.exists())

OUTPUT:
[PosixPath('runs/detect/train')]

run_dir = runs_dir / 'train'
weights_file = run_dir / 'weights' / 'best.pt'

print("Weights file exists?", weights_file.exists())

'''

#              LOAD MODEL
'''
model = YOLO(weights_file)

torchinfo.summary(model)

We need to get the model set up to load the data. 
The easiest way to do that is to train it for an epoch.

When you call .train() on a YOLO model, it sets up a 
data loader, if it doesn't already exist. Unfortunately, 
there's no easy way to trigger that set-up step without 
doing an epoch of training. 😔

result = model.train(
    data=model.overrides["data"],
    epochs=1,
    batch=8,
    workers=1,
)

'''

#                SAVE MODEL
'''
The model should now have a .trainer attribute, which has a .train_loader 
attribute. This will be a DataLoader that loads the training data.

loader = model.trainer.train_loader

print(type(loader))

OUTPUT:
<class 'ultralytics.data.build.InfiniteDataLoader'>
'''

#                      LOAD DATA TO BATCH
'''

# Data loaders are iterables. That is, you can put them in 
# a for loop to load data one batch at a time. We just want 
# to read one batch from it, though.

# Load one batch from loader into the variable batch. 
# You can do this by constructing a for loop over loader 
# and calling break inside the loop, so that it only runs once.

for batch in loader:
    break
print(type(batch))

OUTPUT: <class 'dict'>

A more advanced way to accomplish this same 
thing is: batch = next(iter(loader))

print(batch.keys())

OUTPUT:

dict_keys(['im_file', 'ori_shape', 'resized_shape', 'img', 
'cls', 'bboxes', 'batch_idx'])
print(batch['img'].shape)

OUTPUT:
torch.Size([8, 3, 640, 640])

The dimension of 3 represents the color channels. The dimension 
of 640 are the width and height. 

8 represents batch size, so this tensor will have 8 training
images

print(batch['bboxes'].shape)

OUTPUT:
torch.Size([102, 4])

# That seems like a lot of bounding boxes(102) for one image, 
# so these must be the boxes for all of the images in the batch.

# The exact number of bounding boxes will depend on the 
# random batch that got delivered to you. If you re-run the
# cell that creates batch, you'll find that you get another 
# number here.

# The image index in the batch that the box corresponds to 
# is given in the batch_idx value.

print(batch["batch_idx"])

OUTPUT:
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 2., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 5., 6., 6., 6., 6., 6., 7., 7., 7., 7., 7., 7., 7.,
        7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7., 7.])
'''

#         PLOT BOUNDING BOXES ON ALL IMAGE
'''
Thus, we can select the bounding boxes for a particular image 
in a batch by finding the rows that correspond to a particular 
batch index value. This is implemented for us in the following 
function, which will plot the bounding boxes on top of the image.

def plot_with_bboxes(img, bboxes, cls, batch_idx=None, index=0, **kw):
    """Plot the bounding boxes on an image.

    Input:  img     The image, either as a 3-D tensor (one image) or a
                    4-D tensor (a stack of images).  In the latter case,
                    the index argument specifies which image to display.
            bboxes  The bounding boxes, as a N x 4 tensor, in normalized
                    XYWH format.
            cls     The class indices associated with the bounding boxes
                    as a N x 1 tensor.
            batch_idx   The index of each bounding box within the stack of
                        images.  Ignored if img is 3-D.
            index   The index of the image in the stack to be displayed.
                    Ignored if img is 3-D.
            **kw    All other keyword arguments are accepted and ignored.
                    This allows you to use dictionary unpacking with the
                    values produced by a YOLO DataLoader.
    """
    if img.ndim == 3:
        image = img[None, :]
        index = 0
        batch_idx = torch.zeros((len(cls),))
    elif img.ndim == 4:
        # Get around Black / Flake8 disagreement
        indp1 = index + 1
        image = img[index:indp1, :]

    inds = batch_idx == index
    res = ultralytics.utils.plotting.plot_images(
        images=image,
        batch_idx=batch_idx[inds] - index,
        cls=cls[inds].flatten(),
        bboxes=bboxes[inds],
        names=CLASS_DICT,
        threaded=False,
        save=False,
    )

    return Image.fromarray(res)

plot_with_bboxes(**batch, index=0)

# It will give weired augmented images with bounding boxes

# SEE data_agumentation.docx for output

# The file names from the batch are stored in the im_file key. 
# We can use that to look up the original image associated with 
# this index and see what it looks like.

Image.open(batch['im_file'][1])

# Above code will give the unaugmented original image
# SEE data_agumentation.docx for output

Comparing the two, we can see that the original image was distorted 
and combined with other images before being loaded into the YOLO model. 
The YOLO model applies a number of augmentation steps by default. 
(You can take a look at all of the augmentation settings in YOLO.
https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters
) 
This increases the diversity of training images, which should help the model generally.
'''

#     LOAD IMAGE, BBOXES, CLASSES to do Data Augmentation with Torchvision
'''
 Yolo defaultly does this data augmentation, but sometimes we need
 to build our own augmentation with torchvision.

 To demonstrate this, we'll load a sample image. The code below will 
 get the file paths for 01.jpg and its associated label file.

 yolo_base = pathlib.Path("data_yolo")
sample_fn = next((yolo_base / "images").glob("*/01.jpg"))
sample_labels = next((yolo_base / "labels").glob("*/01.txt"))

print(sample_fn)
print(sample_labels)

OUTPUT:
data_yolo/images/val/01.jpg
data_yolo/labels/val/01.txt

Convert the image to a tensor. In the transforms version 2 module, 
this can be done with the confusingly-named ToImage transform.

sample_torch = v2.ToImage()(sample_image)

print(sample_torch.shape)

OUTPUT:
torch.Size([3, 800, 1200])

# The bounding boxes are stored in the label file. Let's take a 
# look a the first five lines to remember what it looks like.

!head -n5 $sample_labels

OUTPUT:
4 0.8 0.74375 0.21166666666666667 0.5125
4 0.7995833333333333 0.424375 0.0975 0.13875
4 0.7995833333333333 0.33 0.08416666666666667 0.0575
13 0.66375 0.595625 0.059166666666666666 0.15875
13 0.66875 0.483125 0.0425 0.05625

# Each line represents a bounding box.  The first element is the class 
# index.  This is followed by the _x_ and _y_ coordinates 
# of the box center, the width, and the height.

# Load the bounding box data into a variable named label_data. 
# It should be a list of the bounding boxes. Each bounding box 
# will itself be a list of five strings in the same order they are 
# in the file. Don't worry about converting the strings to numbers yet.

 # Load the data into `label_data`
with open(sample_labels, "r") as f:
    label_data=[row.split() for row in f]
    
label_data[:5]

OUTPUT:

[['4', '0.8', '0.74375', '0.21166666666666667', '0.5125'],
 ['4', '0.7995833333333333', '0.424375', '0.0975', '0.13875'],
 ['4', '0.7995833333333333', '0.33', '0.08416666666666667', '0.0575'],
 ['13', '0.66375', '0.595625', '0.059166666666666666', '0.15875'],
 ['13', '0.66875', '0.483125', '0.0425', '0.05625']]

#  Create a tensor containing the class indices. For compatibility with 
#  our plotting function it should be a 
#  N * 1 tensor.

 classes = torch.Tensor([[int(row[0])] for row in label_data])

print("Tensor shape:", classes.shape)
print("First 5 elements:\n", classes[:5])

OUTPUT:

Tensor shape: torch.Size([30, 1])
First 5 elements:
 tensor([[ 4.],
        [ 4.],
        [ 4.],
        [13.],
        [13.]])

# Load the bounding box coordinates into a 
# N * 4 tensor.

bboxes = torch.Tensor([[float(el) for el in row[1:]] for row in label_data])

print("Tensor shape:", bboxes.shape)
print("First 5 elements:\n", bboxes[:5])

OUTPUT:
Tensor shape: torch.Size([30, 4])
First 5 elements:
 tensor([[0.8000, 0.7437, 0.2117, 0.5125],
        [0.7996, 0.4244, 0.0975, 0.1388],
        [0.7996, 0.3300, 0.0842, 0.0575],
        [0.6637, 0.5956, 0.0592, 0.1587],
        [0.6687, 0.4831, 0.0425, 0.0562]])

# All of these coordinates are normalized by the width 
# or height, as appropriate. This won't work with transformations 
# like rotation, which need the same units used on each axis.

# Convert the bounding box coordinates to pixel units.

sample_width, sample_height = sample_image.size

scale_factor = torch.Tensor([sample_width, sample_height, sample_width, sample_height])

bboxes_pixels = bboxes * scale_factor

print("Tensor shape:", bboxes_pixels.shape)
print("First 5 elements:\n", bboxes_pixels[:5])

OUTPUT:
Tensor shape: torch.Size([30, 4])
First 5 elements:
 tensor([[960.0000, 595.0000, 254.0000, 410.0000],
        [959.5000, 339.5000, 117.0000, 111.0000],
        [959.5000, 264.0000, 101.0000,  46.0000],
        [796.5000, 476.5000,  71.0000, 127.0000],
        [802.5000, 386.5000,  51.0000,  45.0000]])

# In order for the transformations to know how to transform the bounding 
# boxes, they need to know that the coordinates represent the centers and 
# dimensions of the boxes. This is done by creating a special BoundingBoxes 
# tensor. This type has a format attribute. By setting this to "CXCYWH", 
# we're telling it that the columns represent the Center X coordinate, 
# the Center Y coordinate, the Width, and the Height. 

# This tensor also is given the size of the image, 
# so it doesn't need to look that up for transformations.

bboxes_tv = torchvision.tv_tensors.BoundingBoxes(
    bboxes_pixels,
    format="CXCYWH",
    # Yes, that's right.  Despite using width x height everywhere
    # else, here we have to specify the image size as height x
    # width.
    canvas_size=(sample_height, sample_width),
)

print("Tensor type:", type(bboxes_tv))
print("First 5 elements:\n", bboxes_tv[:5])

OUTPUT:

Tensor type: <class 'torchvision.tv_tensors._bounding_boxes.BoundingBoxes'>
First 5 elements:
 tensor([[960.0000, 595.0000, 254.0000, 410.0000],
        [959.5000, 339.5000, 117.0000, 111.0000],
        [959.5000, 264.0000, 101.0000,  46.0000],
        [796.5000, 476.5000,  71.0000, 127.0000],
        [802.5000, 386.5000,  51.0000,  45.0000]])

# Let's double check that we did all of those conversions 
# correctly.  Do the bounding boxes line up with the correct objects?

plot_with_bboxes(sample_torch, bboxes, classes)

REFER data_augmentation.py file for OUTPUT image
'''

#     HORIZONTAL DATA AUGMENTATION WITH TORCHVISION
'''

# If everything looks good, we'll introduce some transformations.  
# The first one will be a horizontal flip.  Many everyday objects 
# have bilateral symmetry (or nearly so), so a flipped image will still 
# have the same object classes in it.  This makes a horizontal flip 
# a good data augmentation transformation.

# (In contrast, up/down symmetry is less common.  A vertical flip 
# is generally not as useful, unless you need to recognize upside-down objects.)

# The transforms version 2 module has a `RandomHorizontalFlip` transformation.  
# This takes the probability of a flip as an argument.

# Use the `RandomHorizontalFlip` transformation to flip the sample image.  Set `p=1` to ensure that the flip happens.

flipped = v2.RandomHorizontalFlip(p=1)(sample_torch)

plot_with_bboxes(flipped, bboxes_tv, classes)

OUTPUT: this code will flips the image to horizontal
         REFER data_augmentation.docx

It will flips the image not bounding boxes, let's do it

flipped, flipped_bboxes = v2.RandomHorizontalFlip(p=1)(sample_torch, bboxes_tv)

plot_with_bboxes(flipped, flipped_bboxes, classes)
'''

#      ROTATE THE IMAGE
'''

# Apply the RandomRotation transformation. This takes an argument 
# of the maximum number of degrees to rotate the image. Set it to 90.

rotated, rotated_bboxes = v2.RandomRotation(90)(sample_torch, bboxes_tv)

plot_with_bboxes(rotated, rotated_bboxes, classes)

REFER data..doc file for image output
'''

#    DO AUGMENTATION WITH COMPOSE
'''
# Multiple augmentation techniques can be chained together to produce even more diversity in the 
# training images. Within Torchvision, this can be accomplished by the Compose element.

# Create an augmentation pipeline that combines the RandomHorizontalFlip 
# with the RandomRotation. This time, set the probability of the flip to 50%.

transforms = v2.Compose(
    [
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=90)
    ]
)

transformed, transformed_bboxes = transforms(sample_torch, bboxes_tv)

plot_with_bboxes(transformed, transformed_bboxes, classes)

REFER .doc file for output

# There are a large number of transformations that can be used
# for data augmentation. Scroll through the documentation 
# to get a view of the range of possibilities.
https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended

# In addition to the transforms we've already used, note:

# RandomResizedCrop will randomly crop the image down, 
# and then it resizes the output to a set dimension.

# ColorJitter can randomly adjust the brightness, contrast, 
# saturation, and hue of the image, within specified ranges.

transforms = v2.Compose(
    [
        v2.RandomResizedCrop(size=(240, 400)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=30),
        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    ]
)

transformed, transformed_bboxes = transforms(sample_torch, bboxes_tv)
plot_with_bboxes(transformed, transformed_bboxes, classes)

'''