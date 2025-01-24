#     PURPOSE OF CUSTOM MODEL
'''
 Yolo has many objects that can be detected. But sometimes
 it will not have the object we want, lets say we want to
 detect a ambulance the yolo is not already pre-trained
 to detect it. Here the custom model plays role 
'''

#   FORMATING DATA TO BUILD CUSTOM MODEL
'''
 import pathlib
import random
import shutil
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
import ultralytics
import yaml
from IPython import display
from PIL import Image
from tqdm.notebook import tqdm
from ultralytics import YOLO

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using {device} device.")

# PATH TO THE TRAINING DIRECTORY

training_dir = pathlib.Path("data_images", "train")
images_dir = training_dir / 'images'
annotations_dir = training_dir / 'annotations'

print("Images     :", images_dir)
print("Annotations:", annotations_dir)

#  print out the first 25 lines of the first annotation file.

!head -n 25 $annotations_dir/01.xml

OUTPUT:

<annotation>
	<folder>Images</folder>
	<filename>02_Motijheel_280714_0005.jpg</filename>
	<path>E:\Datasets\Dataset\Images\02_Motijheel_280714_0005.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1200</width>
		<height>800</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>bus</name>
		<pose>Unspecified</pose>
		<truncated>1</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>833</xmin>
			<ymin>390</ymin>
			<xmax>1087</xmax>
			<ymax>800</ymax>
		</bndbox>
	</object>

 object tag has the detected object's class name with bouding
 box,

 Unfortunately, this is not the format that YOLO needs. 
 The YOLO format is a text file, with each object being a line 
 of the format like in the below format

 class_index x_center y_center width height

 where class_index is a number assigned to class. 
 The bounding box is centered at (x_center, y_center), 
 with a size of width height. All of these 
 dimensions are given as a fraction of the image size, 
 rather than in pixels. These are called normalized coordinates.

 classes = [
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

class_mapping = {cls : idx for idx,cls in enumerate(classes)}

def xml_to_yolo_bbox(bbox, width, height):
    """Convert the XML bounding box coordinates into YOLO format.

    Input:  bbox    The bounding box, defined as [xmin, ymin, xmax, ymax],
                    measured in pixels.
            width   The image width in pixels.
            height  The image height in pixels.

    Output: [x_center, y_center, bb_width, bb_height], where the bounding
            box is centered at (x_center, y_center) and is of size
            bb_width x bb_height.  All values are measured as a fraction
            of the image size."""

    xmin, ymin, xmax, ymax = bbox
    x_center = (xmax + xmin) / 2 / width
    y_center =  (ymin + ymax) / 2 / height
    bb_width = (xmax - xmin) / width
    bb_height = (ymax - ymin) / height

    return [x_center, y_center, bb_width, bb_height]


xml_to_yolo_bbox([xmin, ymin, xmax, ymax], width, height)

# In below function we pass annotation file and extract
# all the heights, width, x,y min max

def parse_annotations(f):
    """Parse all of the objects in a given XML file to YOLO format.

    Input:  f      The path of the file to parse.

    Output: A list of objects in YOLO format.
            Each object is a list [index, x_center, y_center, width, height]."""

    objects = []

    tree = ET.parse(f)
    root = tree.getroot()
    width = int(root.find("size").find("width").text)
    height = int(root.find("size").find("height").text)

    for obj in root.findall("object"):
        label = obj.find("name").text
        class_id = class_mapping[label]
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        yolo_bbox = xml_to_yolo_bbox([xmin, ymin, xmax, ymax], width, height)

        objects.append([class_id] + yolo_bbox)

    return objects


objects = parse_annotations(annotations_dir / "01.xml")
print("First object:", objects[0])

OUTPUT: First object: [4, 0.8, 0.74375, 0.21166666666666667, 0.5125]


                # Preparing the Directory Structure

# yolo needs specific file structure that is what we going to do

#First:

# YOLO can read PNG files as well as JPEG files. But as a 
# format, PNG is inefficient for photographs, compared to JPEG. A
# dditionally, some of these files are mildly corrupted. 
# YOLO can open them, but it'll print out warnings when it does.

# To address both of these issues, we'll convert 
# all of the images to RGB JPEG files before feeding them 
# into YOLO. The following function implements the conversion.

def convert_image(fin, fout):
    """Open the image at `fin`, convert to a RGB JPEG, and save at `fout`."""
    Image.open(fin).convert("RGB").save(fout, "JPEG")

test_image = images_dir / "193.png"
convert_image(test_image,'test_image.jpg')

display.display(
    Image.open(images_dir / "193.png"),
    Image.open(test_image)  # Add path to the test JPEG

)

 # converted and unconverted image are same, we have to check
 # this after conversion 

# TWO:

 #For training, YOLO expects a directory structure like so:

data_yolo
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val

# That is what the below code does
 
yolo_base = pathlib.Path("data_yolo")

# It's important to clear out the directory, if it already
# exists.  We'll get a different train / validation split
# each time, so we need to make sure the old images are
# cleared out.
shutil.rmtree(yolo_base, ignore_errors=True)

(yolo_base / "images" / "train").mkdir(parents=True)
# Create the remaining directories.
(yolo_base / "images" / "val").mkdir(parents=True)
(yolo_base / "labels" / "train").mkdir(parents=True)
(yolo_base / "labels" / "val").mkdir(parents=True)

!tree $yolo_base

 # Next we have split train and val data set that
 # is what the below code does

 def write_label(objects, filename):
    """Write the annotations to a file in the YOLO text format.

    Input:  objects   A list of YOLO objects, each a list of numbers.
            filename  The path to write the text file."""

    with open(filename, "w") as f:
        for obj in objects:
            # Write the object out as space-separated values
            f.write(' '.join(str(x) for x in obj))
            # Write a newline
            f.write('\n')

train_frac = 0.8 #denotes 80% of the data is train and 20% is val
images = list(images_dir.glob("*"))

for img in tqdm(images):
    # randomly splits train and val
    split = "train" if random.random() < train_frac else "val"
    
    #The .stem attribute in Python's pathlib module is used to extract the 
    # base name of a file without its extension

    annotation = annotations_dir / f"{img.stem}.xml"  
    try:
        # This might raise an error:
        parsed = parse_annotations(annotation)
    except Exception as e:
        print("Igonored", img.stem)
        print(e)
        continue
    
    dest = yolo_base / "labels" / split / f"{img.stem}.txt"
    write_label(parsed, dest)

    dest = yolo_base / "images" / split / f"{img.stem}.jpg"
    convert_image(img, dest)
'''

#   TRAINING THE MODEL
'''
 For training yolo model a file called YAML
 needs to be created(which is like json) that
 contains all the information about the model

 The data for training a YOLO model needs to be described 
 in a YAML file. 

 metadata = {
    "path": str(
        yolo_base.absolute()
    ),  # It's easier to specify absolute paths with YOLO.
    
    "train":"images/train", # Training images, relative to above.
    
    "val": "images/val", # Validation images
    
    "names": classes, # Class names, as a list
    
    "nc": len(classes), # Number of classes
}

print(metadata)

 # We dont need to define the path for labels
 # yolo will automatically picks from the
 # absoulate path if we correctly give the
 # train and val image path

# Writing the YAML file
yolo_config = "data.yaml"
# Using yaml.safe_dump() protects you from some oddities in the YAML format.
# It takes the same arguments as json.dump().
yaml.safe_dump(metadata, open(yolo_config, 'w'))


!cat data.yaml

# we will use the YOLOv8n (n for nano) model. The nano model 
# is less than 30% of the size of the small model, but it still 
# manages 80% of the small model's performance. This'll cut down t
# he training time without hurting performance too much.

model = YOLO("yolov8n.pt")

print(model)

# Loads pretrained model
results = model.train(
    data=yolo_config,
    epochs=5
    # The number of epochs to wait before early stopping
    patience=2,
    batch = 8,
    # The no of CPU threads loading data from disk, The
    # default is 18
    workers=2,
)

# Results of the model will be store in this
# dir runs/detect/train 

# Don't change this
save_dir = Path("runs/detect/train")

# the prints all the contents inside that dir in linux
!tree $save_dir

# Train will also returns different plots which will
# have the information about what was learned during
# training process, one of the plot is PR_curve.png (precision-recall curves)
# A precision-recall curve is a graph that shows 
# the trade-off between precision (positive prediction accuracy) 
# and recall (ability to detect all positive cases) at different classification thresholds.


'''