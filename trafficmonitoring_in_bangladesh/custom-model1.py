#     PURPOSE OF CUSTOM MODEL
'''
 We can custom whatever object we want to detect in the image
 For example, we can detect a car, a bus, a person, etc.

 By default Yolo predicts all the possible objects in the image.
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

Image.open(save_dir/"PR_curve.png")

# Another plot is confusion matrix that helps to
# how our model getting confused in predictions to find correct class
# and how it founds the class

REFER About yolo(obj_classification).doc fot output

# Another file will be also generated called results.csv 
# that has the the values for three loss functions
# errors for train and val dataset

df = pd.read_csv(save_dir / "results.csv", skipinitialspace=True).set_index(
    "epoch"
)
df.head()

OUTPUT:

epoch
	time	train/box_loss	train/cls_loss	train/dfl_loss	metrics/precision(B)	metrics/recall(B)	metrics/mAP50(B)	metrics/mAP50-95(B)	val/box_loss	val/cls_loss	val/dfl_loss	lr/pg0	lr/pg1	lr/pg2
														
1	35.1072	1.45223	3.32406	1.16199	0.59554	0.11979	0.10365	0.06298	1.36083	2.15096	1.11174	0.000133	0.000133	0.000133
2	67.5593	1.42356	2.24330	1.17132	0.67236	0.15496	0.13611	0.08650	1.33130	1.89784	1.10711	0.000257	0.000257	0.000257
3	99.9395	1.38437	2.04035	1.16109	0.56155	0.18055	0.16316	0.10168	1.30970	1.70307	1.10191	0.000373	0.000373	0.000373
4	131.5240	1.36695	1.91453	1.15004	0.60072	0.19290	0.17265	0.10846	1.28410	1.66555	1.08647	0.000360	0.000360	0.000360
5	163.3040	1.32782	1.81436	1.12958	


#Lets plot the classification loss over period for train and val

# The `.plot` method on DataFrames may be useful.
df[['train/cls_loss','val/cls_loss']].plot(marker='.')

#Weights for last training epoch and 
# best training performance epoch's weight loss are will be stored
# in the path runs/detect/train/weights/best.pt
#By load those best weight loss value we dont need
# to train it again to get better result

saved_model = YOLO(save_dir/'weights'/'best.pt')

print(saved_model)

# Lets make prediction on this saved model
# we'll make prediction on the extracted frames frame_600 from the video

predict_results = saved_model.predict(
    "data_video/extracted_frames/frame_600.jpg",
    # Only return objects detected at this confidence level or higher
    conf=0.5,
    # Save output to disk
    save=True,
)

f"Results type: {type(predict_results)}, length {len(predict_results)}"

# predict_results will give you the result which has confidence
# in 50% that(conf=0.5) and saves the result on current dir

OUTPUT:

image 1/1 /app/data_video/extracted_frames/frame_600.jpg: 384x640 7 cars, 1 suv, 3 three wheelers (CNG)s, 60.8ms
Speed: 2.5ms preprocess, 60.8ms inference, 206.7ms postprocess per image at shape (1, 3, 384, 640)
Results saved to runs/detect/predict4
"Results type: <class 'list'>, length 1"

Here saved to runs/detect/predict4 is the saved path

# The below code will give you the bounding boxes on the frame_600.jpg
predict_results[0].boxes

OUTPUT:
ltralytics.engine.results.Boxes object with attributes:

cls: tensor([ 5.,  5., 17.,  5., 17.,  5., 15., 17.,  5.,  5.,  5.], device='cuda:0')
conf: tensor([0.9604, 0.9439, 0.9421, 0.8939, 0.7899, 0.7891, 0.7002, 
id: None
is_track: False
orig_shape: (360, 640)
shape: torch.Size([11, 6])
xywh: tensor([[ 80.4396, 292.1061, 160.5392, 134.6830],
        [36
xywhn: tensor([[0.1257, 0.8114, 0.2508, 0.3741],
    

# Below code will opens the saved result
Image.open(pathlib.Path(predict_results[0].save_dir) / "frame_600.jpg")
'''