#                       INTRO
'''
 Here we are going to develop the face recognizer to a 
 flask app
'''

#                      CODE in main.py
'''
%load_ext autoreload
%autoreload 2

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

Next we'll initlize the resnet and take face embeddings from 
previous


mtcnn = MTCNN(image_size=240, keep_all=False, min_face_size=40)
resnet = InceptionResnetV1(pretrained='vggface2')

resnet = resnet.eval()

print(f"MTCNN image size: {mtcnn.image_size}")
print(f"MTCNN keeping all faces: {mtcnn.keep_all}")
print(f"InceptionResnet weight set: {resnet.pretrained}")

# We'll be getting images uploaded to our app, rather than reading from disk. But we'll 
# need to test things as we go, so let's get a few sample images.

# Create a variable to access the extracted frames in 
# project4/data/extracted_frames. Use pathlib.

project_dir = Path('project4')
images_dir = project_dir / 'data' / 'extracted_frames'

print(images_dir)

sample_single = Image.open(images_dir / "frame_10.jpg")
sample_multiple = Image.open(images_dir / "frame_1280.jpg")
'''