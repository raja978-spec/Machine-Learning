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


mtcnn = MTCNN(image_size=240, keep_all=True, min_face_size=40)
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

#        Locating Faces
'''
def locate_faces(image):
    cropped_images, probs = mtcnn(image, return_prob=True)
    boxes, _ = mtcnn.detect(image)

    if boxes is None or cropped_images is None:
        return []
    else:
        return list(zip(boxes, probs, cropped_images))

multiple_faces = locate_faces(sample_multiple)
print(f"How many faces in the sample with 5 faces: {len(multiple_faces)}")

OUTPUT:

How many faces in the sample with 5 faces: 5

# For the moment, we'll skip the step about filtering out low probability 
# faces and come back to it. Then next steps after that are to get the 
# embedding for our face, and compare it to the known faces. We'll want 
# to get back both the name (if we know it) and the distance.
'''

#         DETERMINE NAME DIST
'''

 the embedding for our face, and compare it to the known faces. 
 We'll want to get back both the name (if we know it) and the distance.

def determine_name_dist(cropped_image, threshold=0.9):
    # Use `resnet` on `cropped_image` to get the embedding.
    # Don't forget to unsqueeze!
    emb = resnet(cropped_image.unsqueeze(0))

    # We'll compute the distance to each known embedding
    distances = []
    for known_emb, name in embedding_data:
        # Use torch.dist to compute the distance between
        # `emb` and the known embedding `known_emb`
        dist = torch.dist(emb, known_emb).item()
        distances.append((dist, name))

    # Find the name corresponding to the smallest distance
    dist, closest = min(distances)

    # If the distance is less than the threshold, set name to closest
    # otherwise set name to "Undetected"
    if dist < threshold:
        name = closest
    else:
        name= "Undetected"

    return name, dist
'''

#           Labeling Images
'''

The function below adds the box and label to an existing image. 
To use it, we'll need to plot our image with matplotlib, 
then call this function in the same cell. We'll be reusing the 
same structure from the previous lessons, with a little 
simplification.

def label_face(name, dist, box, axis):
    """Adds a box and a label to the axis from matplotlib
    - name and dist are combined to make a label
    - box is the four corners of the bounding box for the face
    - axis is the return from fig.subplots()
    Call this in the same cell as the figure is created"""

    # Add the code to generate a Rectangle for the bounding box
    # set the color to "blue" and fill to False
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    axis.add_patch(rect)

    # Set color to be red if the name is "Undetected"
    # otherwise set it to be blue
    if name == "Undetected":
        color = 'red'
    else:
        color = 'blue'
    
    label = f"{name} {dist:.2f}"
    axis.text(box[0], box[1], label, fontsize="large", color=color)

To demonstrate how it works, we'll plot the first face found in 
the multiple faces. The code at the beginning sets `matplotlib` 
to create an output image the same size as the photo we're
working with.

# This sets the image size
# and draws the original image
width, height = sample_multiple.size
dpi = 96
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
axis = fig.subplots()
axis.imshow(sample_multiple)
plt.axis("off")
face = multiple_faces[0]
cropped_image = face[2]
box = face[0]

name, dist = determine_name_dist(cropped_image)

label_face(name, dist, box, axis)

#Now we can run this in a loop on each face.

# This sets the image size
# and draws the original image
width, height = sample_multiple.size
dpi = 96
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
axis = fig.subplots()
axis.imshow(sample_multiple)
plt.axis("off")

for face in multiple_faces:
    box, prob, cropped_image = face

    name, dist = determine_name_dist(cropped_image)

    label_face(name, dist, box, axis)


REFER face_embeding_with_resent.docx for output
'''

#      PUTTING ALL THE ABOVE INTO A SINGLE FUNCTION
'''
def add_labels_to_image(image):
    # This sets the image size
    # and draws the original image
    width, height = image.width, image.height
    dpi = 96
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = fig.subplots()
    axis.imshow(image)
    plt.axis("off")

    # Use the function locate_faces to get the individual face info
    faces = locate_faces(image)

    for box, prob, cropped in faces:
        # If the probability is less than 0.90,
        # It's not a face, skip this run of the loop with continue
        if prob < 0.90:
            continue
        
        # Call determine_name_dist to get the name and distance
        name, dist = determine_name_dist(cropped)

        # Use label_face to draw the box and label on this face
        label_face(name, dist, box, axis)

    return fig
'''


#           START BULIDING APP
'''
Our Flask application will have three files:

The face_recognition.py we just created, that has our logic
app.py, the main application that will handle interaction
upload.html, a web page to display to users
The upload.html already exists for us, it's in the templates 
directory. It creates an interface web page with two buttons, 
one to select which image to run on, and one to upload the file 
to be processed.

Let's build up the app.py so our users can interact with our code without knowing Python.
'''

#         APP.py
'''
# **AI Lab: Deep Learning for Computer Vision**
# **WorldQuant University**
#
#

# **Usage Guidelines**
#
# This file is licensed under Creative Commons Attribution-NonCommercial-
# NoDerivatives 4.0 International.
#
# You **can** :
#
#   * ✓ Download this file
#   * ✓ Post this file in public repositories
#
# You **must always** :
#
#   * ✓ Give credit to WorldQuant University for the creation of this file
#   * ✓ Provide a link to the license
#
# You **cannot** :
#
#   * ✗ Create derivatives or adaptations of this file
#   * ✗ Use this file for commercial purposes
#
# Failure to follow these guidelines is a violation of your terms of service and
# could lead to your expulsion from WorldQuant University and the revocation
# your certificate.
#
#

import io


from flask import Flask, render_template, request, send_file
from PIL import Image

# Import our face recognition code
from face_recognition import add_labels_to_image
# Starts Flask
app = Flask(__name__)


# Set the route to "/"
@app.route('/')
def home():
    return render_template("upload.html")


@app.route("/recognize", methods=["POST"])
def process_image():
    # Display an error message if no image found
    if "image" not in request.files:
        return "No image provided", 400

    # Get the file sent along with the request
    file = request.files["image"]

    # Video also shows up as an image
    # we want to reject those as well
    if not file.mimetype.startswith("image/"):
        return "Image format not recognized", 400

    image_data = file.stream

    # Run our face recognition code!
    img_out = run_face_recognition(image_data)

    if img_out == Ellipsis:
        return "Image processing not enabled", 200
    else:
        # Our function returns something from matplotlib,
        # convert it to a web-friendly form and return it
        out_stream = matplotlib_to_bytes(img_out)
        return send_file(out_stream, mimetype="image/jpeg")


def run_face_recognition(image_data):
    # Open image_data with PIL
    input_image = Image.open(image_data)

    # Run our function on the PIL image
    img_out = add_labels_to_image(input_image)

    return img_out


def matplotlib_to_bytes(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="jpg", bbox_inches="tight")
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    app.run(debug=True)


# This file © 2024 by WorldQuant University is licensed under CC BY-NC-ND 4.0.

'''

#     TEMPLATES/UPLOAD.HTML
'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Upload Image or Video</title>
</head>
<body>
    <h1>Upload Image for Facial Recognition</h1>
    <form action="recognize" method="post" enctype="multipart/form-data">
        <label for="fileUpload">Choose an image or video:</label>
        <input type="file" id="fileUpload" name="image" accept="image/*,video/*">
        <button type="submit">Upload and Analyze</button>
    </form>
</body>
</html>

'''

#           FACE_RECOG.py
'''

We will put all the above code to this file for our flask
app

# **AI Lab: Deep Learning for Computer Vision**
# **WorldQuant University**
#
#

# **Usage Guidelines**
#
# This file is licensed under Creative Commons Attribution-NonCommercial-
# NoDerivatives 4.0 International.
#
# You **can** :
#
#   * ✓ Download this file
#   * ✓ Post this file in public repositories
#
# You **must always** :
#
#   * ✓ Give credit to WorldQuant University for the creation of this file
#   * ✓ Provide a link to the license
#
# You **cannot** :
#
#   * ✗ Create derivatives or adaptations of this file
#   * ✗ Use this file for commercial purposes
#
# Failure to follow these guidelines is a violation of your terms of service and
# could lead to your expulsion from WorldQuant University and the revocation
# your certificate.
#
#

# Import needed libraries
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Load MTCNN, Resnet, and the embedding data

mtcnn = MTCNN(image_size=240, keep_all=True, min_face_size=40)

resnet = InceptionResnetV1(pretrained='vggface2')
embedding_data = torch.load('embeddings.pt')

resnet = resnet.eval()


# Fill in the locate_face function
def locate_faces(image):
    cropped_images, probs = mtcnn(image, return_prob=True)
    boxes, _ = mtcnn.detect(image)

    if boxes is None or cropped_images is None:
        return []
    else:
        return list(zip(boxes, probs, cropped_images))
    


# Fill in the determine_name_dist function
def determine_name_dist(cropped_image, threshold=0.9):
    # Use `resnet` on `cropped_image` to get the embedding.
    # Don't forget to unsqueeze!
    emb = resnet(cropped_image.unsqueeze(0))

    # We'll compute the distance to each known embedding
    distances = []
    for known_emb, name in embedding_data:
        # Use torch.dist to compute the distance between
        # `emb` and the known embedding `known_emb`
        dist = torch.dist(emb, known_emb).item()
        distances.append((dist, name))

    # Find the name corresponding to the smallest distance
    dist, closest = min(distances)

    # If the distance is less than the threshold, set name to closest
    # otherwise set name to "Undetected"
    if dist < threshold:
        name = closest
    else:
        name= "Undetected"

    return name, dist    


# Fill in the label_face function
def label_face(name, dist, box, axis):
    """Adds a box and a label to the axis from matplotlib
    - name and dist are combined to make a label
    - box is the four corners of the bounding box for the face
    - axis is the return from fig.subplots()
    Call this in the same cell as the figure is created"""

    # Add the code to generate a Rectangle for the bounding box
    # set the color to "blue" and fill to False
    rect = plt.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color="blue"
    )
    axis.add_patch(rect)

    # Set color to be red if the name is "Undetected"
    # otherwise set it to be blue
    if name == "Undetected":
        color = 'red'
    else:
        color = 'blue'
    
    label = f"{name} {dist:.2f}"
    axis.text(box[0], box[1], label, fontsize="large", color=color)    


# Fill in the add_labels_to_image function
def add_labels_to_image(image):
    # This sets the image size
    # and draws the original image
    width, height = image.width, image.height
    dpi = 96
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = fig.subplots()
    axis.imshow(image)
    plt.axis("off")

    # Use the function locate_faces to get the individual face info
    faces = locate_faces(image)

    for box, prob, cropped in faces:
        # If the probability is less than 0.90,
        # It's not a face, skip this run of the loop with continue
        if prob < 0.90:
            continue
        
        # Call determine_name_dist to get the name and distance
        name, dist = determine_name_dist(cropped)

        # Use label_face to draw the box and label on this face
        label_face(name, dist, box, axis)

    return fig

# This file © 2024 by WorldQuant University is licensed under CC BY-NC-ND 4.0.

'''