'''
 In this lesson, we'll learn how to work with image and 
 video data for object detection. We'll explore the specific 
 traffic dataset for this project. The focus will be on how 
 video data can be converted to images, understanding bounding 
 boxes for object classification, and using XML annotations to 
 represent those bounding boxes.
 
 
'''

#                 OBJECT DETECTION CODE
#      EXPLORING DATA
'''
 import sys
 import xml.etree.ElementTree as ET
 from collections import Counter
 from pathlib import Path

 import cv2
 import matplotlib.pyplot as plt
 from IPython.display import Video
 import torch
 import torchvision
 from torchvision import transforms
 from torchvision.io import read_image
 from torchvision.transforms.functional import to_pil_image
 from torchvision.utils import draw_bounding_boxes, make_grid

 # Path to dataset
 dhaka_image_dir = Path.home()/'data_images'/'train'

 print("Data directory:", dhaka_image_dir)

 # This data will contains two files .xml, .jpg
 # .xml files: These contain the annotations for the images.
 # .jpg files: These are the actual image files.
 # Each image typically has a corresponding XML file.
 # Below code show the files

 dhaka_files = list(dhaka_image_dir.iterdir())
 dhaka_files[-5:]

 # Below code give the no of file
 # in this directory
 file_extension_counts = Counter(Path(file).suffix for file in dhaka_files)

 for extension, count in file_extension_counts.items():
    print(f"Files with extension {extension}: {count}")

 OUTPUT:
 Files with extension .xml: 3003
 Files with extension .jpg: 2844
 Files with extension .JPG: 143
 Files with extension .png: 12
 Files with extension .jpeg: 2
 Files with extension .PNG: 2
 Files with extension : 1
'''

#   Separating images and bounding boxes data
'''
 images_dir = dhaka_image_dir / "images"
 annotations_dir = dhaka_image_dir / "annotations"

 images_dir.mkdir(exist_ok=True)
 annotations_dir.mkdir(exist_ok=True)

 for file in dhaka_files:
    if file.suffix.lower() in (".jpg", ".jpeg", ".png"):
        target_dir = images_dir
    elif file.suffix.lower() == ".xml":
        target_dir = annotations_dir
    file.rename(target_dir / file.name)

 images_files = list(images_dir.iterdir())[:3003]
 annotations_files = list(annotations_dir.iterdir())
 assert len(images_files) == len(annotations_files)
'''

#                  ANNOTATIONS
'''
 The annotations are the labels for the data. Each image has an 
 annotation that contains the coordinates and type of object for 
 each bounding box in a given image.

 Let's look at the structure of the annotations by loading the 
 first 25 lines of a file. The annotations are stored as XML
 which is a way to store structured documents. The <annotation> 
 tag is the root element, containing all the information about this 
 particular image annotation. The tags within store other information 
 such as <folder>. The most important tag for the current project 
 is the <object>. It describes an object detected in the image, 
 this associated image contains a "bus". 
 
 The tag <bndbox> is the bounding box information. There are the coordinates
 of a rectangle surrounding the bus in the image (in pixels): <xmin> is the left edge, 
 <ymin> is the top edge, <xmax> is the right edge, and <ymax> is 
 the bottom edge.

 xml_filepath = annotations_dir / "01.xml"
 !head -n 25 $xml_filepath

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


 The ElementTree (ET) module in Python can parse an XML file. 
 In XML, the root is the top-level element that contains all other 
 elements. The tag attribute contains the name of the element.

 tree = ET.parse(xml_filepath)
 root = tree.getroot()
 print(root.tag)

 OUTPUT: annotation

 The find method is used to locate the first occurrence 
 of a sub-element with a given tag. Let's find the width 
 and height of the image.

 width = int(root.find("size").find("width").text)
 height = int(root.find("size").find("height").text)
 print(f"image width: {width}  image height: {height}")

 OUTPUT: image width: 1200  image height: 800

 The `findall` method finds all occurrences of a sub-element 
 within a given tag. We can use that method to get all the 
 relevant data for the bounding boxes.

 bounding_boxes = []
 labels = []
 for obj in root.findall("object"):
    label = obj.find("name").text
    labels.append(label)
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
    bounding_boxes.append([xmin, ymin, xmax, ymax])

 for label, bounding_box in zip(labels, bounding_boxes):
    print(f"{label}: {bounding_box}")

 OUTPUT:

 bus: [833, 390, 1087, 800]
bus: [901, 284, 1018, 395]
bus: [909, 241, 1010, 287]
rickshaw: [761, 413, 832, 540]
rickshaw: [777, 364, 828, 409]
rickshaw: [120, 351, 177, 423]
rickshaw: [178, 340, 245, 419]
rickshaw: [551, 229, 581, 267]
rickshaw: [849, 211, 870, 240]
rickshaw: [854, 191, 872, 208]
rickshaw: [395, 250, 437, 286]
rickshaw: [626, 209, 653, 240]
motorbike: [863, 241, 882, 268]
car: [218, 252, 289, 285]
car: [495, 216, 531, 244]
car: [485, 201, 520, 219]
three wheelers (CNG): [254, 347, 298, 418]
three wheelers (CNG): [398, 307, 457, 353]
three wheelers (CNG): [240, 290, 303, 344]
'''

#   BOUNDING BOXES TO PYTORCH TENSORS AND DRAW BOUNDING BOXES ON IMAGE
'''
 bboxes_tensor = torch.tensor(bounding_boxes, dtype=torch.float)
 print(bboxes_tensor)

 image_path = images_dir/'01.jpg'
 image = read_image(str(image_path))
 print(image.shape)
 
 We can use the draw_bounding_boxes function from torchvision to add the 
 bounding boxes and labels to the image

 image = draw_bounding_boxes(
    image=image,
    boxes=bboxes_tensor,
    labels=labels,
    width=3,
    fill=False,
    font="arial.ttf",
    font_size=10,
 )

 Display the composite image with bounding boxes and labels.
 to_pil_image(image)

'''

#        OPENING VIDEO
'''
 The Video class from the IPython.display module is used to embed and 
 display a video within a Jupyter Notebook. This is particularly helpful 
 for visualizing multimedia content directly in your notebook.

 EX: Video("https://www.example.com/example.mp4", width=640, height=360)

 This project example

 video_dir = Path("data_video")
 video_name = "dhaka_traffic.mp4"
 video_path = video_dir/video_name

 print(video_path)

 Video(video_path, embed=True) #opens video in jupyter notebook

 We are going to capture still images from the video to make it easier 
 to draw bounding boxes.

 video_capture = cv2.VideoCapture(video_path)

 if not video_capture.isOpened():
    print("Error: Could not open video.")
 else:
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame rate: {frame_rate}")
    print(f"Total number of frames: {frame_count:,}")

 OUTPUT:
    Frame rate: 25.0
    Total number of frames: 9,333

 so we have to create 9,333 images

 Let's look the first frame.
 
 success, first_frame = video_capture.read()
 if success:
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    plt.title("First Frame")
    plt.axis("off")
    plt.show()
 else:
    print("Error: Could not read frame.")

 video_capture.set(cv2.CAP_PROP_POS_FRAMES, 100) # This capture 100th frame
 success, later_frame = video_capture.read()
 if success:
    plt.imshow(cv2.cvtColor(later_frame, cv2.COLOR_BGR2RGB))
    plt.title("First Frame")
    plt.axis("off")
    plt.show()
 else:
    print("Error: Could not read frame.")

 Create a directory for the frames using the pathlib syntax.

 frames_dir = video_dir / "extracted_frames"
 frames_dir.mkdir(exist_ok=True)

 frame_count = 0
 
 # The below code saves all the extracted frames to a directory 

 while True:
    success, frame = video_capture.read()
    if not success:
        break

    # Save frames at the frame_rate
    if frame_count % frame_rate == 0:
        frame_path = frames_dir / f"frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)

    frame_count += 1

 video_capture.release()

 # We can look at the frames we have extracted and saved using the `display_sample_images` 
 # function. This function reads the first 10 images and displays them in a grid on jupyter.

 def display_sample_images(dir_path, sample=5):
    image_list = []
    images = sorted(dir_path.iterdir())
    if images:
        sample_images = images[:sample]
        for sample_image in sample_images:
            image = read_image(str(sample_image))
            resize_transform = transforms.Resize((240, 240))
            image = resize_transform(image)
            image_list.append(image)
    grid = make_grid(image_list, nrow=5)
    image = to_pil_image(grid)
    return image


display_sample_images(frames_dir, sample=10)
'''
