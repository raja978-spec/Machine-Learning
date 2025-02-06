#     ULTRALYTICS
'''
This is the module that provides YOLO

import sys
from collections import Counter
from pathlib import Path

import PIL
import torch
import torchvision
import ultralytics
from IPython.display import Video
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from torchvision.utils import make_grid

# Here
from ultralytics import YOLO
'''

#  Image detection with YOLO

'''
 For this project we use yolo version 8
 yolo = YOLO(task="detect", model="yolov8s.pt")

 What classes can this pretrained model detect? That's stored in yolo.names.

 yolo.names

 OUTPUT:

 {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 ...more 74 classes...}

 so it can detect 80 objects

 class_assigned_to_23 = yolo.names[23]
 print(f"{class_assigned_to_23} corresponds to 23")

 OUTPUT: 
 giraffe corresponds to 23

 # check if a object doesn't have in yolo
 # below code checks if ambulance is not in yolo

 "ambulance" not in yolo.names.values()

 # In a later lesson, we'll retrain the YOLO model to include 
 # the missing classes. For this lesson, we are OK with what's 
 # already provided. We are most interested in the first 13 classes. 
 # Those classes are objects often found in traffic.

 #Let's use the YOLO model to identify objects in one frame of 
 # our video data. We'll use Path provided by pathlib.

 data_dir = Path("data_video", "extracted_frames")
 image_path = data_dir / "frame_1050.jpg"

 result = yolo(image_path)

 What is result?

 print(f"Type of result: {type(result)}")
 print(f"Length of result: {len(result)}")

 OUTPUT:
 Type of result: <class 'list'> 
 Length of result: 1

 There's another way to use the YOLO model. It's to use the object's .predict 
 method. The advantage is that it's clearer what we're doing 
 and allows us to overwrite any default values when predicting. 
 
 For example, we can control the confidence value of the resulting 
 bounding boxes. Let's use the .predict method and specify a 50% 
 threshold for the bounding box and save the results to disk as 
 a text file.

 result = yolo.predict(image_path, conf=0.5, save=True, save_txt=True)

 The results are contained in the created runs directory. And the result
 will returns the summary of what happend during the prediction.

 OUTPUT:

 image 1/1 /app/data_video/extracted_frames/frame_1050.jpg: 384x640 1 person, 
 1 bicycle, 3 cars, 2 buss, 10.9ms
 Speed: 2.6ms preprocess, 10.9ms inference, 
 1.4ms postprocess per image at shape (1, 3, 384, 640)
 Results saved to runs/detect/predict
 1 label saved to runs/detect/predict/labels

 # result[0] contains a special object with the results of the 
 # prediction stored as attributes.

 results[0]

 OUTPUT:
 
 ultralytics.engine.results.Results object with attributes:

 boxes: ultralytics.engine.results.Boxes object
 keypoints: None
 masks: None
 names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',...}
 obb: None
 orig_img: array([[[215, 240, 236],
        [218, 241, 237],
        [220, 241, 238],
        ...
 orig_shape: (360, 640)
 path: '/app/data_video/extracted_frames/frame_1050.jpg'
 probs: None
 save_dir: 'runs/detect/predict'
 speed: {'preprocess': 2.6040077209472656, 'inference': 10.933637619018555, 
 'postprocess': 1.367807388305664}

 .boxes contains the data for the bounding boxes. 
 These bounding boxes are the main things we want 
 from object detection. These boxes are then used to 
 create a box around the detected objects.

 result[0].boxes

 OUTPUT:

 cls: tensor([2., 2., 2., 1., 5., 5., 0.], device='cuda:0')
 conf: tensor([0.9039, 0.8548, 0.8375, 0.8323, 0.8061, 0.6084, 0.5534], device='cuda:0')
 data: tensor([[4.3695e+02, 1.6984e+02, 5.4202e+02, 2.3865e+02, 9.0395e-01,
 id: None
 is_track: False
 orig_shape: (360, 640)
 shape: torch.Size([7, 6])
 xywh: tensor([[489.4863, 204.2482, 105.0626,  68.8101],
        [240.4418, 202.1921, 167.6509, 100.2223],
        [576.7197, 206.7732,  74.0134,  46.4463],
        [362.5140, 2
 xywhn: tensor([[0.7648, 0.5674, 0.1642, 0.1911],
        [0.3757, 0.5616, 0.2620, 0.2784],
 xyxy: tensor([[4.3695e+02, 1.6984e+02, 5.4202e+02, 2.3865e+02], 
 xyxyn: tensor([[6.8274e-01, 4.7179e-01, 8.4690e-01, 6.6293e-01],
 
 print(result[0].boxes.cls) # Retruns the detected objects tensor
 print(f"Number of objects detected: {len(result[0].boxes.cls)}")

 OUTPUT:
 In this 2, 2, 2, 1 represents car, people, bike the detected object's
 class

 tensor([2., 2., 2., 1., 5., 5., 0.], device='cuda:0')
 Number of objects detected: 7

 # Now let's see what the objects we detected. The keys of yolo.names are 
 # integers so we'll need to cast the floats in result[0].boxes.cls to integers.

 object_counts = Counter([yolo.names[int(cls)] for cls in result[0].boxes.cls])
 object_counts

 OUTPUT:
 Counter({'car': 3, 'bus': 2, 'bicycle': 1, 'person': 1})

 # Determine the most common class and the number of times 
 # it was detected in frame_2575.jpg.

 object_counts_task = Counter([yolo.names[int(cls)] for cls in result_task[0].boxes.cls])

 most_common_class, count_of_class = object_counts_task.most_common(n=1)[0]
 print(f"Most common class: {most_common_class}")
 print(f"Number of detected {most_common_class}: {count_of_class}")

 OUTPUT:
 Most common class: car
 Number of detected car: 6

 # Another important attribute is .conf which has the 
 # confidence of the detected bounding boxes. The confidence is 
 # stored in a PyTorch tensor. We should expect this tensor's 
 # length to match the number we saw earlier.

 print(result[0].boxes.conf) # gives list of confidence for each detected object
 print(f"Number of objects detected: {len(result[0].boxes.conf)}")

 OUTPUT:
 tensor([0.9039, 0.8548, 0.8375, 0.8323, 0.8061, 0.6084, 0.5534], device='cuda:0')
 Number of objects detected: 7

 # When calling .predict, we set the confidence threshold to 50%. 
 # That is why all values in the confidence tensor is 
 # greater than 0.5. How many of the bounding boxes have a 
 # confidence value greater than 75%? For frame frame_1050.jpg,
 #  that would be:

 number_of_confident_objects = (result[0].boxes.conf > 0.75).sum().item()
 print(f"Number of objects detected with 50% confidence: {number_of_confident_objects}")

 OUTPUT:
 Number of objects detected with 50% confidence: 5

 #.orig_shape is just the original shape of the input. 
 # The attribute is_track indicates whether object tracking has 
 # been turned on. 

 # .xywh is a tensor with four columns for each row. 
 # Each row represents one box. The first and second column 
 # is the x and y coordinates of the top-left corner of the box, 
 # respectively. The third and fourth columns are width and height, 
 # respectively.

 result[0].boxes.xywh

 OUTPUT:

 tensor([[489.4863, 204.2482, 105.0626,  68.8101],
        [240.4418, 202.1921, 167.6509, 100.2223],
        [576.7197, 206.7732,  74.0134,  46.4463],
        [362.5140, 234.9969, 168.0120, 114.2923],
        [ 38.6910, 112.8627,  76.4582, 205.5199],
        [581.2806, 161.1671,  65.5864,  48.8184],
        [341.0811, 179.0009,  60.1201, 150.9053]], device='cuda:0')

 The output returns 7 row's box's size which is the
 detected object have, fist two column is x and y
 third and fourth column is w and h

 # .xywhn is very similar to .xywh but these coordinates have 
 # been normalized by the image size. We can remind ourselves 
 # of the original shape with .orig_shape.

 result[0].orig_shape

 OUTPUT:
 (360, 640)

 This means the image is 360 pixels high and 640 pixels wide. 
 Let's examine one row of the normalized bounding box.

 result[0].boxes.xywhn[0]

 OUTPUT: tensor([0.7648, 0.5674, 0.1642, 0.1911], device='cuda:0')

 #The third provided bounding box form is .xyxy. This form 
 # contains two coordinates, the (x, y) coordinate for the 
 # top left corner and the (x, y) coordinate of the bottom 
 # right corner.

 The last form is .xyxyn which is the normalized form of .xyxy.
 result[0].boxes.xyxyn

 # .save_dir is just the location where we've saved 
 # the resulting bounding boxes. We'll use the method 
 # exists of a Path object to make sure the location 
 # actually exists.

 location_of_results = Path(result[0].save_dir)

 print(f"Results saved to {location_of_results}")
 location_of_results.exists()

 #.speed gives the time taken to process input
 result[0].speed

 OUTPUT:

 {'preprocess': 2.508401870727539,
 'inference': 7.887125015258789, # prediction part
 'postprocess': 1.378774642944336 # cleaning up files time
 }

 print(f"Total time in milliseconds: {sum(result[0].speed.values())}")

 # By saving our results, we've created an image file with 
 # the bounding boxes drawn in.
 # Opens the saved image by the model
 Image.open(location_of_results / "frame_1050.jpg")

 # The bounding boxes were saved as a text file. that can be read
 # we told the store the txt file by saying save_txt=True
 with (location_of_results / "labels" / "frame_1050.txt").open("r") as f:
    print(f.read())

 OUTPUT:
 2 0.764822 0.567356 0.16416 0.191139
 2 0.37569 0.561645 0.261955 0.278395
 2 0.901125 0.57437 0.115646 0.129017
 1 0.566428 0.652769 0.262519 0.317479
 5 0.0604546 0.313508 0.119466 0.570889
 5 0.908251 0.447686 0.102479 0.135607
 0 0.532939 0.497225 0.0939376 0.419181

 1st column is the label(car,bike) other columns are
 normalized x,y,w,h
'''

#  Using YOLO on Multiple Images and Video Source¶
'''
def display_sample_images(dir_path, sample=5):
    dir_path = Path(dir_path) if isinstance(dir_path, str) else dir_path

    image_list = []
    # Sort the images to ensure they are processed in order
    images = sorted(dir_path.glob("*.jpg"))
    if not images:
        return None

    # Iterate over the first 'sample' images
    for img_path in images[:sample]:
        img = read_image(str(img_path))
        resize_transform = transforms.Resize((240, 240))
        img = resize_transform(img)
        image_list.append(img)

    # Organize the grid to have 'sample' images per row
    Grid = make_grid(image_list, nrow=5)
    # Convert the tensor grid to a PIL Image for display
    img = torchvision.transforms.ToPILImage()(Grid)
    return img

 images_path = list(data_dir.iterdir())[:25]
 images_path_task = images_path[-10:]

 print(f"Number of frames in list: {len(images_path_task)}")
 images_path_task

 # We'll once again use yolo.predict but this time we'll 
 # make use of two additional arguments to control where 
 # the results are saved. By using project and name, 
 # the saved results will be in project/name.
 results = yolo.predict(
    images_path,
    conf=0.5,
    save=True,
    save_txt=True,
    project=Path("runs", "detect"),
    name="multiple_frames",
 )
 image = display_sample_images(results[0].save_dir, sample=25)
 image
'''

#    USE YOLO ON VIDEOS
'''
 Now let's try to use YOLO on a video source instead of 
 the frames extracted from a video. The cell below displays 
 the video.

 video_path = Path("data_video", "dhaka_traffic.mp4")
 Video(video_path)

 #To speed things up, we're going to cut our video and 
 #run YOLO against the truncated version. We'll use ffmpeg, 
 #a command line tool for video and audio editing. The part 
#  that controls the timestamps for truncation are the numbers that 
#  follow -ss and -to. 
 
#  The number after -ss is the starting timestamp and -to is the 
#  ending timestamp. The value data_video/dhaka_traffic_truncated.mp4 
#  is the path of the created file.
# the below code cuts the video from 0-39 sec and saved it on data_video/dhaka_traffic_truncated.mp4
# directory

# -y means yes to procced to do the cut
# -i means input file name

 !ffmpeg -ss 00:00:00 -to 00:00:30 -y -i $video_path -c copy data_video/dhaka_traffic_truncated.mp4

 
# that saved cutted video are used here to make yolo detection
 video_truncated_path = Path("data_video", "dhaka_traffic_truncated.mp4")
 Video(video_truncated_path)

 #To use YOLO on a video source, we just need to tell 
 #it the location of the video and set stream to True.

 results_video = yolo.predict(
    video_truncated_path,
    conf=0.5,
    save=True,
    stream=True,
    project=Path("runs", "detect"),
    name="video_source",
 )

 #Unlike before, the returned value of yolo.predict is 
 #a generator rather than a list. Detection happens only as 
 #we iterate over the generator, giving us control over 
 #when the actual computation takes place.

 for result in results_video:
    continue
    
 YOLO has a command line interface. This is great if we are working 
 with shell scripts. You can see how its usage is very similar to what 
 we saw earlier.

 NOTE: It works in linux
 !yolo task=detect mode=predict conf=0.5 model=yolov8s.pt source=$video_truncated_path project="runs/detect" name="command_line" > /dev/null
'''

