#                    FFMPEG
'''
 It is a command line tool used to take frames from 
 a video

 EX:

 def cut_video(start_time, input_file, duration, output_file):
    """
    Cuts a portion of the video.

    :parameter input_file: Path to the input video file.
    :parameter output_file: Path to the output video file.
    :parameter start_time: Start time of the cut in seconds or in `HH:MM:SS` format.
    :parameter duration: Duration of the cut in seconds or in `HH:MM:SS` format.
    """
    command = [
        "ffmpeg",
        "-ss",
        str(start_time),
        "-i",
        input_file,
        "-t",
        str(duration),
        "-c",
        "copy",
        output_file,
    ]
    subprocess.run(command)


cut_video?

input_video = video_dir / video_name

print(input_video)

output_video_name = "output.mp4"

output_video = video_dir / output_video_name

print(output_video)

start_time = "00:00:00"  # Start at 00 seconds
duration = "00:01:00"  # Cut 1 minute

# Call cut_video function
cut_video(start_time, input_video, duration, output_video)

# Let's learn a bit more about our video data. We're curious 
# about things like the frame rate, total frame count, and 
# frame shape. To get this information, we'll use the opencv-python 
# library just like in the previous project (cv2 which we imported at 
# the top of the notebook).

# The first step is to create a video capture using cv2.VideoCapture and pass 
# in the path to our video.

video_capture = cv2.VideoCapture(output_video)

if not video_capture.isOpened():
    print("Error: Could not open video.")
else:
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame rate: {frame_rate}")
    print(f"Total number of frames: {frame_count}")

OUTPUT:

Frame rate: 25.0
Total number of frames: 150

# Now let's display the first frame. We can fetch the first frame of our 
# video capture by calling the read() method on it.

ret, first_frame = video_capture.read()

if ret:
    plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
    plt.title("First Frame")
    plt.axis("off")
    plt.show()
else:
    print("Error: Could not read frame.")

# Get the shape of the frame and the number of channels by 
# calling the shape attribute on the first_frame variable.

Fill in the missing code below that saves every fifth frame from the video.

interval = frame_rate * 0.20  # Extract every fifth frame from the video
frame_count = 0

print("Start extracting individual frames...")
while True:
    # read next frame from the video_capture
    ret, frame = video_capture.read()
    if not ret:
        print("Finished!")
        break  # Break the loop if there are no more frames

    # Save frames at every 'interval' frames
    if frame_count % interval == 0:
        frame_path = frames_dir / f"frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)

    frame_count += 1

video_capture.release()

OUTPUT:
Start extracting individual frames...
Finished!
'''