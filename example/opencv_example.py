"""
This is a simple script for writing images of specific onsets within a clip
"""

import numpy
import cv2 as cv

"""
# cv2 cheat sheet

# cap.get(0): CAP_PROP_POS_MSEC - Current position of the video file in milliseconds or video capture timestamp
# cap.get(1): CAP_PROP_POS_FRAMES - 0-based index of the frame to be decoded/captured next.
# cap.get(5): CAP_PROP_FPS - Frame rate of the video.

# CAP_PROP_FRAME_WIDTH - frame width
# CAP_PROP_FRAME_HEIGHT - frame higth
# CAP_PROP_FRAME_COUNT - number of all frames in the videos

"""
#%%

"""
This is a simple script for writing a video clip based on selected random onsets (one step earlier proper onsets must be selected instead of random onsets)
"""

video_sample = "../data/input/101_YSes0R7EksY.mp4"
image_path = "../data/output/"
cap = cv.VideoCapture(video_sample)
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# reading video, detecting video duration and selecting onsets for random clips
    
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv.CAP_PROP_FPS))
video_duration_in_seconds = int(frame_count/fps)
clip_duration_in_seconds = 10
clip_duration_in_frames = clip_duration_in_seconds * fps


random_onsets =  [ 10, 30, 60]
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 

for onset in random_onsets:
    for i in range(10):
        
        ms =  (onset + i ) * 1000
        cap.set(cv.CAP_PROP_POS_MSEC, ms)      # Go to the ms niliseconds. position
        ret,frame = cap.read()                   # Retrieves the frame at the specified second
        cv.imwrite(image_path + str(onset) + "_" + str(i) + ".jpg", frame)          # Saves the frame as an image
        # cv.imshow("Frame Name",frame)           # Displays the frame on screen
        # cv.waitKey()                       


