"""
This is a simple script for reading and showing a video
"""

import numpy
import cv2 as cv
kh

# general information

# cap.get(0): CAP_PROP_POS_MSEC - Current position of the video file in milliseconds or video capture timestamp
# cap.get(1): CAP_PROP_POS_FRAMES - 0-based index of the frame to be decoded/captured next.
# cap.get(5): CAP_PROP_FPS - Frame rate of the video.

# CAP_PROP_FRAME_WIDTH - frame width
# CAP_PROP_FRAME_HEIGHT - frame higth
# CAP_PROP_FRAME_COUNT - number of all frames in the videos
#%%
video_sample = "../data/input/101_2Ihlw5FFrx4.mp4"

# Open the capture interface:
cap = cv.VideoCapture(video_sample)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")

while(cap.isOpened()):
    frame_reading_successful, frame = cap.read()
    if frame_reading_successful == True:
        cv.imshow('Frame',frame)
        # Press Q on keyboard to  exit
        if cv.waitKey(25) & 0xFF == ord('q'):
             break
    else:
        break
    # When everything done, release the video capture object
cap.release()
    # Closes all the frames
cv.destroyAllWindows()

#%%    
"""
This is a simple script for writing a video 
"""
video_sample = "../data/input/101_2Ihlw5FFrx4.mp4"
# Open the capture interface:
cap = cv.VideoCapture(video_sample)
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
    
    
clip_path =  "../data/output/"  
# Define the codec and create VideoWriter object.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 
fps = int(cap.get(5))

out = cv.VideoWriter(clip_path + 'outpy.avi',cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))


while(True):
  ret, frame = cap.read()

  if ret == True:   
    # Write the frame into the file 'output.avi'
    out.write(frame)
    # Display the resulting frame    
    cv.imshow('frame',frame)
    # Press Q on keyboard to stop recording
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  # Break the loop
  else:
    break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv.destroyAllWindows()


#%%

"""
This is a simple script for writing a video clip based on selected random onsets (one step earlier proper onsets must be selected instead of random onsets)
"""

import random
video_sample = "../data/input/101_2Ihlw5FFrx4.mp4"

cap = cv.VideoCapture(video_sample)
if (cap.isOpened()== False):
    print("Error opening video stream or file")

# reading video, detecting video duration and selecting onsets for random clips
    
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv.CAP_PROP_FPS))
video_duration_in_seconds = int(frame_count/fps)
clip_duration_in_seconds = 10
clip_duration_in_frames = clip_duration_in_seconds * fps
number_of_clips = int(video_duration_in_seconds/clip_duration_in_seconds)

min_onset = 0
max_onset = video_duration_in_seconds - clip_duration_in_seconds
random_onsets =  random.sample(range(min_onset, max_onset), number_of_clips)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4)) 


# writing random clips

clip_path =  '../data/output/'


# from Elias code

random_onsets = [10,35,75]

for clip_onset_time in random_onsets:
    
    clip_onset_frame = clip_onset_time * fps
    clip_offset_frame = clip_onset_frame + clip_duration_in_frames
    
    out_clip_name = clip_path + 'clip_' + str(clip_onset_time) + '.avi'
    out_clip = cv.VideoWriter(out_clip_name,cv.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))
    
    
    frame_position = clip_onset_frame
    
while(True):
  check, frame = cap.read()

  if check == True:   
    # Write the frame into the file 'output.avi'
    out.write(frame)
    # Display the resulting frame    
    cv.imshow('frame',frame)
    # Press Q on keyboard to stop recording
    if cv.waitKey(1) & 0xFF == ord('q'):
      break
  # Break the loop
  else:
    break



# next test this 

import cv2

cap = cv2.VideoCapture('video.avi')

# Get the frames per second
fps = cap.get(cv2.CAP_PROP_FPS) 

# Get the total numer of frames in the video.
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

frame_number = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number) # optional
success, image = cap.read()

while success and frame_number <= frame_count:

    # do stuff

    frame_number += fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = cap.read()