# video_preprocessing

This repository contains various video processing tasks for audio-video associative learning and alignment.



# Project Description

The project aims to study audio-visual self-supervised learning and alignment using small-scale video data.

# Model Description

The model consists of two parallel branches: one for processing visual data and one for processing audio data. The model is trained in a contrastive manner, contrasting between positive audiovisual pairs (simultaneous audiovisual frames) and negative pairs (irrelevant video frames and sound clips).

# Data

The data used in these experiments are instructional videos from YouTube, including "YouCook2" and "CrossTask".

The "download" folder contains scripts for downloading YouTube videos.

# Visual pre-processing

The "opencv" folder contains scripts for detecting and processing video frames.

# Audio pre-processing

The "yamnet" folder contains scripts for audio preprocessing.

# Annotations

The "annotations" folder contains scripts to read data annotations.
