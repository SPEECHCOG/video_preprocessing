
"""
This is a simple example for AVL model training on YOUCOOK2 data
"""



############################################################################### configuration file
paths = {  
  "split": "test",
  "featuredir": "../../data/youcook2/output/",
  "outputdir": "../../model/youcook2/"
}

basic = {
    "dataset": "YOUCOOK2",
    "audio_model" : "yamnet",
    "visual_model": "resnet152", 
    "layer_name": "avg_pool",
    "save_results" : True,
    "plot_results" : False
}

video_settings = {
  "audio_sample_rate": 16000,
  "clip_length_seconds" : 10,
}


############################################################################### main file

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.compat.v1.Session(config=config)

###############################################################################
   
#import config as cfg

split =paths['split']
featuredir = paths['featuredir']
outputdir = paths['outputdir']
audio_model = basic['audio_model']
visual_model = basic['visual_model']
layer_name = basic['layer_name']
clip_length_seconds = video_settings['clip_length_seconds']
dataset = basic['dataset']

############################################################################### model
