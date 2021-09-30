
###############################################################################

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.compat.v1.Session(config=config)

###############################################################################
   
import config as cfg

split = cfg.paths['split']
datadir = cfg.paths['datadir']
outputdir = cfg.paths['outputdir']

audio_model = cfg.basic['audio_model']
visual_model = cfg.basic['visual_model']

layer_name = cfg.basic['layer_name']

dataset = cfg.basic['dataset']
video_settings = cfg.video_settings

###############################################################################

#from analysis import Analysis

#run_analysis = Analysis( audio_model,dataset, datadir,outputdir,split, video_settings )

from vfe import VisualFeatureExtractor

vgg_feature_extractor = VisualFeatureExtractor( visual_model,layer_name , dataset, datadir,outputdir, split, video_settings )
vgg_feature_extractor()


import pickle
with open("/worktmp/khorrami/project_5/video/data/youcook2/output/test/0/af", 'rb') as handle:
    b = pickle.load(handle)
    
# run_analysis.split
import pickle
with open("/worktmp/khorrami/project_5/video/data/youcook2/output/test/0/vf_resnet152", 'rb') as handle:
    c = pickle.load(handle)