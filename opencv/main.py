
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
dataset = cfg.basic['dataset']
video_settings = cfg.video_settings

###############################################################################

from analysis import Analysis
from vfe import VisualFeatureExtractor
#run_analysis = Analysis( audio_model,dataset, datadir,outputdir,split, video_settings )
#vgg_feature_extractor = VisualFeatureExtractor( audio_model,dataset, datadir,outputdir,split, video_settings )
#vgg_feature_extractor()


import pickle
with open("/worktmp/khorrami/project_5/video/data/youcook2/output/test/147/af", 'rb') as handle:
    b = pickle.load(handle)
    
# run_analysis.split