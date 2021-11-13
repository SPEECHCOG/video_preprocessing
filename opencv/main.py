
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
exp_name = cfg.paths['exp_name']


visual_model_name = cfg.basic['visual_model_name']
visual_layer_name = cfg.basic['visual_layer_name']
save_images = cfg.basic['save_images']
save_visual_features = cfg.basic['save_visual_features']

video_settings = cfg.video_settings

###############################################################################

from analysis import Analysis

run_analysis = Analysis( visual_model_name, visual_layer_name , save_images, save_visual_features, datadir, outputdir, exp_name, split,  video_settings )
run_analysis()


# from vfe import VisualFeatureExtractor

# visual_feature_extractor = VisualFeatureExtractor( visual_model_name, visual_layer_name , datadir, outputdir, split, exp_name, video_settings )
# visual_feature_extractor()


# import pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/output/test/0/af", 'rb') as handle:
#     b = pickle.load(handle)
    
# # run_analysis.split
# import pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/output/test/1/vf_resnet152", 'rb') as handle:
#     c = pickle.load(handle)