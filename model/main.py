from model import Net
###############################################################################

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.compat.v1.Session(config=config)

###############################################################################
   
import config as cfg

split = cfg.paths['split']
featuredir = cfg.paths['featuredir']
outputdir = cfg.paths['outputdir']

audio_model = cfg.basic['audio_model']
visual_model = cfg.basic['visual_model']
layer_name = cfg.basic['layer_name']

feature_settings = cfg.feature_settings

# import pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/output/train/27/af", 'rb') as handle:
#     af = pickle.load(handle)


# import pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/output/test/1/vf_resnet152", 'rb') as handle:
#     vf = pickle.load(handle)   
    
    
# logmel = af['logmel40']  
# resnet = vf['resnet152_avg_pool']  
# # run_analysis.split
# import pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/output/test/1/vf_resnet152", 'rb') as handle:
#     c = pickle.load(handle)

###############################################################################

model_object = Net(visual_model, layer_name, featuredir, outputdir, split , feature_settings)

recall10_av , recall10_va = model_object.build_network()
# af_all = model_object.get_audio_features()

# vf_all = model_object.get_visual_features()
    
# dict_errors = model_object.dict_errors