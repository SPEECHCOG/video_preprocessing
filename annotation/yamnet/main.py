
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
yamnet_settings = cfg.yamnet_settings

###############################################################################

from analysis import Analysis

run_analysis = Analysis( audio_model,dataset, datadir, outputdir,split,  yamnet_settings )
run_analysis()

# # import pickle
# import json
# with open("/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2/annotations/youcookii_annotations_test_segments_only.json", 'rb') as handle:
#     b = json.load(handle)
# database = b['database']

import pickle
with open("/worktmp/khorrami/project_5/video/features/ouput/youcook2/ann-based/validation/0/af", 'rb') as handle:
    af = pickle.load(handle)

# b = {}    
# for video_name, onset_dict in b2.items():
#     b[video_name] = onset_dict
# all_n = []
# for fnum, video_name in b2.items():
#     all_n.append(fnum)




    