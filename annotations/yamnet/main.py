
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
run_analysis.save_wav_clips()

# import json
# with open("/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2/annotations/youcookii_annotations_trainval.json", 'rb') as handle:
#     b = json.load(handle)
# database = b['database']

import pickle
with open("/worktmp2/hxkhkh/current/video/data/youcook2/annotations/validation_clips", 'rb') as handle:
    af = pickle.load(handle)

# import csv 
# with open("/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2/features/feat_csv/train_frame_feat_csv/101/17v08qtr8UM/0004") as handle:
#     vf = csv.reader(handle)
#     rows = []
#     for row in vf:
#         rows.append(row)
    




    