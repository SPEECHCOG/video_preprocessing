
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

run_analysis = Analysis( audio_model,dataset, datadir,outputdir,split, yamnet_settings )
run_analysis()

import pickle
with open("/worktmp/khorrami/project_5/video/data/youcook2/output/train_errors", 'rb') as handle:
    c = pickle.load(handle)

# import pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/output/train1/train_errors", 'rb') as handle:
#     b1 = pickle.load(handle)

# b = {}    
# for video_name, onset_dict in b2.items():
#     b[video_name] = onset_dict
# all_n = []
# for fnum, video_name in b2.items():
#     all_n.append(fnum)


    
    