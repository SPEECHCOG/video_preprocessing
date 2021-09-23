
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
mylist = run_analysis.create_video_list()

wav_data, duration = run_analysis.load_video(mylist[2])
my_logmel = run_analysis.extract_logmel_features (wav_data)
scores, embeddings, log_mel_yamnet =  run_analysis.execute_yamnet (mylist[2])