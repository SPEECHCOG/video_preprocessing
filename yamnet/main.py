
###############################################################################

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=config)

###############################################################################
   
import config as cfg

split = cfg.paths['split']
datadir = cfg.paths['datadir']
outputdir = cfg.paths['outputdir']
exp_name = cfg.paths['exp_name']

audio_model = cfg.basic['audio_model']
dataset = cfg.basic['dataset']
save_wavs = cfg.basic["save_wavs"]
rsd = cfg.basic['run_speech_detection']
yamnet_settings = cfg.yamnet_settings

###############################################################################

from analysis import Analysis

run_analysis = Analysis( audio_model,dataset, datadir,outputdir,split, save_wavs, yamnet_settings, rsd , exp_name)
run_analysis.save_wav_clips()

# import pickle
# with open("/worktmp2/hxkhkh/current/video/features/youcook2/ann-based/errors/training_errors", 'rb') as handle:
#     b = pickle.load(handle)

# import pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/output/yamnet-based/exp3/test/137/af", 'rb') as handle:
#     b2 = pickle.load(handle)
    
# onsets1 = b1['onsets_second']
# onsets2 = b2['onsets_second']

# import pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/output/train1/train_errors", 'rb') as handle:
#     b1 = pickle.load(handle)

# b = {}    
# for video_name, onset_dict in b2.items():
#     b[video_name] = onset_dict
# all_n = []
# for fnum, video_name in b2.items():
#     all_n.append(fnum)

# # scanning the signal (yamnet output scores)

# scanned_speech = []
# len_silde_window = 21    
# for counter in  range(len(speech_segments)):
#     print(counter)
#     slide_window_temp = speech_segments[counter:counter + len_silde_window]
#     speech_portion = sum(slide_window_temp)
#     scanned_speech.append(speech_portion)
#     print(speech_portion)
    
# import numpy
# initial_seq = [onset>= 17 for onset in scanned_speech]
# initial_seq = numpy.multiply(initial_seq,1)
# from matplotlib import pyplot as plt
# plt.plot(speech_segments)
# plt.plot(scanned_speech)
# plt.plot(initial_seq)

# # greedy search

# accepted_overlap_len = 10 # almost 5 seconds
# skip_len = 21 - accepted_overlap_len

# import copy
# updated_seq = copy.copy(initial_seq)

# for counter, value in enumerate(updated_seq):
#     if value==1:
#         updated_seq[counter+ 1: counter + skip_len] = 0
            
# plt.plot(updated_seq)
# sum(updated_seq)
# accepted_onsets = [counter for counter,value in enumerate(updated_seq) if value==1]















