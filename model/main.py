from model import Net
#from features import Features
###############################################################################

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.7
# sess = tf.compat.v1.Session(config=config)

###############################################################################

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.7
# sess = tf.Session(config=config)
 
###############################################################################  
import config as cfg

split = cfg.paths['split']
featuredir = cfg.paths['featuredir']
train_feature_type = cfg.paths['train_feature_type']
test_feature_type = cfg.paths['test_feature_type']
outputdir = cfg.paths['outputdir']

audiochannel = cfg.basic['audiochannel']
loss = cfg.basic['loss']
audio_model = cfg.basic['audio_model']
visual_model = cfg.basic['visual_model']
layer_name = cfg.basic['layer_name']

feature_settings = cfg.feature_settings

###############################################################################

model_object = Net(audiochannel , loss, visual_model, layer_name, featuredir,train_feature_type ,test_feature_type, outputdir, split , feature_settings)
recall10_av , recall10_va = model_object()

kh
x = model_object.get_audio_features()

#feature_object = Features(audiochannel, visual_model, layer_name, featuredir_train, featuredir_test, outputdir, split , feature_settings)
# model_object.split = 'train'
# audio_features = model_object.get_audio_features()
# kh
# visual_features = model_object.get_visual_features()
# # kh

# import json
# file2 = open('/worktmp/khorrami/project_5/video/data/youcook2/youcookii_annotations_trainval.json','r')
# text2 = json.load(file2)

# import pickle5 as pickle
# with open("", 'wb') as handle:
#     pickle.dump(visual_features , handle,protocol=pickle.HIGHEST_PROTOCOL)

import pickle5 as pickle   
with open("/worktmp/khorrami/project_5/video/data/youcook2/output/yamnet-based/exp2/test_onsets", 'rb') as handle:
    test = pickle.load(handle)
 
################################################################## testing MMS
# from keras import backend as K
# import tensorflow as tf

# out_audio = K.random_normal(shape=( 120, 1536), seed=42)
# out_visual = K.random_normal(shape=( 120, 1536), seed=62)
# out_audio.shape
# out_visual.shape

# A = K.eval(out_audio)
# I = K.eval(out_visual)

# out_audio = K.expand_dims(out_audio, 0)
# out_visual = K.expand_dims(out_visual, 0)
# target = tf.eye(120)
# Sinitial = K.squeeze(K.batch_dot(out_audio, out_visual,axes=[-1,-1]), axis = 0)
# print(Sinitial.shape)

# margine = 0.001 
   
# S = Sinitial - K.max(Sinitial, axis = 0)    
# S_diag =  tf.linalg.diag_part (S) 
# S_diag_margin = K.exp(S_diag - margine)
# Factor = K.exp(S_diag)
# S_sum = K.sum(K.exp(S) , axis = 0)
# S_other = S_sum - Factor
# Output = S_diag_margin / ( S_diag_margin + S_other) 
# Y_hat1 = Output
# I2C_loss = - K.mean ( K.log(Y_hat1 + 0.00001) , axis = 0)
# #Y_hat2 =  margine_softmax(K.transpose(S) ,margine) 


