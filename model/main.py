from model import Net
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
outputdir = cfg.paths['outputdir']

audiochannel = cfg.basic['audiochannel']
loss = cfg.basic['loss']
audio_model = cfg.basic['audio_model']
visual_model = cfg.basic['visual_model']
layer_name = cfg.basic['layer_name']

feature_settings = cfg.feature_settings
# file = '/tuni/groups/3101050_Specog/corpora/youcook2_dataset/clip_annotations/youcook2/train/101_2Ihlw5FFrx4_clip0_step0.pickle'
# import pickle
# with open(file, 'rb') as handle:
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

model_object = Net(audiochannel , loss, visual_model, layer_name, featuredir, outputdir, split , feature_settings)

recall10_av , recall10_va = model_object()



# af_all = model_object.get_audio_features()

# vf_all = model_object.get_visual_features()
    
# dict_errors = model_object.dict_errors


# import json
# file2 = open('/worktmp/khorrami/project_5/video/data/youcook2/youcookii_annotations_trainval.json','r')
# text2 = json.load(file2)

from keras import backend as K
import tensorflow as tf
out_audio = K.random_normal(shape=( 120, 1536), seed=42)
out_visual = K.random_normal(shape=( 120, 1536), seed=62)
out_audio.shape
out_visual.shape

A = K.eval(out_audio)
I = K.eval(out_visual)

out_audio = K.expand_dims(out_audio, 0)
out_visual = K.expand_dims(out_visual, 0)
target = tf.eye(120)
S1 = K.squeeze(K.batch_dot(out_audio, out_visual,axes=[-1,-1]), axis = 0)
S1.shape
#...................................................... method 0
margine = - 0.2

        
S_diag =  tf.linalg.diag_part (S1) 
Factor = K.exp(S_diag + margine)
Y_hat1 =  (1 / ( 1 + (Factor) * ( K.sum(K.exp(S1) , axis = 0) - K.exp(S_diag)) ) ) 
        
y_hat1 = K.eval(Y_hat1)    

Loss =  - (K.log(Y_hat1)) 

loss = K.eval(Loss)
import numpy
test = - numpy.log(y_hat1)
#...................................................... method 1


P1 = K.softmax(S1, axis = 1) #row-wise softmax
P1 = P1 + 0.1
Y_hat = tf.linalg.diag_part (P1)

p1 = K.eval(P1)
y_hat = K.eval(Y_hat)

Loss = K.sum ( -  (K.log(Y_hat)) , axis = 0)
loss= K.eval(Loss)

import numpy
test_row = numpy.sum(p1, axis = 0)
test_col = numpy.sum(p1, axis = 1)
# Y_hat = P1
# I2C_loss = K.sum (K.sum ( - target * (K.log(Y_hat)) , axis = 0))

# S2 = K.transpose(S1)
# P2 = K.softmax(S2, axis = -1) #row-wise softmax
# Y_hat = P2
# C2I_loss = K.sum (K.sum ( - target * (K.log(Y_hat)) , axis = 0) )

# loss = I2C_loss + C2I_loss


#  # ...................................................... method 1
 
# S1 = K.squeeze(K.batch_dot(out_audio, out_visual,axes=[-1,-1]), axis = 0)

# S_eye = K.sum (target*S1 , axis = 0)
# S_masked = S1-S_eye
# S_norm = K.sum(K.exp(S_masked),  axis=0)
# I2C_loss = K.sum (K.log ( K.exp(S_eye) / ( K.exp(S_eye) + S_norm ) ))

# S2 = K.transpose(S1)

# S_eye = K.sum (target*S2 , axis = 0)
# S_masked = S2-S_eye
# S_norm = K.sum(K.exp(S_masked),  axis=0)
# C2I_loss = K.sum (K.log ( K.exp(S_eye) / ( K.exp(S_eye) + S_norm ) ))

# loss = I2C_loss + C2I_loss

 # ...................................................... method 2
# I2C_loss = tf.nn.log_poisson_loss(tf.nn.log_softmax(S, dim=1), target)
# C2I_loss = tf.nn.log_poisson_loss(tf.nn.log_softmax(tf.transpose(S), dim=1), target)
# loss = I2C_loss + C2I_loss