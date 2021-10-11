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
featuredir_train = cfg.paths['featuredir_train']
featuredir_test = cfg.paths['featuredir_test']
outputdir = cfg.paths['outputdir']

audiochannel = cfg.basic['audiochannel']
loss = cfg.basic['loss']
audio_model = cfg.basic['audio_model']
visual_model = cfg.basic['visual_model']
layer_name = cfg.basic['layer_name']

feature_settings = cfg.feature_settings

###############################################################################

model_object = Net(audiochannel , loss, visual_model, layer_name, featuredir_train, featuredir_test, outputdir, split , feature_settings)

#feature_object = Features(audiochannel, visual_model, layer_name, featuredir_train, featuredir_test, outputdir, split , feature_settings)
# model_object.split = 'test'
# audio_features = model_object.get_audio_features()
# # kh
# visual_features = model_object.get_visual_features()
# # kh


# import pickle5 as pickle
# with open("/worktmp/khorrami/project_5/video/data/youcook2/features/yamnet/zp50/vf_train", 'wb') as handle:
#     pickle.dump(visual_features , handle,protocol=pickle.HIGHEST_PROTOCOL)
    
# with open("/worktmp/khorrami/project_5/video/data/youcook2/features/yamnet/zp50/vf_train", 'rb') as handle:
#     test = pickle.load(handle)
 
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
Sinitial = K.squeeze(K.batch_dot(out_audio, out_visual,axes=[-1,-1]), axis = 0)
print(Sinitial.shape)

margine = 0.001
# S = Sinitial - K.max(Sinitial, axis = 0)    
# S_diag =  tf.linalg.diag_part (S) 
# S_diag_margin = K.exp(S_diag - margine)
# Factor = K.exp(S_diag)
# S_sum = K.exp(S)# , axis = 0)
# S_other = S_sum - Factor
# Output = S_diag_margin / ( S_diag_margin + S_other) 
   
S = Sinitial - K.max(Sinitial, axis = 0)    
S_diag =  tf.linalg.diag_part (S) 
S_diag_margin = K.exp(S_diag - margine)
Factor = K.exp(S_diag)
S_sum = K.sum(K.exp(S) , axis = 0)
S_other = S_sum - Factor
Output = S_diag_margin / ( S_diag_margin + S_other) 
Y_hat1 = Output
I2C_loss = - K.mean ( K.log(Y_hat1 + 0.00001) , axis = 0)
#Y_hat2 =  margine_softmax(K.transpose(S) ,margine) 

s = K.eval(S)
s_diag =  K.eval(S_diag)
s_diag_margin = K.eval(S_diag_margin)
factor = K.eval(Factor)
s_sum = K.eval(S_sum)
s_other = K.eval(S_other)
output = K.eval(Output)
i2c = K.eval(I2C_loss)

import numpy
out_audio = numpy.random.randn(120, 1536)
out_visual = numpy.random.randn( 1536 , 120)
out_audio.shape
out_visual.shape

S = numpy.dot(out_audio, out_visual)
print(S.shape)

margine = 0.001    
S_diag =  numpy.diag (S) 
S_diag_margin = numpy.exp(S_diag - margine)


S_other = K.sum(K.exp(S) , axis = 0) - K.exp(S_diag)
Output = S_diag_margin / ( S_diag_margin + S_other) 
   

Y_hat1 = Output
I2C_loss = - K.mean ( K.log(Y_hat1) , axis = 0)
#Y_hat2 =  margine_softmax(K.transpose(S) ,margine) 
   
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

K.eval(K.epsilon())
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