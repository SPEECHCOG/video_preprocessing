from train_validate_matchmap import Train_AVnet
# from inspection import Inspection
#from features import Features
###############################################################################

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.7
# sess = tf.compat.v1.Session(config=config)

###############################################################################

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config)
 
###############################################################################  
import config as cfg

training_config = cfg.training_config
model_config = cfg.model_config
feature_config = cfg.feature_config
###############################################################################

obj = Train_AVnet(model_config , feature_config, training_config)
[X1shape , X2shape, Yshape] = obj.get_input_shapes()
obj()


#i,w, preds  = obj.predict()
# visual_embedding_model, audio_embedding_model, final_model = obj.build_network( X1shape , X2shape , Yshape)




# import pickle5 as pickle   
# with open("/worktmp/khorrami/project_5/video/features/youcook2/yamnet-based/testing/0/af", 'rb') as handle:
#     af = pickle.load(handle)
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

    
