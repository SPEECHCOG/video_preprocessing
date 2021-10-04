
#import pickle  # for video env
import pickle5 as pickle # for myPython env
import numpy
import tensorflow as tf

import scipy.spatial as ss


from tensorflow.keras import backend as K


def triplet_loss(y_true,y_pred):    
    margin = 0.1
    penalty_factor = 1
    # penalty factor might speed up training or improve its quality 
    # since it asks the system to penalize the more unsimilar pairs more than less unsimilar ones
    # thus it adds the rule to the training procedure that e.g. image woman to speech "person" gets less penalty than to speech "dog"
    # margin and penalty factors can be changed during training
    # i.e margin can start from small values (0.1) and be increased gradually during training
    # penalty factor can start from 1 and be increased during training
    
    Sp = y_pred[0::3]
    Si = y_pred[1::3]
    Sc = y_pred[2::3]      
    return K.sum(penalty_factor * K.maximum(0.0,(Sc-Sp + margin )) + penalty_factor * K.maximum(0.0,(Si-Sp + margin )),  axis=0) 

def randOrderTriplet(n_t):
    random_order = numpy.random.permutation(int(n_t))
    random_order_X = numpy.random.permutation(int(n_t))
    random_order_Y = numpy.random.permutation(int(n_t))
    
    data_orderX = []
    data_orderY = []     
    for group_number in random_order:
        
        data_orderX.append(group_number)
        data_orderY.append(group_number)
        
        data_orderX.append(group_number)
        data_orderY.append(random_order_Y[group_number])
        
        data_orderX.append(random_order_X[group_number])
        data_orderY.append(group_number)
        
    return data_orderX,data_orderY

def make_bin_target (n_sample):
    target = []
    for group_number in range(n_sample):    
        target.append(1)
        target.append(0)
        target.append(0)
        
    return target


def normalizeX (dict_logmel, len_of_longest_sequence):
    number_of_audios = numpy.shape(dict_logmel)[0]
    number_of_audio_features = numpy.shape(dict_logmel[0])[1]
    X = numpy.zeros((number_of_audios ,len_of_longest_sequence, number_of_audio_features),dtype ='float32')
    for k in numpy.arange(number_of_audios):
       logmel_item = dict_logmel[k]
       logmel_item = logmel_item[0:len_of_longest_sequence]
       X[k,len_of_longest_sequence-len(logmel_item):, :] = logmel_item
    return X

def prepare_triplet_data (Xdata , Ydata):
    n_samples = len(Ydata)
    orderX,orderY = randOrderTriplet(n_samples)
    bin_triplet = numpy.array(make_bin_target(n_samples)) 
    Ydata_triplet = Ydata[orderY]
    Xdata_triplet = Xdata[orderX]
    return Ydata_triplet, Xdata_triplet, bin_triplet   



def calculate_recallat10( embedding_1,embedding_2, sampling_times, number_of_all_audios, pool):   
    recall_all = []
    recallat = 10  
    for trial_number in range(sampling_times):      
        data_ind = numpy.random.randint(0, high=number_of_all_audios, size=pool)       
        vec_1 = [embedding_1[item] for item in data_ind]
        vec_2 = [embedding_2[item] for item in data_ind]           
        distance_utterance = ss.distance.cdist( vec_1 , vec_2 ,  'cosine') # 1-cosine
       
        r = 0
        for n in range(pool):
            ind_1 = n #random.randrange(0,number_of_audios)                   
            distance_utterance_n = distance_utterance[n]            
            sort_index = numpy.argsort(distance_utterance_n)[0:recallat]
            r += numpy.sum((sort_index==ind_1)*1)   
        recall_all.append(r)
        del distance_utterance  
        
    return recall_all


def prepare_data (audio_features , visual_features):
 
    n_samples = len(audio_features) 
    random_order = numpy.random.permutation(int(n_samples))
    
    Ydata_rand = visual_features[random_order ]
    Xdata_rand = audio_features[random_order ]
 
    target = numpy.ones( n_samples)
    return Ydata_rand, Xdata_rand , target

def mms_loss_alignment(y_true , y_pred): 
    
    out_visual = y_pred [:,0,:]
    out_audio = y_pred [:,1,:]   
    
    out_visual = K.expand_dims(out_visual, 0)
    out_audio = K.expand_dims(out_audio, 0)
    print(out_visual.shape)
    
    S = K.squeeze( K.batch_dot(out_audio, out_visual,axes=[-1,-1]) , axis = 0)
    
    print(S.shape)
    # ......................................................
    P1 = K.softmax(S, axis = 0) #row-wise softmax
    P2 = K.softmax(S, axis = 1) #column-wise softmax
    
    P1 = P1 + 0.005
    P2 = P2 + 0.005
    
    Y_hat1 = tf.linalg.diag_part (P1)
    Y_hat2 = tf.linalg.diag_part (P2)
    
    # ......................................................
    
    I2C_loss = K.sum ( - (K.log(Y_hat1)) , axis = 0)
    C2I_loss = K.sum ( -  (K.log(Y_hat2)) , axis = 0) 
    
    loss = I2C_loss + C2I_loss
    
    print('loss shape')
    
    return loss 

    

def mms_loss(y_true , y_pred): 
    
    
    print('this is y_true')
    print(y_true.shape)
    
    
    print('this is y_pred')
    print(y_pred.shape)
    
    out_visual = y_pred [:,0,:]
    out_audio = y_pred [:,1,:]

    
    out_visual = K.expand_dims(out_visual, 0)
    out_audio = K.expand_dims(out_audio, 0)
 
    S = K.squeeze( K.batch_dot(out_audio, out_visual,axes=[-1,-1]) , axis = 0)
    
    # ...................................................... method 0
    # margine = 0.1
    # def margine_softmax(S ,margine):
        
    #     S_diag =  tf.linalg.diag_part (S) 
    #     factor = K.exp(S_diag + margine)
    #     output = 1 / ( 1 + (factor) * ( K.sum(K.exp(S) , axis = 0) - K.exp(S_diag)) )
    #     return output
    
    # Y_hat1 =  margine_softmax(S ,margine) + 0.005
    # Y_hat2 =  margine_softmax(K.transpose(S) ,margine) + 0.005
      
    # ...................................................... method 1
    P1 = K.softmax(S, axis = 0) #row-wise softmax
    P2 = K.softmax(S, axis = 1) #column-wise softmax
    
    P1 = P1 + 0.05
    P2 = P2 + 0.05
    
    Y_hat1 = tf.linalg.diag_part (P1)
    Y_hat2 = tf.linalg.diag_part (P2)
    
    # ......................................................
    
    I2C_loss = K.sum ( - (K.log(Y_hat1)) , axis = 0)
    C2I_loss = K.sum ( -  (K.log(Y_hat2)) , axis = 0) 
    
    loss = I2C_loss + C2I_loss
    
    print('loss shape')
    
    return loss
