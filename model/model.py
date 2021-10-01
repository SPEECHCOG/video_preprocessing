import os
import pickle
import numpy

import scipy.spatial as ss


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization, dot
from tensorflow.keras.layers import  MaxPooling1D,  Conv1D
from tensorflow.keras.optimizers import Adam

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

class Net():
    def __init__(self,visual_model, layer_name, featuredir, outputdir, split , feature_settings):
        self.visual_model = visual_model
        self.featuredir = featuredir
        self.outputdir = outputdir
        self.split = split
        
        self.feature_settings = feature_settings     
        self.clip_length_seconds = self.feature_settings ["clip_length_seconds"]
        self.visual_model = visual_model
        self.layer_name = layer_name
         
        self.video_name = ''
        self.folder_name = ''
        self.feature_path = ''   
        self.dict_errors = {}
        
    def load_dict_onsets (self):       
        input_name =  self.featuredir + self.split + '_onsets'  
        with open(input_name, 'rb') as handle:
            dict_onsets = pickle.load(handle)
        return dict_onsets        


    def load_af (self):       
        af_file = os.path.join(self.feature_path , 'af')   
        with open(af_file, 'rb') as handle:
            af = pickle.load(handle)           
        return af    

    def load_vf (self):       
        vf_file = os.path.join(self.feature_path , 'vf_' + self.visual_model)   
        with open(vf_file, 'rb') as handle:
            vf = pickle.load(handle)           
        return vf   
    
    def update_error_list (self):
        self.dict_errors[self.video_name] = self.folder_name
          
    
    def get_audio_features (self):             
        dict_onsets = self.load_dict_onsets ()        
        af_all = []                
        for video_name, value in dict_onsets.items():             
            self.video_name = video_name
            self.folder_name = value['folder_name']            
            if len(value['onsets']) == 0:
                self.update_error_list()
            else:
                self.feature_path = os.path.join(self.featuredir , self.split ,  str(self.folder_name))      
                af = self.load_af()            
                logmel = af['logmel40']            
                af_all.extend(logmel)
        audio_features = numpy.array(af_all)
        return audio_features
    
    def get_visual_features (self):
        dict_onsets = self.load_dict_onsets ()
        vf_all = []       
        for video_name, value in dict_onsets.items():             
            self.video_name = video_name
            self.folder_name = value['folder_name']                                  
            self.feature_path = os.path.join(self.featuredir , self.split ,  str(self.folder_name))      
            vf = self.load_vf()
            resnet = vf['resnet152_avg_pool']  
            vf_all.extend(resnet) 
        visual_features = numpy.array(vf_all)
        return visual_features  

        
        
    def build_audio_model (self, Xshape):     
        dropout_size = 0.3
        activation_C='relu'
    
        audio_sequence = Input(shape=Xshape) #(1000, 40)
                     
        forward2 = Conv1D(64,11,padding="same",activation=activation_C,name = 'conv2')(audio_sequence)
        dr2 = Dropout(dropout_size)(forward2)
        bn2 = BatchNormalization(axis=-1)(dr2)         
        pool2 = MaxPooling1D(3,strides = 2, padding='same')(bn2)
          
        forward3 = Conv1D(256,17,padding="same",activation=activation_C,name = 'conv3')(pool2)
        dr3 = Dropout(dropout_size)(forward3)
        bn3 = BatchNormalization(axis=-1)(dr3)       
        pool3 = MaxPooling1D(3,strides = 2,padding='same')(bn3)
          
        forward4 = Conv1D(512,17,padding="same",activation=activation_C,name = 'conv4')(pool3)
        dr4 = Dropout(dropout_size)(forward4)
        bn4 = BatchNormalization(axis=-1)(dr4) 
        pool4 = MaxPooling1D(3,strides = 2,padding='same')(bn4)
           
        forward5 = Conv1D(1024,17,padding="same",activation=activation_C,name = 'conv5')(pool4)
        dr5 = Dropout(dropout_size)(forward5)
        bn5 = BatchNormalization(axis=-1,name='audio_branch')(dr5) 
        pool5 = MaxPooling1D(500,padding='same')(bn5)
        
        out_audio_channel = Reshape([pool5.shape[2]],name='reshape_audio')(pool5) 
        audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
        #audio_model.summary()
        return audio_sequence , out_audio_channel , audio_model
        
    def build_visual_model (self, Yshape):
        
        dropout_size = 0.3
        visual_sequence = Input(shape=Yshape) #(10,2048)
        #visual_sequence = BatchNormalization(axis=-1, name = 'bn0_visual')(visual_sequence)
        
        #resh0 = Reshape([1, visual_sequence.shape[1],visual_sequence.shape[2]],name='reshape_visual')(visual_sequence) 
        forward_visual = Conv1D(1024,3,strides=1,padding = "same", activation='relu', name = 'conv_visual')(visual_sequence)
        dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
        bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)
        #resh1 = Reshape([bn_visual.shape[2],bn_visual.shape[3]],name='reshape_visual')(bn_visual) 
       
        pool_visual = MaxPooling1D(10,padding='same')(bn_visual)
        
        out_visual_channel = Reshape([pool_visual.shape[2]],name='reshape_visual')(pool_visual)        
        visual_model = Model(inputs= visual_sequence, outputs = out_visual_channel )
        #visual_model.summary()
        return visual_sequence , out_visual_channel , visual_model

        
    def build_network(self):
        
        audio_features = self.get_audio_features()       
        Xshape = numpy.shape(audio_features)[1:]
        visual_features = self.get_visual_features()        
        Yshape = numpy.shape(visual_features)[1:]
        
        audio_sequence , out_audio_channel , audio_model = self.build_audio_model (Xshape)
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model (Yshape)  
        
        # post-processing for Audio and Image channels (Dense + L2 norm layers)
        
        dnA = Dense(512,activation='linear',name='dense_audio')(out_audio_channel) 
        normA = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(dnA)
        
        dnI = Dense(512,activation='linear',name='dense_visual')(out_visual_channel) 
        normI = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(dnI)
      
        # combining audio and visual channels             
        A = normA
        I = normI
        
        mapIA = dot([I,A],axes=-1,normalize = True,name='dot_matchmap') 
        
        
        final_model = Model(inputs=[visual_sequence, audio_sequence], outputs = mapIA )
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs= I, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=audio_sequence,outputs= A, name='visual_embedding_model')
        final_model.summary()
        
        
        final_model.compile(loss=triplet_loss, optimizer= Adam(lr=1e-04))
        
        Ydata_triplet, Xdata_triplet, bin_triplet = prepare_triplet_data (audio_features , visual_features)
        
        # 15% validation
        y, x, bt = Ydata_triplet[0:72000], Xdata_triplet[0:72000], bin_triplet[0:72000]
        ye, xe, bte = Ydata_triplet[72000:84000], Xdata_triplet[72000:84000], bin_triplet[72000:84000]
        
        final_model.evaluate([ye,xe], bte, batch_size=120)
        final_model.fit([y,x], bt, shuffle=False, epochs=30, batch_size=120)  
        final_model.evaluate([ye,xe], bte, batch_size=120)
        
        yt = ye[::3]
        xt = xe[::3]      
        visual_embeddings = visual_embedding_model.predict(yt)     
        audio_embeddings = audio_embedding_model.predict(xt)   
        ########### calculating Recall@10                    
        poolsize =  1000
        number_of_trials = 10
        number_of_samples = len(xt)
        recall_av_vec = calculate_recallat10( audio_embeddings, visual_embeddings, number_of_trials,  number_of_samples  , poolsize )          
        recall_va_vec = calculate_recallat10( visual_embeddings , audio_embeddings, number_of_trials,  number_of_samples , poolsize ) 
        recall10_av = numpy.mean(recall_av_vec)/(poolsize)
        recall10_va = numpy.mean(recall_va_vec)/(poolsize)
        return recall10_av , recall10_va
        

    def __call__ (self):
        pass            
            
