import os
#import pickle  # for video env
import pickle5 as pickle # for myPython env
import numpy


from utils import triplet_loss, normalizeX, mms_loss, prepare_triplet_data, prepare_data, calculate_recallat10




from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization, dot
from tensorflow.keras.layers import  MaxPooling1D,AveragePooling1D,  Conv1D, Concatenate, ReLU, Add
from tensorflow.keras.optimizers import Adam




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
        
        
        #audio_features = numpy.array(af_all)
        # if normalization of logmels
        audio_features = normalizeX (af_all, 1000)
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

        
        
    def build_simple_audio_model (self, Xshape):     
        dropout_size = 0.3
        activation_C='relu'
    
        audio_sequence = Input(shape=Xshape) #(1000, 40)
                     
        forward2 = Conv1D(128,11,padding="same",activation=activation_C,name = 'conv2')(audio_sequence)
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
        audio_model.summary()
        return audio_sequence , out_audio_channel , audio_model

    def build_audio_model (self, Xshape):     
        dropout_size = 0.3
        activation_C='relu'
    
        audio_sequence = Input(shape=Xshape) #Xshape = (1000, 40)
          
        # layer 1
        x = audio_sequence
             
        x1 = Conv1D(128,9,strides = 2, padding="same")(audio_sequence)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1) # (1000, 128)
        
        x2 = Conv1D(128,9,strides = 1, padding="same")(x1)
        x2 = BatchNormalization(axis=-1)(x2)
        x2 = ReLU()(x2)
        
        downsample = Conv1D(128,9,strides = 2, padding="same")(audio_sequence)
        
        out = Add()([x1,downsample])
        # layer 2

        
        # layer 3

        
        # layer 4

        out_audio_channel  = out
        #out_audio_channel = Reshape([pool5.shape[2]],name='reshape_audio')(out) 
        audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
        audio_model.summary()
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
        visual_model.summary()
        return visual_sequence , out_visual_channel , visual_model

        
    def build_network(self):
        
        self.split = "train"
        audio_features_train = self.get_audio_features()            
        visual_features_train = self.get_visual_features()
        
        Xshape = numpy.shape(audio_features_train)[1:]        
        Yshape = numpy.shape(visual_features_train)[1:]
        
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
 
        self.split = "val"
        audio_features_val = self.get_audio_features()            
        visual_features_val = self.get_visual_features()
        
        audio_features = numpy.concatenate((audio_features_train, audio_features_val), axis = 0)
        visual_features = numpy.concatenate((visual_features_train, visual_features_val), axis = 0)
        del audio_features_train,audio_features_val ,visual_features_train, visual_features_val
        
        self.split = "test"
        audio_features_test = self.get_audio_features()            
        visual_features_test = self.get_visual_features()
        Ytest, Xtest, bin_val = prepare_triplet_data (audio_features_test , visual_features_test)

        
        for epoch in range(30):
            Ytrain, Xtrain, bin_train = prepare_triplet_data (audio_features , visual_features)
            final_model.fit([Ytrain,Xtrain], bin_train, shuffle=False, epochs=1, batch_size=120) 
            final_model.evaluate([Ytest,Xtest], bin_val, batch_size=120)
            del Ytrain,Xtrain,bin_train

            audio_embeddings = audio_embedding_model.predict(audio_features_test)    
            visual_embeddings = visual_embedding_model.predict(visual_features_test)     
            
            ########### calculating Recall@10                    
            poolsize =  1000
            number_of_trials = 10
            number_of_samples = len(audio_features_test)
            recall_av_vec = calculate_recallat10( audio_embeddings, visual_embeddings, number_of_trials,  number_of_samples  , poolsize )          
            recall_va_vec = calculate_recallat10( visual_embeddings , audio_embeddings, number_of_trials,  number_of_samples , poolsize ) 
            recall10_av = numpy.mean(recall_av_vec)/(poolsize)
            recall10_va = numpy.mean(recall_va_vec)/(poolsize)         
            del audio_embeddings, visual_embeddings
            print('............. results for retrieval ............ av and va ')
            print(epoch)
            print(recall10_av)
            print(recall10_va)
        return recall10_av , recall10_va


    def build_network_MMS(self):
        
        self.split = "train"
        audio_features_train = self.get_audio_features()            
        visual_features_train = self.get_visual_features()
        
        
        
        Xshape = numpy.shape(audio_features_train)[1:]        
        Yshape = numpy.shape(visual_features_train)[1:]
        
        audio_sequence , out_audio_channel , audio_model = self.build_audio_model (Xshape)
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model (Yshape)  
        
        # post-processing for Audio and Image channels (Dense + L2 norm layers)
        
        dnA = Dense(512,activation='linear',name='dense_audio')(out_audio_channel) 
        normA = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_audio')(dnA)
        
        dnI = Dense(512,activation='linear',name='dense_visual')(out_visual_channel) 
        normV = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='out_visual')(dnI)
      
        # combining audio and visual channels             
        A = Reshape([1 , normA.shape[1]])(normA) 
        V = Reshape([1 , normV.shape[1]])(normV) 
        s_output = Concatenate(axis=1)([V, A])
        
        final_model = Model(inputs=[visual_sequence, audio_sequence], outputs = s_output )
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs= normV, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=audio_sequence,outputs= normA, name='visual_embedding_model')
        final_model.summary()
        
        final_model.compile(loss=mms_loss, optimizer= Adam(lr=1e-04))
 
        self.split = "val"
        audio_features_val = self.get_audio_features()            
        visual_features_val = self.get_visual_features()
        
        audio_features = numpy.concatenate((audio_features_train, audio_features_val), axis = 0)
        visual_features = numpy.concatenate((visual_features_train, visual_features_val), axis = 0)
        del audio_features_train,audio_features_val ,visual_features_train, visual_features_val
        
        
        self.split = "test"
        audio_features_test = self.get_audio_features()            
        visual_features_test = self.get_visual_features()      
        Ytest, Xtest, bin_val = prepare_data (audio_features_test , visual_features_test)
        
        
        for epoch in range(30):
            Ydata, Xdata, bin_target = prepare_data (audio_features , visual_features)
            history = final_model.fit([Ydata, Xdata ], bin_target,  shuffle=False, epochs=1, batch_size=128)
            final_trainloss = history.history['loss'][0]
            final_model.evaluate([Ytest,Xtest], bin_val, batch_size=128)
        

        
        
            audio_embeddings = audio_embedding_model.predict(audio_features_test)    
            visual_embeddings = visual_embedding_model.predict(visual_features_test)     
            
            ########### calculating Recall@10                    
            poolsize =  1000
            number_of_trials = 10
            number_of_samples = len(audio_features_test)
            recall_av_vec = calculate_recallat10( audio_embeddings, visual_embeddings, number_of_trials,  number_of_samples  , poolsize )          
            recall_va_vec = calculate_recallat10( visual_embeddings , audio_embeddings, number_of_trials,  number_of_samples , poolsize ) 
            recall10_av = numpy.mean(recall_av_vec)/(poolsize)
            recall10_va = numpy.mean(recall_va_vec)/(poolsize)
            del audio_embeddings, visual_embeddings
            print('............. results for retrieval ............ av and va ')
            print(epoch)
            print(recall10_av)
            print(recall10_va)
            
        return recall10_av , recall10_va        

    def __call__ (self):
        pass            
            
