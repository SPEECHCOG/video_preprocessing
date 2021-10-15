import os
#import pickle  # for video env
import pickle5 as pickle # for myPython env
import numpy
from scipy.io import savemat
from matplotlib import pyplot as plt

from utils import triplet_loss,  mms_loss,  prepare_data, preparX, preparY, calculate_recallat10 




from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization, dot, Softmax, Permute
from tensorflow.keras.layers import  MaxPooling1D,AveragePooling1D,  Conv1D, Concatenate, ReLU, Add, Multiply, GRU
from tensorflow.keras.optimizers import Adam




class Net():
    def __init__(self, audiochannel, loss, visual_model, layer_name, featuredir,train_feature_type ,test_feature_type, outputdir, split , feature_settings):
        self.audiochannel = audiochannel
        self.loss = loss
        self.visual_model = visual_model
        self.featuredir = featuredir
        self.train_feature_type = train_feature_type
        self.test_feature_type = test_feature_type
        self.outputdir = outputdir
        self.split = split

        
        self.feature_settings = feature_settings     
        self.zeropadd = self.feature_settings["zeropadd"]
        self.visual_model = visual_model
        self.layer_name = layer_name
         
        self.video_name = ''
        self.folder_name = ''
        self.feature_path = '' 
        
        self.dict_errors = {}
        self.av_all = []
        self.va_all = []
        
    def load_dict_onsets (self): 
        input_path = os.path.join(self.featuredir , self.featuretype , self.split)
        input_name =  input_path + '_onsets'  
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
                self.feature_path = os.path.join(self.featuredir , self.featuretype, self.split ,  str(self.folder_name))      
                af = self.load_af()            
                logmel_all = af['logmel40'] 
                logmel = logmel_all#[0:10]
                if self.featuretype == "ann-based":                  
                    af_all.append(logmel)
                elif self.featuretype == "yamnet-based":
                    af_all.extend(logmel)
                    
        if self.featuretype == "ann-based": 
            audio_features = []
            len_of_longest_sequence = 100 * self.zeropadd
            for af_video in af_all:       
                logmel_padded = preparX (af_video,len_of_longest_sequence )
                audio_features.extend(logmel_padded)
            audio_features = numpy.array(audio_features)
        elif self.featuretype == "yamnet-based":
            audio_features = numpy.array(af_all)      
        return audio_features
    
    def get_visual_features (self):
        dict_onsets = self.load_dict_onsets ()
       
        vf_all = []

        for video_name, value in dict_onsets.items():   
            self.video_name = video_name
            self.folder_name = value['folder_name']                                  
            self.feature_path = os.path.join(self.featuredir, self.featuretype , self.split ,  str(self.folder_name))      
            vf = self.load_vf()
            resnet_all = vf['resnet152_avg_pool'] # resnet features for each onset (10*2048)
            resnet = resnet_all #[0:10] 
            if self.featuretype == "ann-based":
                len_of_longest_sequence = self.zeropadd
                resnet_padded = preparY (resnet , len_of_longest_sequence) # 50*2048
                vf_all.append(resnet_padded)
            elif self.featuretype == "yamnet-based":
                vf_all.extend(resnet) 
                
        if self.featuretype == "ann-based": 
            visual_features = []
            len_of_longest_sequence =  self.zeropadd
            for vf_video in vf_all:       
                resnet_padded = preparY (vf_video , len_of_longest_sequence) # 50*2048
                visual_features.extend(resnet_padded)
            visual_features = numpy.array(visual_features)
        elif self.featuretype == "yamnet-based":
            visual_features = numpy.array(vf_all)      
        
        
        return visual_features  

    def build_apc (self, Xshape):
        audio_sequence = Input(shape=Xshape) #Xshape = (1000, 40)
        prenet = Dense(128)(audio_sequence) # (1000, 128)
        context = GRU(32, return_sequences=True)(prenet) # (1000, 32)
        postnet = Conv1D (40, kernel_size=1, padding='same')(context) # (1000, 40)
        predictor = Model(audio_sequence, postnet)
        predictor.compile(optimizer=Adam(lr=1e-04), loss='mean_absolute_error')
        apc = Model(audio_sequence, context)
        #predictor.fit(x_train, y_train)
        return predictor, apc
    
    def train_apc (self,x_train ):
        self.split = "train"
        audio_features_test = self.get_audio_features()
        x_train =  audio_features_test
        
        
        predictor, apc = self.build_apc(Xshape= (1000, 40))
        predictor.fit(x_train, x_train , epochs=10, batch_size=120)
        
        
        # self.split = "test"
        # self.featuredir = self.featuredir_test
        # audio_features_test = self.get_audio_features()
        # x_test =  audio_features_test       
        # predictor.evaluate(x_test, x_test, batch_size=120)
        
        apc_features = apc.predict(x_train)
        
        return apc_features
        
        
    def build_simple_audio_model (self,Xshape ):     
        dropout_size = 0.3
        activation_C='relu'
    
        audio_sequence = Input(shape=Xshape) #(2500, 40)
                     
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
        
        out_audio_channel = pool5
        out_audio_channel = Reshape([pool5.shape[2]],name='reshape_audio')(out_audio_channel) 
        audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
        audio_model.summary()
        return audio_sequence , out_audio_channel , audio_model

    def build_resDAVEnet (self, Xshape):     
    
        audio_sequence = Input(shape=Xshape) #Xshape = (2500, 40)
        
        strd = 2
        
        x0 = Conv1D(128,1,strides = 1, padding="same")(audio_sequence)
        x0 = BatchNormalization(axis=-1)(x0)
        x0 = ReLU()(x0) 
          
        # layer 1  
        in_residual = x0  
        x1 = Conv1D(128,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)       
        x2 = Conv1D(128,9,strides = strd, padding="same")(x1)  
        x1downsample = Conv1D(128,9,strides = strd, padding="same")(x1)
        out = Add()([x1downsample,x2])
        out_1 = ReLU()(out) # (500, 128) 
        
        # layer 2
        in_residual = out_1  
        x1 = Conv1D(256,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)    
        x2 = Conv1D(256,9,strides = strd, padding="same")(x1)  
        x1downsample = Conv1D(256,9,strides = strd, padding="same")(x1)
        out = Add()([x1downsample,x2])
        out_2 = ReLU()(out) # (256, 256)
        
        # layer 3
        in_residual = out_2  
        x1 = Conv1D(512,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)    
        x2 = Conv1D(512,9,strides = strd, padding="same")(x1)  
        x1downsample = Conv1D(512,9,strides = strd, padding="same")(x1)
        out = Add()([x1downsample,x2])
        out_3 = ReLU()(out) # (128, 512)

        
        # layer 4
        in_residual = out_3  
        x1 = Conv1D(1024,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)       
        x2 = Conv1D(1024,9,strides = strd, padding="same")(x1)  
        x1downsample = Conv1D(1024,9,strides = strd, padding="same")(x1)
        out = Add()([x1downsample,x2])
        out_4 = ReLU()(out)   # (64, 1024)  
        
        #self attention
        # input_attention = out_4
        # query = input_attention
        # key = input_attention
        # value = Permute((2,1))(input_attention)
        
        # score = dot([query,key], normalize=False, axes=-1,name='scoreA')
        # scaled_score = score/int(query.shape[1])
        # weight = Softmax(name ='weigthA')(scaled_score)
        # attention = dot([weight, value], normalize=False, axes=-1,name='attentionA')
        
        # poolAtt = AveragePooling1D(64, padding='same')(attention) # N, 1, 1024
        
        # poolquery = AveragePooling1D(64, padding='same')(query)# N, 1, 1024
        
        # outAtt = Concatenate(axis=-1)([poolAtt, poolquery])
        # outAtt = Reshape([2048],name='reshape_out_attA')(outAtt)
        # out_audio_channel = outAtt
        
        # poling
        out_audio_channel  = AveragePooling1D(64,padding='same')(out_4) 
        out_audio_channel = Reshape([out_audio_channel.shape[2]])(out_audio_channel) 
        
        out_audio_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_audio')(out_audio_channel)
        
        audio_model = Model(inputs= audio_sequence, outputs = out_audio_channel )
        audio_model.summary()
        return audio_sequence , out_audio_channel , audio_model   
     
    def build_visual_model (self, Yshape):
        
        dropout_size = 0.3
        visual_sequence = Input(shape=Yshape) #Yshape = (25,2048)
        
        #resh0 = Reshape([1, visual_sequence.shape[1],visual_sequence.shape[2]],name='reshape_visual')(visual_sequence) 
        forward_visual = Conv1D(1024,3,strides=1,padding = "same", activation='relu', name = 'conv_visual')(visual_sequence)
        dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
        bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)
        
        ##self attention
        # input_attention = bn_visual
        # query = input_attention
        # key = input_attention
        # value = Permute((2,1))(input_attention)
        
        # score = dot([query,key], normalize=False, axes=-1,name='scoreV')
        # scaled_score = score/int(query.shape[1])
        # weight = Softmax(name='weigthV')(scaled_score)
        # attention = dot([weight, value], normalize=False, axes=-1,name='attentionV')
        
        # poolAtt = AveragePooling1D(10, padding='same')(attention) # N, 1, 1024
        
        # poolquery = AveragePooling1D(10, padding='same')(query)# N, 1, 1024
        
        # outAtt = Concatenate(axis=-1)([poolAtt, poolquery])
        # outAtt = Reshape([2048],name='reshape_out_attV')(outAtt)
        # out_visual_channel = outAtt
        
        #max pooling
        pool_visual = MaxPooling1D(10,padding='same')(bn_visual)
        out_visual_channel = Reshape([pool_visual.shape[2]])(pool_visual)
        
        
        out_visual_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_visual')(out_visual_channel)
        
        visual_model = Model(inputs= visual_sequence, outputs = out_visual_channel )
        visual_model.summary()
        return visual_sequence , out_visual_channel , visual_model

     
    def build_network(self, Xshape , Yshape):
           
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model (Yshape)
        if self.audiochannel=="resDAVEnet":        
            audio_sequence , out_audio_channel , audio_model = self.build_resDAVEnet (Xshape)            
        elif  self.audiochannel=="simplenet": 
            audio_sequence , out_audio_channel , audio_model = self.build_simple_audio_model (Xshape)
        
        
        V = out_visual_channel
        A = out_audio_channel
        
        gate_size = 2048
        gatedV_1 = Dense(gate_size)(V)
        gatedV_2 = Dense(gate_size,activation = 'sigmoid')(gatedV_1)        
        gatedV = Multiply(name= 'multiplyV')([gatedV_1, gatedV_2])

        gatedA_1 = Dense(gate_size)(A)
        gatedA_2 = Dense(gate_size,activation = 'sigmoid')(gatedA_1)        
        gatedA = Multiply(name= 'multiplyA')([gatedA_1, gatedA_2])
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs = gatedV, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=audio_sequence, outputs = gatedA, name='audio_embedding_model')
        
        if self.loss == "triplet":
            
            mapIA = dot([gatedV,gatedA],axes=-1,normalize = True,name='dot_matchmap')       
            final_model = Model(inputs=[visual_sequence, audio_sequence], outputs = mapIA )
            final_model.compile(loss=triplet_loss, optimizer= Adam(lr=1e-04))
            
        elif self.loss == "MMS":
            s_output = Concatenate(axis=1)([Reshape([1 , gatedV.shape[1]])(gatedV) ,  Reshape([1 ,gatedA.shape[1]])(gatedA)])
            final_model = Model(inputs=[visual_sequence, audio_sequence], outputs = s_output )
            final_model.compile(loss=mms_loss, optimizer= Adam(lr=1e-03))
    
        final_model.summary()
    
        return visual_embedding_model,audio_embedding_model,final_model

 
       

    def __call__ (self):

        self.split = "test"
        self.featuretype = self.test_feature_type
        print('.........................................in testing feature type is')
        print(self.featuretype)
        
        audio_features_test = self.get_audio_features()            
        visual_features_test = self.get_visual_features()
        Ytest, Xtest, bin_val = prepare_data (audio_features_test , visual_features_test , self.loss)       
        Xshape = numpy.shape(audio_features_test)[1:]        
        Yshape = numpy.shape(visual_features_test)[1:] 
        
        visual_embedding_model,audio_embedding_model,final_model = self.build_network( Xshape , Yshape )
        
        final_model.evaluate([Ytest,Xtest], bin_val, batch_size=120)
    
        
  
        for epoch in range(50):
            
            self.split = "train"
            self.featuretype = self.train_feature_type
            print('...................in training feature type is')
            print(self.featuretype)
            
            audio_features_train = self.get_audio_features()            
            visual_features_train = self.get_visual_features()
            Y,X,b = prepare_data (audio_features_train , visual_features_train  , self.loss) 
            del audio_features_train, visual_features_train
            final_model.fit([Y,X], b, shuffle=False, epochs=1, batch_size=120)
            del Y,X,b
            
            # self.split = "val"
            # self.featuredir = self.featuredir_train
            # audio_features_train = self.get_audio_features()            
            # visual_features_train = self.get_visual_features()
            # Y, X, b = prepare_data (audio_features_train , visual_features_train , self.loss)
            # del audio_features_train, visual_features_train
            # final_model.fit([Y,X], b, shuffle=False, epochs=1, batch_size=120) 
            # del Y,X,b
            
            final_model.evaluate([Ytest,Xtest], bin_val, batch_size=120)

            audio_embeddings = audio_embedding_model.predict(audio_features_test)    
            visual_embeddings = visual_embedding_model.predict(visual_features_test) 
            # audio_embeddings = numpy.squeeze(audio_embeddings)
            # visual_embeddings = numpy.squeeze(visual_embeddings)
            print('checking embedd shape')
            print(audio_embeddings.shape)
            print(visual_embeddings.shape)
            
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
            
            ########### saving the results     
            savepath = '/worktmp/khorrami/project_5/video/model/youcook2/test/'
            self.av_all.append(recall10_av)
            self.va_all.append(recall10_va)
            savemat(os.path.join(savepath,'recalls.mat'), {"av_all":self.av_all,"va_all":self.va_all})
            
            plt.plot(self.av_all)
            plt.plot(self.va_all)
            plt.grid()
            plt.savefig(os.path.join(savepath,'recalls.pdf'))
            
        return self.av_all , self.va_all            
            
