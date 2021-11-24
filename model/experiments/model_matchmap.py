
from utils import triplet_loss,  mms_loss,  prepare_data, preparX, preparY, calculate_recallat10 

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization, dot, Softmax, Permute, UpSampling1D, Masking
from tensorflow.keras.layers import  MaxPooling1D, MaxPooling2D, AveragePooling1D,  Conv1D, Concatenate, ReLU, Add, Multiply, GRU
from tensorflow.keras.optimizers import Adam




class AVnet():
    
    def __init__(self, model_config):
        self.audio_model_name = model_config["audio_model_name"]
        self.visual_model_name = model_config["visual_model_name"]
        self.visual_layer_name = model_config["visual_layer_name"]      
        self.loss = model_config["loss"]
        self.clip_length = model_config["clip_length"]
        
   
    def build_apc (self, Xshape):      
        audio_sequence = Input(shape=Xshape) #Xshape = (995, 40)
        prenet = Dense(512, name = 'prenet', activation='relu')(audio_sequence)
        #prenet = Conv1D(128,3, activation='relu', padding='causal')(prenet1)      
        #context = GRU(256, return_sequences=True, name = 'GRU')(prenet3) # (N, 1000, 256)
        context = Conv1D(512, kernel_size=3, padding='causal', dilation_rate=1, activation='relu' , name = 'context1')(prenet)
        context = Conv1D(512, kernel_size=3, padding='causal', dilation_rate=2, activation='relu', name = 'context2')(context)
        context = Conv1D(256, kernel_size=3, padding='causal', dilation_rate=4, activation='relu', name = 'context3')(context)
   
          
        postnet_audio = Conv1D(40, kernel_size=1, padding='same')(context)
    
        predictor = Model(audio_sequence, postnet_audio)
        predictor.compile(optimizer=Adam(lr=1e-04), loss='mean_absolute_error',  metrics=['accuracy'])
        apc = Model(audio_sequence, context)
       
        return predictor, apc

    # def build_apc_visual (self, Xshape):
    #     #visual block
    #     visual_sequence = Input(shape=(10,2048))  
    #     prenet_visual = Dense(256, activation='relu', name = 'prenetvisual')(visual_sequence)
    #     pool_visual = UpSampling1D(100, name = 'upsamplingvisual')(prenet_visual) # (N, 1000, 256)
    #     #forward_visual = Dense(256,activation='relu')(pool_visual)#(N, 1000, 256)
    #     out_visual = pool_visual[:,:-5,:]
        
        
    #     audio_sequence = Input(shape=Xshape) #Xshape = (995, 40)
    #     prenet = Dense(256, activation='relu', name = 'prenet')(audio_sequence)
    #     #prenet = Conv1D(128,3, activation='relu', padding='causal')(prenet1)
        
    #     #context = GRU(256, return_sequences=True, name = 'GRU')(prenet) # (N, 1000, 256)
    #     context = Conv1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=1,name = 'context1')(prenet)
    #     context = Conv1D(256, kernel_size=3, padding='causal', dilation_rate=2, name = 'context2')(context)
    #     context = Conv1D(256, kernel_size=3, padding='causal', dilation_rate=4, name = 'context3')(context)
    #     context_audiovisual = Add()([context, out_visual])# (N, 1000, 256)  
    #     postnet_audiovisual = Conv1D(40, kernel_size=1, padding='same')(context_audiovisual) # (1000, 40) 
    #     predictor = Model(audio_sequence,postnet_audiovisual)
    #     predictor.compile(optimizer=Adam(lr=1e-04), loss='mean_absolute_error',  metrics=['accuracy'])
    #     apc = Model(audio_sequence, context)
        
    #     return predictor, apc
    

        
    def build_simple_speech_model (self,Xshape ):     
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
        #audio_model.summary()
        return audio_sequence , out_audio_channel , audio_model

    def build_resDAVEnet (self, X1shape, X2shape): 
        
        audio_sequence = Input(shape=X1shape) #X1shape = (21, 1024)
        speech_sequence = Input(shape=X2shape) #X2shape = (1000, 40)
        
        # audio channel
        audio_sequence_masked = Masking (mask_value=0., input_shape=X1shape)(audio_sequence)
        strd = 2
        
        a0 = Conv1D(1024,1,strides = 1, padding="same")(audio_sequence_masked)
        a0 = BatchNormalization(axis=-1)(a0)
        a0 = ReLU()(a0) #(21,1024)
        
        out_sound_channel = UpSampling1D(3, name = 'upsampling_sound')(a0)
        #out_sound_channel = Reshape([out_sound_channel.shape[2]])(out_sound_channel)  #(N, 1024)
        out_sound_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_sound')(out_sound_channel)
        
        # speech channel
        speech_sequence_masked = Masking(mask_value=0., input_shape=X2shape)(speech_sequence)
        strd = 2
        
        x0 = Conv1D(128,1,strides = 1, padding="same")(speech_sequence_masked)
        x0 = BatchNormalization(axis=-1)(x0)
        x0 = ReLU()(x0) 
          
        # layer 1  
        in_residual = x0  
        x1 = Conv1D(128,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)       
        x2 = Conv1D(128,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(128,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_1 = ReLU()(out) # (500, 128) 
        
        # layer 2
        in_residual = out_1  
        x1 = Conv1D(256,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)    
        x2 = Conv1D(256,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(256,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_2 = ReLU()(out) # (256, 256)
        
        # layer 3
        in_residual = out_2  
        x1 = Conv1D(512,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)    
        x2 = Conv1D(512,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(512,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
        out_3 = ReLU()(out) # (128, 512)

        
        # layer 4
        in_residual = out_3  
        x1 = Conv1D(1024,9,strides = 1, padding="same")(in_residual)
        x1 = BatchNormalization(axis=-1)(x1)
        x1 = ReLU()(x1)       
        x2 = Conv1D(1024,9,strides = strd, padding="same")(x1)  
        downsample = Conv1D(1024,9,strides = strd, padding="same")(in_residual)
        out = Add()([downsample,x2])
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
        #out_4 = Masking (mask_value=0., input_shape=out_4.shape[1:]) (out_4)
        
        out_speech_channel = out_4
          
        out_speech_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_speech')(out_speech_channel)
        
        # combining sound and speech branches
        out_audio_channel = Concatenate(axis=-1)([out_sound_channel, out_speech_channel])
        
        
        audio_model = Model(inputs= [audio_sequence, speech_sequence], outputs = out_audio_channel )
        #audio_model.summary()
        return audio_sequence , speech_sequence  , out_audio_channel , audio_model   
     
    def build_visual_model (self, Yshape):
        
        dropout_size = 0.3
        visual_sequence = Input(shape=Yshape) #Yshape = (10,7,7,2048)
        
        #resh0 = Reshape([1, visual_sequence.shape[1],visual_sequence.shape[2]],name='reshape_visual')(visual_sequence) 
        # forward_visual = Conv1D(1024,3,strides=1,padding = "same", activation='relu', name = 'conv_visual')(visual_sequence)
        # dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
        # bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)
        
        visual_sequence_reshaped = Reshape([ visual_sequence.shape[1] ,visual_sequence.shape[2]*visual_sequence.shape[3], visual_sequence.shape[4]],name='reshape_visual')(visual_sequence) 
        visual_sequence_reshaped.shape
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
        pool_visual = MaxPooling2D((10,1),padding='same')(visual_sequence_reshaped)
        out_visual_channel = Reshape([pool_visual.shape[2], pool_visual.shape[3]])(pool_visual)
        
        
        out_visual_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_visual')(out_visual_channel)
        
        visual_model = Model(inputs= visual_sequence, outputs = out_visual_channel )
        #visual_model.summary()
        return visual_sequence , out_visual_channel , visual_model

     
    def build_network(self, X1shape , X2shape , Yshape):
           
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model (Yshape)
        if self.audio_model_name == "resDAVEnet":        
            audio_sequence , speech_sequence  , out_audio_channel , audio_model   = self.build_resDAVEnet (X1shape , X2shape)            
        elif  self.audio_model_name == "simplenet": 
            audio_sequence , out_audio_channel , audio_model = self.build_simple_audio_model (X1shape , X2shape)
        
        
        V = out_visual_channel
        A = out_audio_channel
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs= V, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=[audio_sequence , speech_sequence],outputs= A, name='visual_embedding_model')
        
        mapIA = dot([V,A],axes=-1,normalize = True,name='dot_matchmap') 
        
        def final_layer(tensor):
            x= tensor 
            score = K.mean( (K.mean(x, axis=1)), axis=-1)        
            output_score = Reshape([1],name='reshape_final')(score)          
            return output_score
        
        lambda_layer = Lambda(final_layer, name="final_layer")(mapIA) 
        
        # s_output = Concatenate(axis=1)([Reshape([1 , V.shape[1]])(V) ,  Reshape([1 ,A.shape[1]])(A)])
        
        final_model = Model(inputs=[visual_sequence, [audio_sequence , speech_sequence ]], outputs = lambda_layer )
        final_model.compile(loss=triplet_loss, optimizer= Adam(lr=1e-04))
        final_model.summary()


        # return final_model, visual_embedding_model, audio_embedding_model        
          
        # gate_size = 4096
        # gatedV_1 = Dense(gate_size, name = "v1")(V)
        # gatedV_2 = Dense(gate_size,activation = 'sigmoid', name = "v2")(gatedV_1)        
        # gatedV = Multiply(name= 'multiplyV')([gatedV_1, gatedV_2])

        # gatedA_1 = Dense(gate_size, name = "a1")(A)
        # gatedA_2 = Dense(gate_size,activation = 'sigmoid', name = "a2")(gatedA_1)        
        # gatedA = Multiply(name= 'multiplyA')([gatedA_1, gatedA_2])
        
        # visual_embedding_model = Model(inputs=visual_sequence, outputs = gatedV, name='visual_embedding_model')
        # audio_embedding_model = Model(inputs=[audio_sequence , speech_sequence], outputs = gatedA, name='audio_embedding_model')
        
        # if self.loss == "triplet":
            
        #     mapIA = dot([gatedV,gatedA],axes=-1,normalize = True,name='dot_matchmap')       
        #     final_model = Model(inputs=[visual_sequence, audio_sequence], outputs = mapIA )
        #     final_model.compile(loss=triplet_loss, optimizer= Adam(lr=1e-04))
            
        # elif self.loss == "MMS":
        #     
        #     final_model = Model(inputs=[visual_sequence, [audio_sequence , speech_sequence]], outputs = s_output )
        #     final_model.compile(loss=mms_loss, optimizer= Adam(lr=1e-03))
    
        #final_model.summary()
    
        return visual_embedding_model,audio_embedding_model,final_model

 
       


            
