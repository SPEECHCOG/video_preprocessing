import keras
from utils import triplet_loss,  my_mms_loss
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Lambda

from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Input, Reshape, Dense, Dropout, BatchNormalization, dot, Softmax, Permute, UpSampling1D, Masking, Permute
from tensorflow.keras.layers import  MaxPooling1D, MaxPooling2D, MaxPooling3D, AveragePooling1D,  Conv1D, Conv3D, Concatenate, ReLU, Add, Multiply, GRU
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
        # audio_sequence_masked = Masking (mask_value=0., input_shape=X1shape)(audio_sequence)
        # strd = 2
        
        # a0 = Conv1D(512,1,strides = 1, padding="same")(audio_sequence_masked)
        # a0 = BatchNormalization(axis=-1)(a0)
        # a0 = ReLU()(a0) #(21,1024)
        
        # out_sound_channel = UpSampling1D(3, name = 'upsampling_sound')(a0)
        # out_sound_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_sound')(out_sound_channel)
        
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
        
        out_speech_channel = out_4
        # out_speech_channel  = AveragePooling1D(64,padding='same')(out_4) 
        # out_speech_channel = Reshape([out_speech_channel.shape[2]])(out_speech_channel) 
        
          
        #out_speech_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_speech')(out_speech_channel)
        
        # combining sound and speech branches
        #out_audio_channel = Concatenate(axis=-1)([out_sound_channel, out_speech_channel])
        out_audio_channel = out_speech_channel
        
        audio_model = Model(inputs= [audio_sequence, speech_sequence], outputs = out_audio_channel )
        audio_model.summary()
        return audio_sequence , speech_sequence  , out_audio_channel , audio_model   
     
    def build_visual_model (self, Yshape):
        
        dropout_size = 0.3
        visual_sequence = Input(shape=Yshape) #Yshape = (10,7,7,2048)
        visual_sequence_norm = BatchNormalization(axis=-1, name = 'bn0_visual')(visual_sequence)
        
        forward_visual = Conv3D(1024,(1,3,3),strides=(1,1,1),padding = "same", activation='linear', name = 'conv_visual')(visual_sequence_norm)
        dr_visual = Dropout(dropout_size,name = 'dr_visual')(forward_visual)
        bn_visual = BatchNormalization(axis=-1,name = 'bn1_visual')(dr_visual)
        
        
        # 
        # visual_sequence_reshaped.shape
        
        #max pooling
        # pool_visual = MaxPooling2D((10,1),padding='same')(visual_sequence_reshaped)
        # out_visual_channel = Reshape([pool_visual.shape[2], pool_visual.shape[3]])(pool_visual)
        
        pool_visual = MaxPooling3D((10,1,1),padding='same')(bn_visual)
        input_reshape = pool_visual
        #out_visual_channel = Reshape([input_reshape.shape[4]])(input_reshape)
        out_visual_channel = Reshape([input_reshape.shape[2]*input_reshape.shape[3],
                                                                  input_reshape.shape[4]], name='reshape_visual')(input_reshape)    
        out_visual_channel.shape
        #out_visual_channel = Lambda(lambda  x: K.l2_normalize(x,axis=-1), name='lambda_visual')(out_visual_channel)
        
        visual_model = Model(inputs= visual_sequence, outputs = out_visual_channel )
        visual_model.summary()
        return visual_sequence , out_visual_channel , visual_model

     
    def build_network(self, X1shape , X2shape , Yshape):
           
        visual_sequence , out_visual_channel , visual_model = self. build_visual_model (Yshape)
        if self.audio_model_name == "resDAVEnet":        
            audio_sequence , speech_sequence  , out_audio_channel , audio_model   = self.build_resDAVEnet (X1shape , X2shape)            
        elif  self.audio_model_name == "simplenet": 
            audio_sequence , out_audio_channel , audio_model = self.build_simple_audio_model (X1shape , X2shape)
        
        
        V = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_visual-final') (out_visual_channel)
        A = Lambda(lambda  x: K.l2_normalize(x,axis=-1),name='lambda_audio-final') (out_audio_channel)
        
        # V = Reshape([1,V.shape[1], V.shape[2]])(V)
        # A = Reshape([1,A.shape[1], A.shape[2]])(A)
        # VA = dot ([V,A],axes=(-1,-1),normalize = True,name='batchdot')
        
        visual_embedding_model = Model(inputs=visual_sequence, outputs= V, name='visual_embedding_model')
        audio_embedding_model = Model(inputs=[audio_sequence , speech_sequence], outputs= A, name='visual_embedding_model')
       
        
        # V_old = AveragePooling1D(64,padding='same')(V)
        # V_old = Reshape([V_old.shape[-1]])(V_old) 

        # A_old = AveragePooling1D(64,padding='same')(A)
        # A_old = Reshape([V_old.shape[-1]])(A_old)         


        # old = K.batch_dot(K.expand_dims(V_old,0), K.expand_dims(A_old,0) , axes=(-1,-1))  
        # old = K.squeeze(old, axis = 0)
        
       
 
        #new = K.squeeze(new, axis = 0)
    
        
        if self.loss == "triplet":  
            mapIA = dot([V,A],axes=-1,normalize = True,name='dot_matchmap')
            # mapIA.shape
            def final_layer(tensor):
                x= tensor 
                score = K.mean( (K.max(x, axis=1)), axis=-1)        
                output_score = Reshape([1],name='reshape_final')(score)          
                return output_score
            lambda_layer = Lambda(final_layer, name="final_layer")(mapIA)  
              
            final_model = Model(inputs=[visual_sequence, [audio_sequence , speech_sequence ]], outputs = mapIA)
            final_model.compile(loss=triplet_loss, optimizer = Adam(lr=1e-04))

            
        elif self.loss == "MMS":
            #s_output = Concatenate(axis=1)([Reshape([1 , V.shape[1], V.shape[2]])(V) ,  Reshape([1 ,A.shape[1], A.shape[2]])(A)])
            # def final_layer(input_tensor):
            #     return input_tensor
            
            # lambda_layer = Lambda(final_layer , name="final_layer" ) ([V,A])     #lambda x,y: [x,y]  
            
            VV = keras.backend.expand_dims(V,0)
            #AA = K.expand_dims(A,0)
            S = keras.backend.batch_dot(VV, A, axes =[-1,-1])
            def final_layer(tensor): #N,N,49,63
                x= tensor 
                score = K.mean( (K.max(x, axis=-1)), axis=-1)        
                     
                return score
            lambda_layer = Lambda(final_layer, name="final_layer")(S)
            final_model = Model(inputs=[visual_sequence, [audio_sequence , speech_sequence]], outputs = lambda_layer)
            final_model.summary()
            final_model.compile(loss=my_mms_loss , optimizer= Adam(lr=1e-03)) # loss={'lambda_visual':mms_loss, 'lambda_speech':mms_loss}
    
        
    
        return visual_embedding_model,audio_embedding_model,final_model

 
       


            
