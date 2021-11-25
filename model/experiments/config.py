
training_config = {  
  "featuredir": "../../../features/youcook2/",
  "featuretype": "ann-based",
  "outputdir": "../../../model/youcook2/experiments/",
  "dataset": "YOUCOOK2",
  "use_pretrained": False,
  "save_results" : True,
  "plot_results" : False
}

feature_config = {
    "audio_feature_name": 'embeddings',
    "speech_feature_name": 'logmel40',
    "image_feature_name": 'resnet152_avg_pool' # 'Xception_block14_sepconv2_act',#,
    }

model_config = { 
    "audio_model_name" : "resDAVEnet",
    "visual_model_name": "resnet152", # "Xception",
    "visual_layer_name": "avg_pool", #'block14_sepconv2_act',#
    
    "loss" : "triplet", 
    "clip_length" : 10
}


