

paths = {  
  "split": "training",
  "datadir": "/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2", #"/worktmp2/hxkhkh/current/video/data/youcook2", #, 
  "outputdir": "../../features/youcook2/yamnet-based/",
  "exp_name" : "exp4",
}

basic = {
    "visual_model_name": "Xception" , #"resnet152", 
    "visual_layer_name": 'block14_sepconv2_act' , #"avg_pool",
    "save_images" : False,
    "save_visual_features" : True,
}

video_settings = {
  "audio_sample_rate": 16000,
  "clip_length_seconds" : 10,
}
