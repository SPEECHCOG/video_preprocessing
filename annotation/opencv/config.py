

paths = {  
  "split": "training",
  "datadir": "/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2",
  "outputdir": "../../../features/ouput/youcook2/ann-based/",
}

basic = {
    "dataset": "YOUCOOK2",
    "audio_model" : "yamnet",
    "visual_model": "resnet152", 
    "layer_name": "avg_pool",
    "save_results" : True,
    "plot_results" : False
}

video_settings = {
  "audio_sample_rate": 16000,
  "clip_length_seconds" : 10,
}
