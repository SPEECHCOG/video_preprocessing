

paths = {  
  "split": "test",
  "datadir": "/tuni/groups/3101050_Specog/corpora/youcook2_dataset",
  "outputdir": "../../../data/youcook2/output/ann-based/"
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
