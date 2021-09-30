

paths = {  
  "split": "train",
  "datadir": "/tuni/groups/3101050_Specog/corpora/youcook2_dataset",
  "outputdir": "../../data/youcook2/output/"
}

basic = {
    "dataset": "YOUCOOK2",
    "audio_model" : "yamnet", 
    "save_results" : True,
    "plot_results" : False
}

video_settings = {
  "audio_sample_rate": 16000,
  "clip_length_seconds" : 10,
}
