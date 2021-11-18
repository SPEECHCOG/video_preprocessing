#  "../../data/input/"

paths = {  
  "split": "testing",
  "datadir": "../../data/youcook2", #"/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/video/youcook2",
  "outputdir": "../../features/youcook2/yamnet-based/",
  "exp_name" : "exp4"
}

basic = {
    "dataset": "YOUCOOK2",
    "audio_model" : "yamnet",
    "run_speech_detection" : True,
    "save_results" : True,
    "save_wavs" : False
}

yamnet_settings = {
  "target_sample_rate": 16000,
  "logmel_bands": 40,
  "win_length_logmel": 0.025,
  "win_hope_logmel":0.01,
  "win_length_yamnet" : 0.96,
  "win_hope_yamnet" : 0.48,
  "class_names_accepted" : ['Speech', 'Child speech, kid speaking' , 'Conversation', 'Narration, monologue' ],
  "class_index_accepted" : [0,1,2,3],
  "clip_length_seconds" : 10,
  "acceptance_snr" : 0.7,
  "skip_seconds" : 3,
  "accepted_overlap_second": 5
}
