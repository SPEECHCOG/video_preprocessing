import json

json_file = '/worktmp2/hxkhkh/current/video/data/youcook2/bb/annotations/yc2_bb_val_annotations.json'
with open(json_file) as handle:
    json_content = json.load(handle)
    
database = json_content ['database']

for vid_name, vid_info in database.items():
    vid_duration = vid_info['duration']  
    vid_rw = vid_info['rwidth']
    vid_rh = vid_info['rheight']
    
    segments = vid_info['segments'] 
    
    for segment_number, bb_info in segments.items():
        
        segment = bb_info['segment']
        onset = segment[0]
        offset = segment[1]
        number_of_frames = offset-onset
        
        objects = bb_info['objects']
        number_of_objects = len(objects)
        
        for item in objects:
            boxes = item ['boxes']
            label = item ['label']