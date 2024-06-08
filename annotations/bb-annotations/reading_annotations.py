import json
import pickle
json_file = '../../../data/youcook2/bb/annotations/yc2_bb_val_annotations.json'
with open(json_file) as handle:
    json_content = json.load(handle)
    
database_bbx = json_content ['database']

for vid_name, vid_info in database_bbx.items():
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
            
#%%

import json
import os


split = 'validation'

datadir = "../../../data/youcook2"
test_ann_file = "../../../data/youcook2/annotations/youcookii_annotations_test_segments_only.json"
train_ann_file = "../../../data/youcook2/annotations/youcookii_annotations_trainval.json"

outputdir = "../../../data/youcook2/annotations/"

with open(train_ann_file) as handle:
    annotations_trainval = json.load(handle) 

with open(test_ann_file) as handle:
    annotations_test = json.load(handle) 

if split == 'training' or split== 'validation':
    database_anns = annotations_trainval['database']
else:
    database_anns = annotations_test['database']
    
#.......................................................... listing video names
    
def create_video_list (datadir, split ):        
    video_dir = os.path.join(datadir, 'videos' , split) 
    video_recepies = os.listdir(video_dir)
    video_list = []
    for rec in video_recepies:
        files = os.listdir(os.path.join(video_dir, rec))
        video_list.extend([os.path.join(split , rec ,f) for f in files])
    return video_list
    
def create_video_names(video_list):
    video_names = []
    for item in video_list:
        item_splitted = item.split('/')
        video_name_extended = item_splitted[-1]
        video_name = video_name_extended.split('.')[0]
        video_names.append(video_name)
    return video_names 

def load_dict_onsets (split):
    input_name =  os.path.join('../../../features/youcook2/ann-based' , split + '_onsets')   
    with open(input_name, 'rb') as handle:
        dict_onsets = pickle.load(handle)
    return dict_onsets

def create_folder_list(split):
    dict_onsets = load_dict_onsets (split)
    folders = []
    for video_name, value in dict_onsets.items():      
        folders.append(value['folder_name']) 
    return folders
folders =  create_folder_list('validation') 
      
video_list = create_video_list (datadir, split )
video_names = create_video_names (video_list)

all_durations = []
all_id_counts = []
all_sentences = []
all_clip_durations = 0
counter_video = 0
all_info = {}

for video_fullname in video_list:
    video_name = video_fullname.split('/')[-1]
    vid_name = video_name.split('.')[0]
    video_info = {}
    video_info ['vid_name'] = vid_name
    
    info_anns = {}
    info_bbx = {}
    try:
        info_anns = database_anns[vid_name]
        
        
    except:
        print('... in video ' + video_fullname + ' annotation is missed ... ')
    
    try:
        info_bbx = database_bbx[vid_name]
               
    except:
        print('... in video ' + video_fullname + ' bbx is missed ... ')
    
    video_info['info_anns'] = info_anns    
    video_info['info_bbx'] = info_bbx
        

    
    all_info[video_fullname] = video_info
    # duration = info_anns['duration']
    # anns = info_anns['annotations']
    # id_counts = len(anns)
    # video_sentences = []
    # for clip in anns:
    #     sentence = clip['sentence']
    #     video_sentences.append(sentence)
        
    #     clip_time = clip['segment']
    #     clip_duration = clip_time[1]-clip_time[0]
    #     all_clip_durations += clip_duration
    
    # all_durations.append(duration)
    # all_id_counts.append(id_counts)
    # all_sentences.append(video_sentences)
    # video_info = {}
    # video_info['video_full_name'] = video_fullname
    # video_info['video_name'] = vid_name
    # video_info['id_counts'] = id_counts
    # video_info['sentences'] = video_sentences
    # all_info [counter_video] = video_info
    # counter_video +=1
    
counter_clip = 0
clips = {}    
for video_fullname, value in all_info.items():
    print(video_fullname)
    info_anns = value['info_anns']
    info_bbx = value['info_bbx']    
    annotations = info_anns['annotations']
    for item_ann in annotations:       
        clip_info = {}
        #using segment as key because it's shared between two annotations       
        clip_info['video_fullname'] = video_fullname 
        clip_info['sentence'] = item_ann['sentence']
        clip_info ['segment'] = item_ann['segment']       
        clips[counter_clip] = clip_info
        counter_clip += 1
    
    
counter_clip = 0
clips_bbx = {}    
for video_fullname, value in all_info.items():
    print(video_fullname)
    info_anns = value['info_anns']
    info_bbx = value['info_bbx']
    annotations = info_anns['annotations']
    
    try:
        annotations_bbx = info_bbx['segments']
        for item_bbx, value_bbx in annotations_bbx.items():           
            clip_info = {}           
            clip_info['video_fullname'] = video_fullname 
            object_list = []
            object_boxes = []
            for item_object in value_bbx['objects']:
                object_list.append(item_object['label'])
                object_boxes.append(item_object['boxes'])           
            clip_info['objects'] = object_list
            clip_info['objects-bbx'] = object_boxes
            clip_info ['segment'] = value_bbx['segment']           
            clips_bbx[counter_clip] = clip_info
            for item_ann in annotations:
                segment = item_ann['segment']
                if segment == value_bbx['segment']:
                    clip_info ['annotation'] = item_ann
            
            counter_clip += 1    
    
    except:
        pass


with open(outputdir + split + '_clips_bbx' , 'wb') as handle:
     pickle.dump(clips_bbx, handle)
     
with open(outputdir + split + '_clips', 'wb') as handle:
     pickle.dump(clips, handle)