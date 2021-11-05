import threading
import time
import os
import subprocess

dataset_root = '../../data/trial'
vid_file_lst = ['../../data/youcook2/splits/train_list.txt', '../../data/youcook2/splits/val_list.txt', '../../data/youcook2/splits/test_list.txt']
split_lst = ['training', 'validation', 'testing']
if not os.path.isdir(dataset_root):
    os.mkdir(dataset_root)

missing_vid_lst = []


def download_video ( line, split, rcp_type, vid_name):
    print("starting ..............")
    #time.sleep(1)
    if not os.path.isdir(os.path.join(dataset_root, split, rcp_type)):
        os.mkdir(os.path.join(dataset_root, split, rcp_type))
    # download the video
    vid_url = 'www.youtube.com/watch?v='+vid_name
    vid_prefix = os.path.join(dataset_root, split, rcp_type, vid_name) 
    print(f'now is downloading {vid_prefix}')
    os.system(' '.join(("youtube-dl -o", vid_prefix, vid_url)))

    #check if the video is downloaded
    if os.path.exists(vid_prefix+'.mp4') or os.path.exists(vid_prefix+'.mkv') or os.path.exists(vid_prefix+'.webm'):
        print('[INFO] Downloaded {} video {}'.format(split, vid_name))
    else:
        missing_vid_lst.append('/'.join((split, line)))
        print('[INFO] Cannot download {} video {}'.format(split, vid_name))
    
    
    print("ending.....................")



start = time.perf_counter()  

threads = []
vid_names = []
    
#download videos for training/validation/testing splits
for vid_file, split in zip(vid_file_lst, split_lst):
    if not os.path.isdir(os.path.join(dataset_root, split)):
        os.mkdir(os.path.join(dataset_root, split))
    with open(vid_file) as f:
        lines = f.readlines()
        for line in lines:         
            rcp_type,vid_name = line.replace('\n','').split('/')
            
            thread = threading.Thread(target=download_video, args=( line, split, rcp_type, vid_name) )
            threads.append(thread)
            vid_names.append(os.path.join(dataset_root, split, rcp_type, vid_name))
            #download_video ( line, split, rcp_type, vid_name)
            

#1100-1200 (3)
#1000-1100 (1) 
#500-600 (1)

threads_chunk = threads[500:600]

for thread in threads_chunk:
    thread.start()

for thread in threads_chunk:
    thread.join()
           
finish = time.perf_counter()    
print(f'Finnished all downloads in {round(finish-start , 2)} seconds')   
# write the missing videos to file
missing_vid = open('missing_videos.txt', 'w')
for line in missing_vid_lst:
    missing_vid.write(line)    
#sanitize and remove the intermediate files
os.system("find ../raw_videos -name '*.part*' -delete")
os.system(f"find {dataset_root} -name '*.f*' -delete")
