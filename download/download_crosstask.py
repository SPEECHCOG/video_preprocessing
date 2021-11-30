import csv

# reading task descriptions
# 
# with open("/worktmp2/hxkhkh/current/video/data/crosstask/crosstask_release/tasks_primary.txt", 'r') as handle:
#     file = handle.readlines()
#     for row in file:
#         print(row)
# handle.close()

# # reading video urls
# # 4700 train (83 tasks each different number of videos)
# # 360 validation ( 18 tasks each 20 videos)

# with open("/worktmp2/hxkhkh/current/video/data/crosstask/crosstask_release/videos.csv", 'r') as handle:
#     file = csv.reader(handle,  delimiter='\n')
#     dict_videos = {}
#     for row in file:        
#         row_splitted = row[0].split(',')
#         task_id = row_splitted[0]
#         vid_name = row_splitted[1]
#         vid_url = row_splitted[2]
#         if task_id in dict_videos:
#             dict_videos [task_id]['vid_name'].append(vid_name)
#             dict_videos [task_id]['vid_url'].append(vid_url)
#         else:
#            dict_videos[task_id] = {} 
#            dict_videos [task_id]['vid_name'] = []
#            dict_videos [task_id]['vid_url'] = []
#            dict_videos [task_id]['vid_name'].append(vid_name)
#            dict_videos [task_id]['vid_url'].append(vid_url)
           
# handle.close()


# downloading code
import threading
import time
import os


dataset_root = '../../data/crosstask/videos/'
dataset_root = '/worktmp2/hxkhkh/current/video/data/crosstask/videos/'
if not os.path.isdir(dataset_root):
    os.mkdir(dataset_root)

missing_vid_lst = []


def download_video ( split, task_id, vid_name):
    print("starting ..............")
    #time.sleep(1)
    if not os.path.isdir(os.path.join(dataset_root, split, task_id)):
        os.mkdir(os.path.join(dataset_root, split, task_id))
    # download the video
    vid_url = 'www.youtube.com/watch?v='+vid_name
    vid_prefix = os.path.join(dataset_root, split, task_id, vid_name) 
    print(f'now is downloading {vid_prefix}')
    os.system(' '.join(("youtube-dl -o", vid_prefix, vid_url)))

    #check if the video is downloaded
    if os.path.exists(vid_prefix+'.mp4') or os.path.exists(vid_prefix+'.mkv') or os.path.exists(vid_prefix+'.webm'):
        print('[INFO] Downloaded {} video {}'.format(split, vid_name))
    else:
        missing_vid_lst.append('/'.join((split, task_id, vid_name )))
        print('[INFO] Cannot download {} video {}'.format(split, vid_name))

    print("ending.....................")



start = time.perf_counter()  

threads = []
vid_fullnames = []
    
#download videos 
split = 'training'
file_name = "/worktmp2/hxkhkh/current/video/data/crosstask/crosstask_release/videos.csv"
with open( file_name, 'r') as handle:
    file = csv.reader(handle,  delimiter='\n')
    for row in file:        
        row_splitted = row[0].split(',')
        task_id = row_splitted[0]
        vid_name = row_splitted[1]
        vid_url = row_splitted[2]

        thread = threading.Thread(target=download_video, args=( split, task_id, vid_name) )
        threads.append(thread)
        vid_fullnames.append(os.path.join(dataset_root, split, task_id, vid_name))
        #download_video ( line, split, rcp_type, vid_name)
            

#1100-1200 (3)
#1000-1100 (1) 
#500-600 (1)

threads_chunk = threads[0:10]

for thread in threads_chunk:
    thread.start()

for thread in threads_chunk:
    thread.join()
           
# finish = time.perf_counter()    
# print(f'Finnished all downloads in {round(finish-start , 2)} seconds')   
# # write the missing videos to file
# missing_vid = open('missing_videos.txt', 'w')
# for line in missing_vid_lst:
#     missing_vid.write(line)    
# #sanitize and remove the intermediate files
# os.system("find ../raw_videos -name '*.part*' -delete")
# os.system(f"find {dataset_root} -name '*.f*' -delete")
