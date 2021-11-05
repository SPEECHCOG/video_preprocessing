import logging
import os
from queue import Queue
from threading import Thread
from time import time

from download import set_download_dir, get_video_name, download_video


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


class DownloadWorker(Thread):

    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            #directory, link = self.queue.get()
            vid_prefix, vid_name = self.queue.get()
            try:
                #download_video(directory, link)
                download_video ( vid_prefix, vid_name)
                
            finally:
                self.queue.task_done()


def main():
    ts = time()
    missing_vid_lst = []
    dataset_root = '../../data/trial'
    vid_file_lst = ['../../data/youcook2/splits/train_list.txt', '../../data/youcook2/splits/val_list.txt', '../../data/youcook2/splits/test_list.txt']
    split_lst = ['training', 'validation', 'testing']
    if not os.path.isdir(dataset_root):
        os.mkdir(dataset_root)

    
    queue = Queue()
    # Create 8 worker threads
    for x in range(12):
        worker = DownloadWorker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
        
    
    #download videos for training/validation/testing splits
    for vid_file, split in zip(vid_file_lst, split_lst):
        if not os.path.isdir(os.path.join(dataset_root, split)):
            os.mkdir(os.path.join(dataset_root, split))
        with open(vid_file) as f:
            lines = f.readlines()
            for line in lines:
                
                rcp_type,vid_name = get_video_name(line)
                vid_prefix = set_download_dir (dataset_root,  split, rcp_type, vid_name)
                
                
                logger.info('Queueing {}'.format(line))
                #queue.put((download_dir, link))
                queue.put((vid_prefix, vid_name))
    queue.join()
    logging.info('Took %s', time() - ts)
                
                
                
                
                
                
    if os.path.exists(vid_prefix+'.mp4') or os.path.exists(vid_prefix+'.mkv') or os.path.exists(vid_prefix+'.webm'):
        print('[INFO] Downloaded {} video {}'.format(split, vid_name))
    else:
        missing_vid_lst.append('/'.join((split, line)))
        print('[INFO] Cannot download {} video {}'.format(split, vid_name))
    
        
    # write the missing videos to file
    missing_vid = open('missing_videos.txt', 'w')
    for line in missing_vid_lst:
        missing_vid.write(line)    
    # sanitize and remove the intermediate files
    # os.system("find ../raw_videos -name '*.part*' -delete")
    #os.system(f"find {dataset_root} -name '*.f*' -delete")






if __name__ == '__main__':
    main()


