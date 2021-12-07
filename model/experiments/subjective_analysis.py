from train_validate_matchmap import Train_AVnet
import numpy
from matplotlib import pyplot as plt
import cv2
###############################################################################

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = False
# config.gpu_options.per_process_gpu_memory_fraction=0.7
# sess = tf.compat.v1.Session(config=config)

###############################################################################

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config)
 
###############################################################################  
import config as cfg

training_config = cfg.training_config
model_config = cfg.model_config
feature_config = cfg.feature_config
###############################################################################

obj = Train_AVnet(model_config , feature_config, training_config)
images,wavs, vid_names, preds  = obj.predict()

preds_rsh = numpy.reshape(preds,[preds.shape[0],7,7,63])


###############################################################################
#%%
# 158
text = ' ... she is going to make this delicious miso soup for us today  \n we are going to use firm Tofu for this recipe...  '
#162
#text = ' ... for our miso soup, we have dried  sea vegetable here \n we are not going to use the whole thing,\n I will show you when we are making it, we have white ...  '
# 246
#text = '... avocado strips across the center of nori\n , squeeze a line of mayonnaise next to this, \n if your mayonnaise is not ... '
m = 60
sample = images[m],wavs[m],preds_rsh[m]

sample_matchmap = sample[2]
matchmap_t = numpy.transpose(sample_matchmap, axes= [2,0,1])

res_target_h = 224    
res_target_w = 224
res_target_t = 1000

res_source_h = 7
res_source_w = 7
res_source_t = 63

scale_t = int(numpy.ceil(res_target_t /res_source_t))
scale_h = int(res_target_h /res_source_h)
scale_w = int(res_target_w /res_source_w)

def upsample_3D (input_tensor, scale_T , scale_H, scale_W):
    tensor_detected_uptime = numpy.repeat(input_tensor,scale_T, axis=0)
    output_tensor = numpy.repeat (numpy.repeat(tensor_detected_uptime,scale_W, axis=2)  , scale_H , axis=1)
    return output_tensor

matchmap_upsampled = upsample_3D (matchmap_t, scale_t , scale_h, scale_w)

# plt.imshow(matchmap_t[62,:,:])
# plt.imshow(matchmap[:,:,62])

sample_image_name = sample[0]


sample_image = plt.imread(sample_image_name + '/1.jpg') # central image
plt.subplot(3,5,1) 
plt.title('image 1', fontsize=10)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(sample_image_name + '/3.jpg') # central image
plt.subplot(3,5,2) 
plt.title('image 3', fontsize=10)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(sample_image_name + '/5.jpg') # central image
plt.subplot(3,5,3) 
plt.title('image 5', fontsize=10)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(sample_image_name + '/7.jpg') # central image
plt.subplot(3,5,4) 
plt.title('image 7', fontsize=10)  
plt.imshow(sample_image)
plt.axis('off')


sample_image = plt.imread(sample_image_name + '/9.jpg') # central image
plt.subplot(3,5,5) 
plt.title('image 9', fontsize=10)  

plt.imshow(sample_image)
plt.axis('off')


for i in range(10):
    plt.subplot(3,5,i+6)
    plt.imshow(matchmap_upsampled[100*i,:,:])
    plt.axis('off')
    plt.title(str(i + 1), fontsize=10)
 
plt.suptitle(text, fontsize=8)
plt.savefig(obj.outputdir + 'samples/' + 'sample_' + str(m) + '.pdf' )




sample_audio_file = sample[1]
print(sample_image_name)
print(sample_audio_file)

plt.imshow(sample_image)