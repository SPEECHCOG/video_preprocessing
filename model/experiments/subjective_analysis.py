from train_validate_matchmap import Train_AVnet
import numpy
from matplotlib import pyplot as plt
import cv2
import scipy.spatial as ss
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
images,wavs, vid_names, preds, visual_feat, audio_embeddings_mean,visual_embeddings_mean  = obj.predict()

preds_rsh = numpy.reshape(preds,[preds.shape[0],7,7,63])


###############################################################################
#%%
# m = 60
# text =' ... this is starting to look awsome, look at that,\n you can see in the last 30 seconds, it changed a lot in texture ...\n'
# m = 158
# text = ' ... she is going to make this delicious miso soup for us today  \n we are going to use firm Tofu for this recipe...  '
# m = 162
# text = ' ... for our miso soup, we have dried  sea vegetable here \n we are not going to use the whole thing,\n I will show you when we are making it, we have white ...  '
m = 246
text = '... avocado strips across the center of nori\n , squeeze a line of mayonnaise next to this, \n if your mayonnaise is not ... '

sample = images[m],wavs[m],preds_rsh[m]
matchmap = sample[2]
image_name = sample[0]
audio_file = sample[1]

plt.Figure()

sample_image = plt.imread(image_name + '/0.jpg') # central image
plt.subplot(4,5,1) 
plt.title('0', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/1.jpg') # central image
plt.subplot(4,5,2) 
plt.title('1', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/2.jpg') # central image
plt.subplot(4,5,3) 
plt.title('2', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/3.jpg') # central image
plt.subplot(4,5,4) 
plt.title('3', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/4.jpg') # central image
plt.subplot(4,5,5) 
plt.title('4', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/5.jpg') # central image
plt.subplot(4,5,6) 
plt.title('5', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/6.jpg') # central image
plt.subplot(4,5,7) 
plt.title('6', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/7.jpg') # central image
plt.subplot(4,5,8) 
plt.title('7', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/8.jpg') # central image
plt.subplot(4,5,9) 
plt.title('8', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')

sample_image = plt.imread(image_name + '/9.jpg') # central image
plt.subplot(4,5,10) 
plt.title('9', fontsize=8)  
plt.imshow(sample_image)
plt.axis('off')


visual_feat_m = visual_feat[m,:]
visual_feat_mean = numpy.mean(visual_feat_m, axis =-1)
for i in range(10):
    plt.subplot(4,5,i+11)
    plt.imshow(visual_feat_mean[i,:])
    plt.axis('off')
    #plt.title(str(i + 1), fontsize=10)
plt.suptitle(text + '\n', fontsize=8)
plt.savefig(obj.outputdir + 'samples/' + 'sample_' + str(m) + '_input.pdf' )    

#%%

matchmap_t = numpy.transpose(matchmap, axes= [2,0,1])

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
# plt.imshow(matchmap_upsampled[0,:,:])
# plt.imshow(matchmap_t[0,:,:])
# plt.imshow(matchmap[:,:,0])
plt.Figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(matchmap_upsampled[100*i,:,:])
    plt.axis('off')
    plt.title(str(i + 1), fontsize=10)
 
plt.suptitle(text, fontsize=8)
plt.savefig(obj.outputdir + 'samples/' + 'sample_' + str(m) + '_adv.pdf' )


print(image_name)
print(audio_file)

#%%
recallat = 10
          
distance_av = ss.distance.cdist( audio_embeddings_mean , visual_embeddings_mean ,  'cosine') # 1-cosine
distance_av_m = distance_av[m]
sort_index = numpy.argsort(distance_av_m)[0:recallat]
results_av = []
results_av.append(images[m])
for n in sort_index: 
    results_av.append(images[n])

plt.Figure()
for i in range(10):
    plt.subplot(2,5,i+1)
    sample_image = plt.imread(results_av[i+1] + '/5.jpg') # central image
    plt.imshow(sample_image)
    plt.axis('off')
    plt.title(str(i + 1), fontsize=8)
 
plt.suptitle(text, fontsize=8)
plt.savefig(obj.outputdir + 'samples/' + 'sample_' + str(m) + '_av.pdf' )