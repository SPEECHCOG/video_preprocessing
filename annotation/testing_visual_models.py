from matplotlib import pyplot as plt
import numpy
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_vgg

from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input as processing_input_resnet
from tensorflow.keras.applications.resnet import decode_predictions as decode_resnet

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as processing_input_xception
from tensorflow.keras.applications.xception import decode_predictions as decode_xception

output_dir = '/worktmp/khorrami/project_5/video/features/ouput/youcook2/examples/visual_detections/'
image_dir = '/worktmp/khorrami/project_5/video/features/ouput/youcook2/ann-based/'



def load_model (visual_model_name):    
    if visual_model_name == "vgg16":
        base_model = VGG16()
    if visual_model_name == "resnet152":
        base_model = ResNet152(weights='imagenet')
    if visual_model_name == "Xception":
        base_model = Xception(weights='imagenet')
    return base_model


def prepare_image (image_fullname, visual_model_name):
    image_original = image.load_img(image_fullname, target_size=(224, 224))
    image_array = image.img_to_array(image_original)
    image_dim = numpy.expand_dims(image_array, axis=0)
    if visual_model_name == "vgg16":
        image_input = preprocess_input_vgg(image_dim)
    if visual_model_name =="resnet152":
        image_input = processing_input_resnet(image_dim)
    if visual_model_name =="Xception":
        image_input = processing_input_xception(image_dim) 
    return image_input


def calculate_visual_features(image_fullname, visual_model_name, visual_model):
    image_input = prepare_image (image_fullname, visual_model_name)   
    features_out = visual_model.predict(image_input)
    features_out_reshaped = numpy.squeeze(features_out)
    return   features_out_reshaped  

def find_image_class (image_fullname, visual_model_name, model) :
    image_input = prepare_image (image_fullname, visual_model_name)
    prediction = model.predict(image_input)
    # convert the probabilities to class labels 
    if visual_model_name == "vgg16":
        label = decode_vgg(prediction)
    if visual_model_name =="resnet152":
        label = decode_resnet(prediction)
    if visual_model_name =="Xception":
        label = decode_xception(prediction)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    
    return label

#%% loading example image
cnt_im = 10
image_name = 'validation/187/images/10/1.jpg'

test_image_name = os.path.join(image_dir,image_name )
test_image_original = image.load_img(test_image_name, target_size=(224, 224))
test_image_array = image.img_to_array(test_image_original)


#%%

visual_model_names = ["vgg16", "resnet152" , "Xception", "AVnet"]
layer_names = ['block5_pool', 'conv5_block3_out', 'block14_sepconv2_act', '']

n = 2
visual_model_name =  visual_model_names[n]
layer_name = layer_names[n]
    
base_model = load_model(visual_model_name)
#base_model.summary()

label = find_image_class (test_image_name, visual_model_name, base_model)
label_name = label[1]
label_confidence = round (label[2] ,2)
print('%s (%.2f%%)' % (label_name, label_confidence*100))


visual_model = Model(inputs=base_model.input,outputs=base_model.get_layer(layer_name).output)
features = calculate_visual_features(test_image_name, visual_model_name,  visual_model)
features_average = numpy.mean(features, axis = -1)

figures,axes = plt.subplots(1,2)
plt.title(visual_model_name + ': ' + label_name + '(' + str(label_confidence) +  ')')
axes[0].imshow(test_image_original)
axes[1].imshow(features_average)

plt.savefig(output_dir + str(cnt_im) + '_' + visual_model_name )
# axes[2].imshow(test_image_original)
# axes[2].imshow(features_average, alpha=0.15)
