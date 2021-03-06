import os 
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
from statistics import mode
import cv2
from keras.models import load_model,Model
import numpy as np
import sys
sys.path.append('./utils')
from inference import apply_offsets
from inference import detect_faces
from inference import draw_text
from inference import draw_bounding_box
from inference import load_detection_model
from inference import load_image
from preprocessor import preprocess_input
from datasets import DataManager
from datasets import get_labels
import tensorflow as tf
import warnings
from keras import layers
import keras.backend as K
from keras.models import Model
from keras.layers import Flatten,Dense,Input,Conv2D, Convolution2D,concatenate
from keras.layers import MaxPooling2D,GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.layers import BatchNormalization,Activation,SeparableConv2D,PReLU,AveragePooling2D
from keras.regularizers import l2
from keras.layers import Dropout,Reshape,Add,merge
from keras.applications import ResNet50,MobileNet
from keras.applications import ResNet50,MobileNet
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.models import Model
import os
from keras.utils import plot_model
sys.path.append('./model')
from models import Net
from wide_resnet import WideResNet

import time

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


is_MTL = True
is_two_input = False
is_grey = False

# parameters for loading data and imagesnp.asarray((255, 0, 0)

is_dropout=False
face_detection_model_path =  'face_detect.xml'
face_detection = load_detection_model(face_detection_model_path)
font = cv2.FONT_HERSHEY_SIMPLEX
frame_window = 10
emotion_offsets = (20, 40)
gender_age_offsets = (30, 60)


is_single_task = False
is_light_net = False

is_pose=True
is_emotion=True
is_age =True
is_gender=True
#is_EGA =True
is_EGA =False
is_EPA=False
is_EPGA = True


image_name='3.jpg'   
main_path='test-images/'
text_left=-20
text_top=-20
text_gap=20
font_size=0.6
font_thickness=2

image_path = main_path+image_name
stored_path = main_path+'{}-{}_'.format(is_light_net,is_EGA)+'result_'+image_name

if is_EGA:
    emotion_labels=get_labels('ferplus')
else:
    emotion_labels = get_labels('fer2013')
gender_labels = get_labels('imdb')
age_labels = get_labels('adience')
pose_labels = get_labels('aflw')

if is_single_task:
    if is_light_net:
        MODEL='mini_xception'
        # emotion_path='train_weights/EmotionNetminixception/expw/freezing_true-drouout_false_1__06-0.56.hdf5'
        # emotion_path = 'train_weights/EmotionNetminixception/fer2013/fer2013mini_XCEPTION.95-0.66.hdf5'
        # gender_age_path=''
        # gender_path='train_weights/GenderNetminixception/adience/freezing_true-drouout_false_6__12-0.86.hdf5'
        # gender_path = 'train_weights/GenderNetminixception/adience/simple_CNN.81-0.96.hdf5'
        # gender_path='train_weights/GenderNetminixception/imdb/freezing_true-drouout_false_1__05-0.95.hdf5'
        # age_path='train_weights/AgeNetEmotionNetminixception/adience/freezing_true-drouout_false_1__11-0.49.hdf5'
        # pose_path='train_weights/PoseNetminixception/aflw/freezing_true-drouout_false_1__14-0.72.hdf5' 
        emotion_path = 'train_weights/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
        gender_path = 'train_weights/trained_models/gender_models/gender_mini_XCEPTION.2878-0.93.hdf5'
        age_path = 'train_weights/trained_models/ag_models/weights.25000-0.03.hdf5'
    else:
        MODEL='vggFace'
        emotion_path='1'
        gender_age_path='1'
        age_path='1'
        pose_path='1'
    # age_model = WideResNet(64, depth=16, k=8)()
    # age_model.load_weights(age_path)
    # age_model = load_model(age_path)
    # emotion_model = load_model(emotion_path)
    # emotion_model=Net(MODEL,1,0,is_dropout,7,5,8,2)
    # gender_model=Net(MODEL,1,10,is_dropout,7,5,8,2)
    # pose_model=Net(MODEL,1,5,is_dropout,7,5,8,2)
    # age_model=Net(MODEL,1,1,is_dropout,7,5,8,2)
    # emotion_model.load_weights(emotion_path)
    # gender_model.load_weights(gender_path)
    # age_model.load_weights(age_path)
    # pose_model.load_weights(pose_path)
    # target_size = (64,64)

else:
    if is_light_net:
        MODEL='mini_xception'
        EGA_path='train_weights/demo_weights/MTL/VGG/EGA/11_0.59-0.96-0.78.hdf5'
        EPGA_path='train_weights/CONFUSION/ferplus-adience-aflw/EPGA-minixception/net-augmentation_false_dropout_false_bn_false_0.005-pesudo-naive_true_distilled_false_pesudo_false_interpolation_false_0.9_0.5-lr-5_1_1_freezing_false_32_.01-0.4715-0.6918-0.6539-0.3392.hdf5' 
        target_size=(64,64)
    else:
        MODEL='vggFace'
        EGA_path='/home/yanhong/Downloads/next_step/HY_MTL/train_weights/CONFUSION/ferplus-adience/EmotionAgenderNetVGGFace_vgg16/10_0.51-0.89-0.7.hdf5'
        EPA_path='train_weights/CONFUSION/expw-aflw-adience/EPA_VGGFace_vgg16/2_.01-0.63-0.78-0.83.hdf5'
        EPGA_path = 'train_weights/CONFUSION/fer2013-adience-aflw/EPGA-VGGFace_vgg16/net-augmentation_false_dropout_false_bn_false_0.005-pesudo-naive_true_distilled_false_pesudo_false_interpolation_false_0.9_0.5-lr-5_1_1_freezing_false_32_.01-0.6412-0.7285-0.9398-0.7813.hdf5'
        target_size=(224,224)
        # EPGA_path='train_weights/CONFUSION/ferplus-adience-aflw/EPGA-VGGFace_vgg16/net-augmentation_false_dropout_false_bn_false_0.005-pesudo-naive_true_distilled_false_pesudo_false_interpolation_false_0.9_0.5-lr-5_1_1_freezing_false_32_.01-0.8091-0.7566-0.9489-0.7716.hdf5'
    

    if is_EGA:
        EGA_multi_model = Net(MODEL,1,4,is_dropout,8,5,8,2)
        EGA_multi_model.load_weights(EGA_path)
    elif is_EPGA:
        EPGA_multi_model = Net(MODEL,1,12,7,5,8,2,False,False,None)
        EPGA_multi_model.load_weights(EPGA_path)
        
    else:
        EPA_multi_model = Net(MODEL,1,9,is_dropout,7,5,8,2)
        EPA_multi_model.load_weights(EPA_path)
    
    



EPGA_multi_model.summary()
emotion_window = []
gender_window = []
pose_window = []
age_window = []


# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')


start = time.clock()
face_cascade = cv2.CascadeClassifier('/home/user/anaconda3/pkgs/libopencv-3.4.2-hb342d67_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image,1.3, 5)
print(np.shape(faces))
# faces = detect_faces(face_detection, gray_image)
for face_coordinates in faces:
    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    rgb_face = rgb_image[y1:y2, x1:x2]
    gray_face = gray_image[y1:y2, x1:x2]
    try:
        if not is_light_net:
            face = cv2.resize(rgb_face,(target_size),interpolation=cv2.INTER_CUBIC)
            face = preprocess_input(face, False)
            face = np.expand_dims(face, 0)
        else:
            face = cv2.resize(gray_face, (target_size),interpolation=cv2.INTER_CUBIC)
            face = preprocess_input(face, False)
            face = np.expand_dims(face, 0)
            face = np.expand_dims(face, -1)
    except:
        continue

    start_time = time.clock()
    if is_single_task:
        emotion_label = emotion_model.predict(face)
        gender_label = gender_model.predict(face)
        pose_label = pose_model.predict(face)
        age_label= age_model.predict(face)   
    else:
        if is_EGA:
            emotion_label,gender_label,age_label = EGA_multi_model.predict(face)
        elif is_EPA:
            emotion_label,pose_label,age_label=EPA_multi_model.predict(face)
        elif is_EPGA:
            print('here')
            emotion_label,pose_label,gender_label,age_label = EPGA_multi_model.predict(face)
    end_time = time.clock()
    print('spend_time:',end_time-start_time)

    if is_single_task:
        emotion_label_arg = np.argmax(emotion_label)
        gender_label_arg = np.argmax(gender_label)
        age_label_arg = np.argmax(age_label)
        pose_label_arg = np.argmax(pose_label)
        emotion_text = emotion_labels[emotion_label_arg]
        age_text = age_labels[age_label_arg]
        gender_text = gender_labels[gender_label_arg]
        pose_text = pose_labels[pose_label_arg]
    else:
        if is_EGA:
            emotion_label_arg = np.argmax(emotion_label)
            gender_label_arg = np.argmax(gender_label)
            age_label_arg = np.argmax(age_label)
            emotion_text = emotion_labels[emotion_label_arg]
            age_text = age_labels[age_label_arg]
            gender_text = gender_labels[gender_label_arg]
        elif is_EPGA:
            emotion_label_arg = np.argmax(emotion_label)
            pose_label_arg = np.argmax(pose_label)
            gender_label_arg = np.argmax(gender_label)
            age_label_arg = np.argmax(age_label)
            emotion_text = emotion_labels[emotion_label_arg]
            pose_text = pose_labels[pose_label_arg]
            age_text = age_labels[age_label_arg]
            gender_text = gender_labels[gender_label_arg]

        else:
            emotion_label_arg = np.argmax(emotion_label)
            pose_label_arg = np.argmax(pose_label)
            age_label_arg = np.argmax(age_label)
            emotion_text = emotion_labels[emotion_label_arg]
            age_text = age_labels[age_label_arg]
            pose_text = pose_labels[pose_label_arg]



    

    emotion_window.append(emotion_text)
    age_window.append(age_text)
    if is_single_task:
        gender_window.append(gender_text)
        pose_window.append(pose_text)
    else:
        if is_EGA:
            gender_window.append(gender_text)
        elif is_EPGA:
            gender_window.append(gender_text)
            pose_window.append(pose_text)

        else:
            pose_window.append(pose_text)


    if len(age_window) > frame_window:
        emotion_window.pop(0)
        age_window.pop(0)
        if is_single_task:
            gender_window.pop(0)
            pose_window.pop(0)
        else:
            if is_EGA:
                gender_window.pop(0)
            else:
                pose_window.pop(0)
    try:
        emotion_mode = mode(emotion_window)
        age_mode = mode(age_window)
        if is_single_task:
            gender_mode = mode(gender_window)
            pose_mode = mode(pose_window)
        else:
            if is_EGA:
                gender_mode = mode(gender_window)
            elif is_EPGA:
                gender_mode = mode(gender_window)
                pose_mode = mode(pose_window)
            else:
                pose_mode = mode(pose_window)
    except:
        continue

    color = [255,255,255]
    color_emotion = (0,0,255)
    color_gender = (0,255,0)
    color_age = (255,0,0)
    color_pose = (255,255,0)


    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_mode,
              color_emotion, text_left, text_top, font_size,font_thickness)
    draw_text(face_coordinates, rgb_image, age_mode,
              color_age, text_left, text_top+text_gap, font_size,font_thickness)
    if is_single_task:
        draw_text(face_coordinates, rgb_image, gender_mode,
                  color_gender, text_left, text_top+text_gap*2, font_size,font_thickness)
        draw_text(face_coordinates, rgb_image, pose_mode,
                  color_pose, text_left, text_top+text_gap*3, font_size,font_thickness)
    else:
        if is_EGA:
            draw_text(face_coordinates, rgb_image, gender_mode,
                  color_gender, text_left, text_top+text_gap*2, font_size,font_thickness)
        elif is_EPGA:
            draw_text(face_coordinates, rgb_image, gender_mode,
                  color_gender, text_left, text_top+text_gap*2, font_size,font_thickness)
            draw_text(face_coordinates, rgb_image, pose_mode,
                  color_pose, text_left, text_top+text_gap*3, font_size,font_thickness)
        else:
            draw_text(face_coordinates, rgb_image, pose_mode,
                  color_pose, text_left, text_top+text_gap*2, font_size,font_thickness)
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(stored_path, bgr_image)

