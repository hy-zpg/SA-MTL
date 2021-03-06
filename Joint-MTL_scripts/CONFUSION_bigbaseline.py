import argparse
import os 
import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ModelCheckpoint, CSVLogger,LearningRateScheduler, TensorBoard,EarlyStopping,ReduceLROnPlateau
from sklearn.model_selection import KFold
# from model.inceptionv3 import EmotionNetInceptionV3
# from model.mobilenetv2 import MultitaskMobileNetV2
# from model.vggface import MultitaskVGGFacenet
# from model.mini_xception import EmotionNetmin_XCEPTION
from utils.datasets import DataManager 
from utils.confusion_MTL.confusion_bigbaseline_generator import DataGenerator
from utils.callback import DecayLearningRate
from model.models import Net
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_emotion',
                    choices=['fer2013','ferplus','sfew','expw'],
                    default='expw',
                    help='Model to be used')
parser.add_argument('--dataset_pose',
                    choices=['fer2013','ferplus','sfew','expw'],
                    default='aflw',
                    help='Model to be used')
parser.add_argument('--dataset_attr',
                    choices=['imdb','adience','sfew'],
                    default='celeba',
                    help='Model to be used')
parser.add_argument('--dataset_age_gender',
                    choices=['imdb','adience','sfew'],
                    default='adience',
                    help='Model to be used')

#add training trick
parser.add_argument('--patience',
                    default=10,
                    type=int,
                    help='Number of traing epoch')
parser.add_argument('--is_augmentation',
                     type= ast.literal_eval,
                    help='whether data augmentation')
parser.add_argument('--is_dropout',
                     type= ast.literal_eval,
                    help='whether dropot')
parser.add_argument('--is_bn',
                     type= ast.literal_eval,
                    help='whether bn')
parser.add_argument('--weights_decay',
                    default=0.005,
                     type= float,
                    help='dense layer weights decay')

#add training strategy
parser.add_argument('--is_freezing',
                     type= ast.literal_eval,
                    help='whether pesudo-label selection')
parser.add_argument('--no_freezing_epoch',
                    default=32,
                    type=int,
                    help='starting no freezing')
parser.add_argument('--E_loss_weights',
                    default= 5,
                    type= float,
                    help='emotion')
parser.add_argument('--P_loss_weights',
                    default= 1,
                    type= float,
                    help='pose')


parser.add_argument('--model',
                    choices=['inceptionv3', 'ssrnet', 'mobilenetv2','vggFace','mini_xception'],
                    default='mini_xception',
                    help='Model to be used')
parser.add_argument('--epoch',
                    default=50,
                    type=int,
                    help='Num of training epoch')
parser.add_argument('--batch_size',
                    default=64,
                    type=int,
                    help='Size of data batch to be used')
parser.add_argument('--num_worker',
                    default=4,
                    type=int,
                    help='Number of worker to process data')


def load_data(dataset_emotion,dataset_pose,dataset_attr,dataset_age_gender):
    emotion = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_emotion) )
    pose = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_pose) )
    attr = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_attr) )
    gender_age = pd.read_csv('data/db/{}_cleaned.csv'.format(dataset_age_gender) )

    data_emotion = emotion
    data_pose = pose
    data_attr = attr
    data_gender_age = gender_age
    del emotion,pose,attr,gender_age

    paths_emotion = data_emotion['full_path'].values
    emotion_label = data_emotion['emotion'].values.astype('uint8')

    paths_pose = data_pose['full_path'].values
    # roll_label = data_pose['roll'].values.astype('float64')
    # pitch_label = data_pose['pitch'].values.astype('float64')
    # yaw_label = data_pose['yaw'].values.astype('float64')
    # pose_label = [roll_label,pitch_label,yaw_label]
    # pose_label = np.transpose(pose_label)
    pose_label = data_pose['pose'].values.astype('uint8')

    paths_gender_age = data_gender_age['full_path'].values
    gender_label = data_gender_age['gender'].values.astype('float64')
    age_label = data_gender_age['age'].values.astype('float64')

    paths_attr = data_attr['full_path'].values
    attr_label = []
    for i in range(40):
        attr_label.append(data_attr['{}'.format(i)].values.astype('uint8'))
    attr_labels = [attr_label[i] for i in range(40)]
    attr_labels = np.transpose(attr_labels)
    return paths_emotion, paths_pose,paths_attr,paths_gender_age, emotion_label,pose_label,attr_labels,gender_label,age_label

def mae(y_true, y_pred):
    return K.mean(K.abs(K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_pred, axis=1) -
                        K.sum(K.cast(K.arange(0, 101), dtype='float32') * y_true, axis=1)), axis=-1)

def freeze_all_but_mid_and_top(model):
    for layer in model.layers[:19]:
        layer.trainable = False
    for layer in model.layers[19:]:
        layer.trainable = True
    return model


class ATTR_AVG(keras.callbacks.Callback):
    def __init__(self,validation_data,interval=1,attr_avg=0):
        self.interval=interval
        self.x_val,self.y_val=validation_data
        self.attr_avg = attr_avg
    def on_epoch_end(self,epoch, logs={}):
        if epoch % self.interval == 0:
            y_score=self.model.predict(self.x_val,verbose=0)
            for i in range(40):
                self.attr_avg+=y_score[i+2]
            self.attr_avg = self.attr_avg / 40
        print(self.attr_avg)



def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
    args = parser.parse_args()
    MODEL = args.model
    EPOCH = args.epoch
    PATIENCE = args.patience
    BATCH_SIZE = args.batch_size
    NUM_WORKER = args.num_worker
    EMOTION_DATASET = args.dataset_emotion
    POSE_DATASET = args.dataset_pose
    ATTR_DATASET = args.dataset_attr

    if EMOTION_DATASET == 'ferplus':
        emotion_classes = 8
    else:
        emotion_classes = 7

    
    gender_classes = 2
    age_classes = 8
    pose_classes = 5
   



    paths_emotion, paths_pose,paths_attr,paths_gender_age, emotion_label,pose_label,attr_labels,gender_label,age_label = load_data(EMOTION_DATASET,POSE_DATASET,ATTR_DATASET,args.dataset_age_gender)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    kf_split_emotion = kf.split(paths_emotion)
    kf_split_pose = kf.split(paths_pose)
    kf_split_attr = kf.split(paths_attr)
    kf_split_gender_age = kf.split(paths_gender_age)

    emotion_kf = [[emotion_train_idx,emotion_test_idx] for emotion_train_idx,emotion_test_idx in kf_split_emotion]
    pose_kf = [[pose_train_idx,pose_test_idx] for pose_train_idx,pose_test_idx in kf_split_pose]
    attr_kf = [[attr_train_idx,attr_test_idx] for attr_train_idx,attr_test_idx in kf_split_attr]
    gender_age_kf = [[gender_age_train_idx,gender_age_test_idx] for gender_age_train_idx,gender_age_test_idx in kf_split_gender_age]

    # for emotion_train_idx,emotion_test_idx in kf_split_emotion:
    #     for gender_age_train_idx,gender_age_test_idx in kf_split_gender_age:
            # print(emotion_train_idx,emotion_test_idx,gender_age_train_idx,gender_age_test_idx)
    emotion_train_idx,emotion_test_idx = emotion_kf[0]
    pose_train_idx,pose_test_idx = pose_kf[0]
    attr_train_idx,attr_test_idx = attr_kf[0]
    gender_age_train_idx,gender_age_test_idx = gender_age_kf[0]


    print(len(emotion_train_idx),len(pose_train_idx),len(attr_train_idx),len(gender_age_train_idx))
    print(len(emotion_test_idx),len(pose_test_idx),len(attr_test_idx),len(gender_age_test_idx))


    train_emotion_paths = paths_emotion[emotion_train_idx]
    train_emotion = emotion_label[emotion_train_idx]
    test_emotion_paths = paths_emotion[emotion_test_idx]
    test_emotion = emotion_label[emotion_test_idx]

    train_pose_paths = paths_pose[pose_train_idx]
    train_pose = pose_label[pose_train_idx]
    test_pose_paths = paths_pose[pose_test_idx]
    test_pose = pose_label[pose_test_idx]


    train_gender_paths = paths_gender_age[gender_age_train_idx]
    train_gender = gender_label[gender_age_train_idx]
    test_gender_paths = paths_gender_age[gender_age_test_idx]
    test_gender = gender_label[gender_age_test_idx]

    train_age_paths = paths_gender_age[gender_age_train_idx]
    train_age = age_label[gender_age_train_idx]
    test_age_paths = paths_gender_age[gender_age_test_idx]
    test_age = age_label[gender_age_test_idx]


    train_attr_paths = paths_attr[attr_train_idx]
    train_attr = attr_labels[attr_train_idx]
    test_attr_paths = paths_attr[attr_test_idx]
    test_attr = attr_labels[attr_test_idx]



    model = None
    if MODEL == 'vggFace':
        model = model = Net(MODEL,1,8,emotion_classes,pose_classes,age_classes,gender_classes,args.is_dropout,args.is_bn,args.weights_decay)
        model = freeze_all_but_mid_and_top(model)
        MODEL = model.name
    else:
        model = model = Net(MODEL,1,8,emotion_classes,pose_classes,age_classes,gender_classes,args.is_dropout,args.is_bn,args.weights_decay)
        MODEL = model.name


    def my_cross_loss(y_true, y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())
        loss = K.categorical_crossentropy(y_true, y_pred) * mask
        return K.sum(loss) / K.sum(mask)

    def my_bin_loss(y_true, y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())
        loss = K.binary_crossentropy(y_true, y_pred) * mask
        return K.sum(loss) / K.sum(mask)
    def my_mean_square(y_true,y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())
        loss = keras.losses.mean_squared_error(y_true,y_pred)* mask
        return K.sum(loss) / K.sum(mask)

    def my_acc(y_true, y_pred):
        mask = K.all(K.equal(y_true, 0), axis=-1)
        mask = 1 - K.cast(mask, K.floatx()) 
        acc = (K.cast(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)),K.floatx()))*mask
        return K.sum(acc)/K.sum(mask)

    if model.task_type == 7:
        loss_dict = {}
        metrics_dict = {}
        attr_predcition=[]
        loss =[]
        acc =[]
        loss_weights = []
        attr_predcition.append('emotion_prediction')
        loss.append(my_cross_loss)
        acc.append(my_acc)
        loss_weights.append(5)
        for i in range(40):
            attr_predcition.append('attr{}_predition'.format(i))
            loss.append(my_cross_loss)
            acc.append(my_acc)
            loss_weights.append(0.1) 
        loss_dict = dict(zip(attr_predcition, loss))
        metrics_dict = dict(zip(attr_predcition, acc))
        weights_path = './train_weights/CONFUSION/{}-{}/{}/'.format(EMOTION_DATASET,ATTR_DATASET,MODEL)
        logs_path = './train_log/CONFUSION/{}-{}/{}/'.format(EMOTION_DATASET,ATTR_DATASET,MODEL)
    
    elif model.task_type == 8:
        loss_dict = {}
        metrics_dict = {}
        attr_predcition=[]
        loss =[]
        acc =[]
        loss_weights=[]
        attr_predcition.append('emotion_prediction')
        attr_predcition.append('pose_prediction')
        attr_predcition.append('gender_prediction')
        attr_predcition.append('age_prediction')
        loss.append(my_cross_loss)
        loss.append(my_cross_loss)
        loss.append(my_cross_loss)
        loss.append(my_cross_loss)
        acc.append(my_acc)
        acc.append(my_acc)
        acc.append(my_acc)
        acc.append(my_acc)
        loss_weights.append(5)
        loss_weights.append(5)
        loss_weights.append(5)
        loss_weights.append(5)
        for i in range(40):
            attr_predcition.append('attr{}_predition'.format(i))
            loss.append(my_cross_loss)
            acc.append(my_acc) 
            loss_weights.append(0.1)
        loss_dict = dict(zip(attr_predcition, loss))
        metrics_dict = dict(zip(attr_predcition, acc))
        weights_path = './train_weights/CONFUSION/{}-{}-{}-{}/{}/'.format(EMOTION_DATASET,POSE_DATASET,ATTR_DATASET,args.dataset_age_gender,MODEL)
        logs_path = './train_log/CONFUSION/{}-{}-{}-{}/{}/'.format(EMOTION_DATASET,POSE_DATASET,ATTR_DATASET,args.dataset_age_gender,MODEL)



    model.summary()

    
    
    if not os.path.exists(weights_path):
        os.makedirs(weights_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    model_names = weights_path + '.{epoch:02d}-{val_emotion_prediction_my_acc:.2f}.hdf5'
    
    csv_name = logs_path + '.log'
    board_name = logs_path 
    checkpoint = ModelCheckpoint(model_names, verbose=1,save_weights_only = True,save_best_only=True)
    csvlogger=CSVLogger(csv_name)
    early_stop = EarlyStopping('val_loss', patience=PATIENCE)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(PATIENCE/2), verbose=1)
    tensorboard = TensorBoard(log_dir=board_name,batch_size=BATCH_SIZE)
    #attr_avg=ATTR_AVG(validation_data = DataGenerator(model,  test_emotion_paths, test_pose_paths,test_attr_paths,  test_emotion,test_pose,test_attr, BATCH_SIZE) )
    
    callbacks = [checkpoint,csvlogger,early_stop,reduce_lr,tensorboard]

    if MODEL == 'ssrnet':
        callbacks = [
            # ModelCheckpoint(
            #     "train_weight/{}-{val_gender_prediction_binary_accuracy:.4f}-{val_age_prediction_mean_absolute_error:.4f}.h5".format(
            #         MODEL),
            #     verbose=1, save_best_only=True, save_weights_only=True),
            ModelCheckpoint(MODEL, 'val_loss', verbose=1,save_best_only=True),
            CSVLogger('train_log/{}-{}.log'.format(MODEL, n_fold)),
            DecayLearningRate([30, 60])]
    
    
    model.compile(optimizer='adam', loss=loss_dict,loss_weights = loss_weights, metrics=metrics_dict)
    model.fit_generator(
        DataGenerator(model, train_emotion_paths,train_pose_paths, train_attr_paths,train_emotion, train_pose, train_attr,BATCH_SIZE),
        validation_data=DataGenerator(model,  test_emotion_paths, test_pose_paths,test_attr_paths,  test_emotion,test_pose,test_attr, BATCH_SIZE),
        epochs=EPOCH,
        verbose=2,
        workers=NUM_WORKER,
        use_multiprocessing=False,
        max_queue_size=int(BATCH_SIZE * 2),
        callbacks=callbacks
    )
    del  train_emotion_paths, train_pose_paths,train_attr_paths,train_emotion,train_pose, train_attr
    del  test_emotion_paths, test_pose_paths,test_attr_paths,test_emotion, test_pose,test_attr


if __name__ == '__main__':
    main()
