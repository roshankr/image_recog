import numpy as np
np.random.seed(2016)
import os
import glob
import cv2
import datetime
import time
import sys
import pandas as pd
import warnings
import platform
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold,StratifiedShuffleSplit
from keras.models import Sequential,Model
from keras.layers import Input,BatchNormalization,merge
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D,AveragePooling2D
from keras.optimizers import Adam, Adagrad, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf1
from sklearn.metrics import log_loss
import time as tm
from sklearn import preprocessing
from keras.applications.inception_v3 import InceptionV3
from keras.applications import vgg16
from sklearn.datasets import fetch_olivetti_faces
from keras.applications.imagenet_utils import  _obtain_input_shape

########################################################################################################################
#Generalized Deep learning Model
########################################################################################################################
#usage
#python Deep_Learning_v01.py train gpu --> For training
#python Deep_Learning_v01.py predict gpu  --> For classification
#######################################################################################################################
#conv2d layer with batch normalization for Inception model
########################################################################################################################
def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(axis=channel_axis, name=bn_name)(x)
    return x

########################################################################################################################
#Inception Network
#https://4.bp.blogspot.com/-TMOLlkJBxms/Vt3HQXpE2cI/AAAAAAAAA8E/7X7XRFOY6Xo/s1600/image03.png
########################################################################################################################
def CNN_Inceptionv03_Classifier(img_rows, img_cols, color_type,num_category):
    global predict_proba
    predict_proba  = False

    # Determine proper input shape
    input_shape1 = _obtain_input_shape(input_shape,
                                      default_size=299,
                                      min_size=139,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=None)

    img_input = Input(shape=input_shape1)

    x = conv2d_bn(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    x = conv2d_bn(x, 32, 3, 3, border_mode='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, border_mode='valid')
    x = conv2d_bn(x, 192, 3, 3, border_mode='valid')
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    for i in range(3):
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(i))

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
                             subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch3x3dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          subsample=(2, 2), border_mode='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
                            subsample=(2, 2), border_mode='valid')

    branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch7x7x3, branch_pool],
              mode='concat', concat_axis=channel_axis,
              name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = merge([branch3x3_1, branch3x3_2],
                          mode='concat', concat_axis=channel_axis,
                          name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
                             mode='concat', concat_axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), border_mode='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                  mode='concat', concat_axis=channel_axis,
                  name='mixed' + str(9 + i))

    # Classification block
    x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
    x = Flatten(name='flatten')(x)
    #x = Dense(1000, activation='softmax', name='predictions1')(x)
    x = Dense(num_category, activation='softmax', name='predictions')(x)

    model = Model(input =img_input, output=x, name='inception_v3')
    sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print(model.summary())

    return model

########################################################################################################################
#Inception Network in-built in Keras, can be used with pretrained weights or without weights
########################################################################################################################
def CNN_Inceptionv03_inbuilt_Classifier(img_rows, img_cols, color_type,num_category):
    global predict_proba
    predict_proba  = False

    print('Loading InceptionV3 Weights ...')
    InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',input_tensor=None,input_shape=input_shape)

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = InceptionV3_notop.get_layer(index = -1).output
    output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
    output = Flatten(name='flatten')(output)
    output = Dense(num_category, activation='softmax', name='predictions')(output)
    model = Model(InceptionV3_notop.input, output)

    print(model.summary())

    optimizer = SGD(lr = 1e-3, momentum = 0.9, decay = 0.0, nesterov = True)
    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    return model

########################################################################################################################
#Set up VGG16 model
########################################################################################################################
def CNN_VGG16_Classifier(img_rows, img_cols, color_type,num_category):
    global predict_proba

    predict_proba  = False
    img_input = Input(shape=input_shape)

    # Block 1
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    #Block 2
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
    x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    #Block 3
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
    x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    #Block 4
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #Block 5
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
    x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_category, activation='softmax', name='predictions')(x)

    model = Model(input =img_input, output=x, name='vgg16')
    #optimizer = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    #optimizer = Adam(lr=1e-3)
    optimizer = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model

########################################################################################################################
#Set up VGG16 , inbuilt in Keras
########################################################################################################################
def CNN_VGG16_inbuilt_Classifier(img_rows, img_cols, color_type,num_category):
    global predict_proba
    predict_proba  = False

    print('Loading VGG16 Weights ...')

    VGG16_notop = vgg16.VGG16(include_top=False, weights=None,
          input_tensor=None, input_shape=input_shape)

    print('Adding Average Pooling Layer and Softmax Output Layer ...')
    output = VGG16_notop.get_layer(index = -1).output

    output = Flatten(name='flatten')(output)
    output = Dense(96, activation='relu',init='he_uniform')(output)
    output = Dropout(0.4)(output)
    output = Dense(24, activation='relu',init='he_uniform')(output)
    output = Dropout(0.2)(output)
    output = Dense(num_category, activation='softmax')(output)
    model = Model(VGG16_notop.input, output)

    print(model.summary())

    optimizer = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=True)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    #print(model.summary())

    return model

########################################################################################################################
#Set up parms for a customized CNN model 1
########################################################################################################################
def CNN_Classifier1(img_rows, img_cols, color_type,num_category):

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(8, 3, 3, activation='relu', init='he_uniform'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(96, activation='relu',init='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='relu',init='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(num_category, activation='softmax'))

    optimizer = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=True)
    #optimizer = Adagrad(lr=1e-3, epsilon=1e-08)
    #optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model

########################################################################################################################
#Merge predicted outputs from multiple folds (simple avg ensembling)
########################################################################################################################
def Merge_CV_folds_mean(data, nfolds):

    print("Merge predicted outputs....")
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

########################################################################################################################
#Create final output file (after classification)
########################################################################################################################
def Create_final_output_file(predictions, test_id):

    print("Create final predicted dataset....")
    predictions = np.clip(predictions,0.02, 0.98, out=None)

    temp_pred = pd.DataFrame(predictions)
    cols = lbl_y.inverse_transform(temp_pred.columns)

    pred_DF = pd.DataFrame(predictions,columns=cols)
    pred_DF['Predicted'] = pred_DF.idxmax(axis=1).str.strip()
    pred_DF.insert(0, 'image', test_id)
    pred_DF['Actual'] = pred_DF['image'].str.split('_', 1).str[0].str.strip()

    pred_DF['match'] = np.where(pred_DF['Predicted']==pred_DF['Actual'],1,0)

    print("Accuracy pct is " + str(len(pred_DF[pred_DF['match'] ==1])*100/float(len(pred_DF))))

    now = datetime.datetime.now()

    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join(file_path_orig,'results', 'result_' + suffix + '.csv')
    pred_DF.to_csv(sub_file, index=False)

########################################################################################################################
#data_augmentation
########################################################################################################################
def data_augmentation(input_data,input_target,type):

    if type == "train":
        # this is the augmentation configuration we will use for training
        aug_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            rotation_range=10.,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)

        # aug_datagen = ImageDataGenerator(
        #     rotation_range=40,
        #     width_shift_range=0.2,
        #     height_shift_range=0.2,
        #     shear_range=0.2,
        #     zoom_range=0.2,
        #     horizontal_flip=True,
        #     fill_mode='nearest')

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        aug_datagen.fit(input_data)

        aug_generator = aug_datagen.flow(
            input_data,
            input_target,
            batch_size = batch_size,
            shuffle = True
            #save_to_dir=os.path.join(file_path,'train_aug'),
            #save_prefix='train_'
            )

    if type == "validation":
        # this is the augmentation configuration we will use for validation:
        # only rescaling
        aug_datagen = ImageDataGenerator(rescale=1./255)

        aug_generator = aug_datagen.flow(
            input_data,
            input_target,
            batch_size = batch_size,
            shuffle = True
            )

    if type == "predict":
        # this is the augmentation configuration we will use for validation:
        # only rescaling
        aug_datagen = ImageDataGenerator(rescale=1./255)

        aug_generator = aug_datagen.flow(
            input_data,
            None,
            batch_size = batch_size,
            shuffle = False
            )

    return aug_generator

########################################################################################################################
#Choose the model as per parm
########################################################################################################################
def Get_model(num_category):

    if model_classifier == "Inception":
        clf = CNN_Inceptionv03_Classifier(img_rows, img_cols, color_type_global,num_category)
    elif model_classifier == "Classifier1":
        clf = CNN_Classifier1(img_rows, img_cols, color_type_global,num_category)
    elif model_classifier == "VGG16":
        clf = CNN_VGG16_Classifier(img_rows, img_cols, color_type_global,num_category)
    elif model_classifier == "Inception_inbuilt":
        clf = CNN_Inceptionv03_inbuilt_Classifier(img_rows, img_cols, color_type_global,num_category)
    elif model_classifier == "VGG16_inbuilt":
        clf = CNN_VGG16_inbuilt_Classifier(img_rows, img_cols, color_type_global,num_category)
    else:
        clf = CNN_Classifier1(img_rows, img_cols, color_type_global,num_category)

    return  clf

########################################################################################################################
#Train the model nfold times, so we will have 'n' models (as we do k fold cross validation)
########################################################################################################################
def Nfold_Training(X, y, target_vect):

    print("Starting Model Training....... at Time: %s" %(tm.strftime("%H:%M:%S")))
    start_time  = time.time()
    random_state = 21

    num_category = len(pd.DataFrame(y).columns)
    print("Number of categories to predict: "+ str(num_category))

    yfull_train = dict()
    yfull_test = []
    num_fold = 0
    sum_score = 0

    X =np.array(X)
    scores=[]

    X, y = shuffle(X, y)
    #ss = StratifiedShuffleSplit(target_vect, n_iter=nfolds, test_size=(1.0/nfolds),random_state=random_state)
    ss = KFold(len(y), n_folds=nfolds,shuffle=True,random_state=random_state)

    i = 1

    for trainCV, testCV in ss:
        X_train, X_test= X[trainCV], X[testCV]
        Y_train, Y_test= y[trainCV], y[testCV]

        X_train = X_train.astype('float32')
        X_test  = X_test.astype('float32')

        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_test), len(Y_test))

        clf = Get_model(num_category)

        #kfold_weights_path = os.path.join(file_path, 'cache', 'weights_kfold_' + str(num_fold) + '.h5')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, verbose=0)
           #,ModelCheckpoint(best_model_file, monitor='val_loss', save_best_only=True, verbose=0),
        ]

        if data_aug:
            print("using fit_generator")
            train_generator = data_augmentation(X_train,Y_train,type="train")
            validation_generator = data_augmentation(X_test,Y_test,type="validation")

            clf.fit_generator(
                    train_generator,
                    samples_per_epoch = len(X_train)*10,
                    nb_epoch = nb_epoch,
                    validation_data = validation_generator,
                    nb_val_samples = len(X_test),
                    callbacks=callbacks)

        else:
            clf.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                      shuffle=True, verbose=1, validation_data=(X_test, Y_test),
                      callbacks=callbacks)

        model_fn = os.path.join(file_path_orig,'models/') + 'model_iteration_' + str(i)+".h5"
        clf.save_weights(model_fn)

        #predict validation dataset
        if data_aug:
            Y_pred = clf.predict_generator(validation_generator, len(X_test))
        else:
            if predict_proba:
                Y_pred=clf.predict_proba(X_test,batch_size=batch_size, verbose=1)
            else:
                Y_pred = clf.predict(X_test, batch_size=batch_size, verbose=1)

        scores.append(log_loss(Y_test, Y_pred))

        print(" %d-iteration... %s " % (i,scores))

        i = i + 1

    #Average ROC from cross validation
    scores=np.array(scores)
    print ("Normal CV Score:",np.mean(scores))

    end_time  = time.time()
    print("Ending Model Training....... at Time: %s" %(tm.strftime("%H:%M:%S")))
    print("Training time taken for "+str(nfolds)+" models : " + str(int(end_time - start_time)) +" seconds")

    return yfull_test

########################################################################################################################
#Classify the model
########################################################################################################################
def Model_prediction(X, y,test_id):

    print("Starting Model Classification....... at Time: %s" %(tm.strftime("%H:%M:%S")))
    start_time  = time.time()

    random_state = 42
    num_category = len(pd.DataFrame(y).columns)

    yfull_test = []
    for model in glob.glob(os.path.join(file_path_orig,"models","*")):
        X_test =np.array(X)

        clf = Get_model(num_category)

        print("model "+str(model))
        clf.load_weights(model)

        if data_aug:

            validation_generator = data_augmentation(X_test,np.array(),type="predict")
            Y_pred = clf.predict_generator(validation_generator, len(X_test))
        else:
            if predict_proba:
                Y_pred=clf.predict_proba(X_test,batch_size=batch_size, verbose=1)
            else:
                Y_pred = clf.predict(X_test, batch_size=batch_size, verbose=1)

        yfull_test.append(Y_pred)

    test_res = Merge_CV_folds_mean(yfull_test, nfolds)

    Create_final_output_file(test_res, test_id)

    print("***************Ending Kfold Cross validation***************")

    end_time  = time.time()
    print("Classifcation time taken for "+str(nfolds)+" models : " + str(int(end_time - start_time)) +" seconds")

    return yfull_test

########################################################################################################################
#Set up the input_shape for theano and tensorflow
########################################################################################################################
def Data_Munging(process_data, process_target, process_target_vect, process_id):
    global input_shape, channel_axis

    #if K.image_dim_ordering() == 'th':
    if K.backend() == 'theano':
        print("using Theano model")
        process_data = process_data.reshape(process_data.shape[0], color_type_global, img_rows, img_cols)
        input_shape = (color_type_global, img_rows, img_cols)
        channel_axis = 1
    else:
        print("using Tensorflow model")
        process_data = process_data.reshape(process_data.shape[0], img_rows, img_cols, color_type_global)
        input_shape = (img_rows, img_cols, color_type_global)
        channel_axis = 3
        K.get_session().run(tf1.global_variables_initializer())

    return process_data, process_target,process_target_vect, process_id

########################################################################################################################
#Read image using open cv and convert to array
########################################################################################################################
def get_im_cv2_mod(path, img_rows, img_cols, color_type):

    # Load as grayscale
    if color_type == 1:
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)

    resized = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_LINEAR)

    return resized

########################################################################################################################
#Read and Load train data from sub folders
########################################################################################################################
def load_data(img_rows, img_cols, color_type,sub_folder):
    X_full = []
    X_full_id = []
    y_full = []
    start_time = time.time()

    print('Read images')
    for tf in sub_folder:
        path = os.path.join(file_path, tf, '*.*')
        files = glob.glob(path)
        print('Load folder %s , Total files :- %d' %(format(tf),len(files)))

        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
            X_full.append(img)
            X_full_id.append(tf+"_"+flbase)
            y_full.append(tf)

    print('Read data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_full, y_full, X_full_id

########################################################################################################################
#Read the images and convert to arrays
########################################################################################################################
def Get_image_data(img_rows, img_cols, color_type,sub_folder):

    global lbl_y

    process_data, process_target, process_id = load_data(img_rows, img_cols, color_type_global,sub_folder)

    process_data = np.array(process_data, dtype=np.uint8)

    lbl_y = preprocessing.LabelEncoder()
    lbl_y.fit(list(process_target))
    process_target = lbl_y.transform(process_target)
    process_target_vect = process_target

    process_target = np_utils.to_categorical(process_target)
    process_target = np.array(process_target, dtype=np.uint8)

    if color_type == 1:
        process_data = process_data.reshape(process_data.shape[0], 1, img_rows, img_cols)
    else:
        process_data = process_data.transpose((0, 3, 1, 2))

    process_data = process_data.astype('float32')
    process_data /= 255
    process_data -= 0.5
    process_data *= 2.

    print('Train shape:', process_data.shape)
    print(process_data.shape[0], 'train samples')

    return process_data, process_target, process_target_vect, process_id

########################################################################################################################
#Data cleansing , feature scaling , splitting
########################################################################################################################
def Data_Loading():

    print("Starting Data Loading....... at Time: %s" % (tm.strftime("%H:%M:%S")))
    if load_data_flag:
        #Load Data from Sklearn
        faces = fetch_olivetti_faces()
        targets = faces.target

        faces = faces.images.reshape((len(faces.images), -1))
        train_data = faces[targets < 30]
        test_data =  faces[targets >= 30]
        full_data =  faces

    else:
        sub_folder = []
        for folder in glob.glob(os.path.join(file_path,"*/")):
            sub_folder.append(os.path.basename(os.path.dirname(folder)))

        full_data, full_target, full_target_vect, full_id = Get_image_data(img_rows, img_cols, color_type_global,sub_folder)

    print("Ending Data Loading....... at Time: %s" % (tm.strftime("%H:%M:%S")))

    return full_data, full_target, full_target_vect, full_id

########################################################################################################################
#Main module                                                                                                           #
########################################################################################################################
def main(argv):

    pd.set_option('display.width', 200)
    pd.set_option('display.height', 500)

    warnings.filterwarnings("ignore")

    global file_path, use_cache, train_folder, test_folder, restore_from_last_checkpoint,\
        img_rows,img_cols,color_type_global,nb_epoch,batch_size,predict_proba,orig_input,data_aug,\
        load_data_flag,file_path_orig,nfolds,model_classifier

    use_cache = 0
    restore_from_last_checkpoint = 0

    color_type_global = 3

    predict_proba = True
    train_folder = 'train'
    test_folder = 'test'
    data_aug = False
    load_data_flag = False

    #Key parms to set up
    ####################################################################################################################
    img_rows, img_cols = 64, 64
    #img_rows, img_cols = 299, 299
    batch_size = 64
    nb_epoch = 100
    nfolds=10
    model_classifier  = "Classifier1"
    ####################################################################################################################

    if(platform.system() == "Windows"):
        file_path_orig = 'C:\\Python\\Others\\data\\att_faces'
    else:
        #aws
        #file_path = '/DS/deep_learning/data/att_faces'
        file_path_orig = '/DS/deep_learning/data/att_faces'
        #file_path_orig = '/DS/deep_learning/data/lfw'
        #file_path_orig =  '/mnt/hgfs/Python/Others/data/att_faces/'
        #file_path_orig = '/home/roshan/Desktop/DS/Others/data/Kaggle/att_faces/'

    try:
        process_type = argv[1]
    except:
        process_type = 'train'

    try:
        processor_type = argv[2]
    except:
        processor_type = 'gpu'

    if processor_type == 'cpu':
        processor_type = '/cpu:0'
    else:
        processor_type = '/gpu:0'

    if not(os.path.isdir(os.path.join(file_path_orig, "models"))):
        os.makedirs(os.path.join(file_path_orig, "models"))

    if not(os.path.isdir(os.path.join(file_path_orig, "results"))):
        os.makedirs(os.path.join(file_path_orig, "results"))

    if process_type.lower().strip() == 'train':
        print("Training in progress....")
        file_path = os.path.join(file_path_orig, "train")
    else:
        print("Classification in progress....")
        file_path = os.path.join(file_path_orig, "test")

    with tf1.device(processor_type):
        full_data, full_target, full_target_vect, full_id = Data_Loading()
        process_data, process_target, process_target_vect, process_id = Data_Munging(full_data, full_target, full_target_vect, full_id)

        if process_type.lower().strip() == 'train':
                yfull_test = Nfold_Training(process_data, process_target,process_target_vect)

        else:
            Model_prediction(process_data, process_target,process_id)

########################################################################################################################
#Main program starts here                                                                                              #
########################################################################################################################
if __name__ == "__main__":
    main(sys.argv)


