import os
import pandas as pd
from glob import glob
import numpy as np

# Importing Keras and other pre-processing libraries
from keras import backend
from keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, LSTM, CuDNNLSTM
from keras import regularizers, optimizers
import keras.backend as K
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path
from tqdm import tqdm
from tensorflow.python.client import device_lib
import tensorflow as tf

print(device_lib.list_local_devices())

#WARNING: USE ONLY WHEN RUNNING WITH GPUs WITH CUDA ENABLED
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)

def append_ext(fn):
    return fn.rstrip('.wav')+".jpg"

def create_spectrogram(filename, name, folder):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = 'images/' + folder + '/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


Data_dir=np.array(glob("train/*"))

# CAREFUL FOR ERROR HERE
# for file in tqdm(Data_dir, total= Data_dir.shape[0]):
#     filename,name = file, file.split('/')[-1].split('.')[0]
#     create_spectrogram(filename,name, 'train')

traindf=pd.read_csv('images/meta/train.csv',dtype=str)
testdf=pd.read_csv('images/meta/test.csv',dtype=str)
traindf["fname"]=traindf["fname"].apply(append_ext)
testdf["fname"]=testdf["fname"].apply(append_ext)

datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

train_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="images/train/",
    x_col="fname",
    y_col="label",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

valid_generator=datagen.flow_from_dataframe(
    dataframe=traindf,
    directory="images/train/",
    x_col="fname",
    y_col="label",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

model = Sequential()

#STACK of Conv Neural Nets
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(41, activation='softmax'))
model.compile(optimizers.rmsprop(lr=0.0005, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

#Fitting keras model, no test gen for now
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=1)

#Evaluate GEnerator
model.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_VALID)

#Testing

Test_dir=np.array(glob("test/*"))

# for file in tqdm(Test_dir[0:10], total= Test_dir.shape[0]):
#     filename, name = file, file.split('/')[-1].split('.')[0]
#     create_spectrogram(filename, name, 'test')

test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="images/test/",
    x_col="fname",
    y_col="label",
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode="categorical",
    target_size=(64,64))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()
loss, acc =  model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)

print(loss, acc)
