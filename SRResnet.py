# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:26:36 2019

@author: khusmodi
"""

import os
import subprocess
if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)
    
import os
os.mkdir('data/hor')

import glob
import cv2
input_filename = glob.glob("data/train" + "/*-out.jpg")
counter = 1
for x in input_filename:
    if counter%2==0:
        rot = 0
    else:
        rot = -1
    img = cv2.imread(x)
    hor = img.copy()
    hor = cv2.flip(img, rot)
    cv2.imwrite("data/hor/horblah"+str(counter)+"-out.jpg", hor)
    cv2.imwrite("data/hor/horblahor"+str(counter)+"-out.jpg", img)
    inp = x.replace("-out.jpg", "-in.jpg")
    img = cv2.imread(inp)
    hor = img.copy()
    hor = cv2.flip(img, rot)
    cv2.imwrite("data/hor/horblah"+str(counter)+"-in.jpg", hor)
    cv2.imwrite("data/hor/horblahor"+str(counter)+"-in.jpg", img)
    counter+=1

import numpy as np

import wandb
from wandb.keras import WandbCallback
import random
import glob
import subprocess
import os
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

import keras
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import BatchNormalization, Activation, PReLU
from tensorflow.keras.layers import Input, Flatten, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg19 import VGG19

from shutil import rmtree
import keras.callbacks.LearningRateSchedule as LRS
#from batch_generator import COCOBatchGenerator
#from bsd100_callback import BSD100_Evaluator
#from utils import print_available_devices, deprocess_HR, deprocess_LR

import matplotlib.pyplot as plt

### setup ###
data_format = 'channels_last'
keras.backend.set_image_data_format('channels_last')

print("Keras : ", keras.__version__)
print("\t data_format : ", keras.backend.image_data_format())
print("Tensorflow : ", tf.__version__)



B = 16 # number of residual block

batch_size = 4
target_size = (256,256)
downscale_factor = 1

num_steps = 200000
steps_per_epoch = 625
epochs = 601

# axis used in Parametric ReLU !
shared_axis = [1,2] if data_format == 'channels_last' else [2,3]

# axis for Batch Normalization
axis = -1 if data_format == 'channels_last' else 1

print('epochs = ', epochs)

run = wandb.init(project='superres')
config = run.config
config.num_epochs = epochs
config.batch_size = batch_size
config.input_height = 32
config.input_width = 32
config.output_height = 256
config.output_width = 256

# build a residual block
def res_block(inputs):
    x = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(inputs)
    x = BatchNormalization(axis=axis)(x)
    #x = Dropout(0.3)(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=shared_axis)(x)
    x = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(x)
    x = BatchNormalization(axis=axis)(x)
    return add([x, inputs])

# build an upscale block
# PixelShuffler is replaced by an UpSampling2D layer (nearest upsampling)

def up_block(x):
    x = Conv2D(256, kernel_size=(3,3), strides=(1,1) , padding='same', activation=None, use_bias=False)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=shared_axis)(x)
    return x

input_generator = Input(shape=(32, 32, 3) if data_format=='channels_last' else (3, None, None), 
                        name='input_generator')

x = Conv2D(filters=128, kernel_size=(9,9),
           strides=(1,1), padding='same',
           activation=None)(input_generator)

x_input_res_block = PReLU(alpha_initializer='zeros',
                          alpha_regularizer=None,
                          alpha_constraint=None,
                          shared_axes=shared_axis)(x)

x = x_input_res_block

# add B residual blocks 
for i in range(B):
    x = res_block(x)

x = Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(x)
x = BatchNormalization(axis=axis)(x)

# skip connection
x = add([x, x_input_res_block])

# two upscale blocks
x = up_block(x)
x = up_block(x)
x = up_block(x)

# final conv layer : activated with tanh -> pixels in [-1, 1]
output_generator = Conv2D(3, kernel_size=(9,9), 
                          strides=(1,1), activation='tanh',
                          use_bias=False, padding='same')(x)

generator = Model(inputs=input_generator, outputs=output_generator)


val_dir = 'data/test'
train_dir = 'data/hor'

config.steps_per_epoch = len(
    glob.glob(train_dir + "/*-in.jpg")) // config.batch_size
config.val_steps_per_epoch = len(
    glob.glob(val_dir + "/*-in.jpg")) // config.batch_size

def image_generator(batch_size, img_dir):
    """A generator that returns small images and large images.  DO NOT ALTER the validation set"""
    input_filenames = glob.glob(img_dir + "/*-in.jpg")
    counter = 0
    random.shuffle(input_filenames)
    while True:
        small_images = np.zeros(
            (batch_size, config.input_width, config.input_height, 3))
        large_images = np.zeros(
            (batch_size, config.output_width, config.output_height, 3))
        if counter+batch_size >= len(input_filenames):
            counter = 0
            random.shuffle(input_filenames)
        for i in range(batch_size):
            img = input_filenames[counter + i]
            small_images[i] = np.array(Image.open(img)) / 255.0
            large_images[i] = np.array(
                Image.open(img.replace("-in.jpg", "-out.jpg"))) / 255.0
        yield (small_images, large_images)
        counter += batch_size
        
        
def perceptual_distance(y_true, y_pred):
    """Calculate perceptual distance, DO NOT ALTER"""
    y_true *= 255
    y_pred *= 255
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]

    return K.mean(K.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))

val_generator = image_generator(config.batch_size, val_dir)
in_sample_images, out_sample_images = next(val_generator)
class ImageLogger(Callback):
    def __init__(self, gene, N):
        self.model = gene
        self.N = N
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)
        if epoch%self.N ==0:
            name1 = '128regcrossmae1deepweights%08d.h5' % epoch
            name2 = '128regcrossmae1deepmodel%08d.h5' % epoch
            self.model.save_weights(name1)
            self.model.save(name2)
        if logs.get('val_perceptual_distance') < 39.0:
            name1 = '39deepweights%08d.h5' % epoch
            name2 = '39deepmodel%08d.h5' % epoch
            self.model.save_weights(name1)
            self.model.save(name2)
            self.model.stop_training = True

opt = Adam(lr=0.0001, beta_1=0.9)
#tpu_model = tf.contrib.tpu.keras_to_tpu_model(generator, strategy=tf.contrib.tpu.TPUDistributionStrategy(tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])))
generator.compile(loss='mae', optimizer=opt, metrics=[perceptual_distance])


generator.fit_generator(image_generator(config.batch_size, train_dir),
                            steps_per_epoch=config.steps_per_epoch,
                            epochs=config.num_epochs, callbacks=[
                            ImageLogger(generator, 15), WandbCallback()],
                            validation_steps=config.val_steps_per_epoch,
                            validation_data=(val_generator))
