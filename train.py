#Setup
import os
os.system('pip3 install wandb')
os.system('wandb login 6a7dc326ddfe055cd3991cd259eeeb789c3d8c34')
import numpy as np
import keras
import wandb
from wandb.keras import WandbCallback
import random
import glob
import subprocess
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
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
import matplotlib.pyplot as plt

### setup ###
data_format = 'channels_last'
keras.backend.set_image_data_format('channels_last')

print("Keras : ", keras.__version__)
print("\t data_format : ", keras.backend.image_data_format())
print("Tensorflow : ", tf.__version__)


B = 16 # number of residual block

batch_size = 8
target_size = (256,256)
downscale_factor = 1

num_steps = 600000
steps_per_epoch = 625
epochs = int(num_steps//steps_per_epoch)

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
    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(inputs)
    x = BatchNormalization(axis=axis)(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=shared_axis)(x)
    x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(x)
    x = BatchNormalization(axis=axis)(x)
    return add([x, inputs])

# build an upscale block
# PixelShuffler is replaced by an UpSampling2D layer (nearest upsampling)

def up_block(x):
    x = Conv2D(256, kernel_size=(3,3), strides=(1,1) , padding='same', activation=None, use_bias=False)(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=shared_axis)(x)
    return x

if not os.path.exists("data"):
    print("Downloading flower dataset...")
    subprocess.check_output(
        "mkdir data && curl https://storage.googleapis.com/wandb/flower-enhance.tar.gz | tar xz -C data", shell=True)
    
input_generator = Input(shape=(32, 32, 3) if data_format=='channels_last' else (3, None, None), 
                        name='input_generator')

x = Conv2D(filters=64, kernel_size=(9,9),
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

x = Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation=None, use_bias=False)(x)
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


generator.summary()

def preproces_vgg(x):
    # scale from [-1,1] to [0, 255]
    x += 1.
    x *= 127.5
    
    # RGB -> BGR
    if data_format == 'channels_last':
        x = x[..., ::-1]
    else:
        x = x[:,::-1,:,:]
    
    # apply Imagenet preprocessing : BGR mean
    mean = [103.939, 116.778, 123.68]
    _IMAGENET_MEAN = K.constant(-np.array(mean))
    x = K.bias_add(x, K.cast(_IMAGENET_MEAN, K.dtype(x)))
    
    return x

def vgg_loss(y_true, y_pred):
    # load pretrained VGG
    vgg19 = VGG19(include_top=False,
                  input_shape=target_size+(3,) if data_format=='channels_last' else (3,)+target_size, 
                  weights='imagenet')
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    
    # create a model that ouputs the features from level 'block2_conv2'
    features_extractor = Model(inputs=vgg19.input, outputs=vgg19.get_layer("block2_conv2").output)
    
    # compute the features, y_pred and y_true are tensors scaled in [-1, 1]
    # it needs te be rescaled and shifted to respect VGG preprocessing strategy
    features_pred = features_extractor(preproces_vgg(y_pred))
    features_true = features_extractor(preproces_vgg(y_true))
    
    # adding the scaling factor (to have similar values as with MSE within image space)
    return 0.006*K.mean(K.square(features_pred - features_true), axis=-1)

val_dir = 'data/test'
train_dir = 'data/train'

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

val_generator = image_generator(config.batch_size, train_dir)
in_sample_images, out_sample_images = next(val_generator)
class ImageLogger(Callback):
    def on_epoch_end(self, epoch, logs):
        preds = self.model.predict(in_sample_images)
        in_resized = []
        for arr in in_sample_images:
            # Simple upsampling
            in_resized.append(arr.repeat(8, axis=0).repeat(8, axis=1))
        wandb.log({
            "examples": [wandb.Image(np.concatenate([in_resized[i] * 255, o * 255, out_sample_images[i] * 255], axis=1)) for i, o in enumerate(preds)]
        }, commit=False)
    
opt = Adam(lr=0.0001, beta_1=0.9)
generator.compile(loss='mse', optimizer=opt, metrics=[perceptual_distance])

generator.fit_generator(image_generator(config.batch_size, train_dir),
                            steps_per_epoch=config.steps_per_epoch,
                            epochs=config.num_epochs, callbacks=[
                            ImageLogger(), WandbCallback()],
                            validation_steps=config.val_steps_per_epoch,
                            validation_data=val_generator)