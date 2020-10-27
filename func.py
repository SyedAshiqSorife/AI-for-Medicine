import keras
import h5py
import cv2


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import (
    Activation, Conv3D, Deconvolution3D, 
    MaxPooling3D, UpSampling3D, )
from keras.optimizers import Adam
from keras.layers.merge import concatenate

K.set_image_data_format("channels_first")

def create_convolution_block(input_layer, n_filters, batch_normalization=False,
                             kernel=(3, 3, 3), activation=None, padding='same', 
                             strides=(1, 1, 1), instance_normalization=False):
    
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)
    
def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), 
                       strides=(2, 2, 2), deconvolution=False):
    
    if deconvolution:
            return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                   strides=strides)
    else:
        return UpSampling3D(size=pool_size)
    

def unet_model_3d(loss_function, input_shape=(4, 160, 160, 16),
                  pool_size=(2, 2, 2), n_labels=3, initial_learning_rate=.00001,
                  deconvolution=False, depth=4, n_base_filters=32,
                  include_label_wise_dice_coefficients=False, metrics=[],
                  batch_normalization=False, activation_name="sigmoid"):
    
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer,
                                          n_filters=n_base_filters * (
                                                  2 ** layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1,
                                          n_filters=n_base_filters * (
                                                  2 ** layer_depth) * 2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth - 2, -1, -1):
        up_convolution = get_up_convolution(pool_size = pool_size,
                                            deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(
            current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1]._keras_shape[1],
            input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1]._keras_shape[1],
            input_layer=current_layer,
            batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_function,
                  metrics=metrics)
    return model

class VolumeDataGenerator(keras.utils.Sequence):
    def __init__(self, sample_list, base_dir,
                 batch_size=1, shuffle=True, dim=(160, 160, 16),
                 num_channels=4, num_classes=3, verbose=1):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.base_dir = base_dir
        self.dim = dim
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.verbose = verbose
        self.sample_list = sample_list
        self.on_epoch_end()
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sample_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return int(np.floor(len(self.sample_list) / self.batch_size))
    
    def __data__generation(self, list_IDs_temp):
        
        X = np.zeros((self.batch_size, self.num_channels, *self.dim), dtype=np.float64)
        y = np.zeros((self.batch_size, self.num_channels, *self.dim), dtype=np.float64)
        
        for i, ID in enumerate(list_IDs_temp):
            
            if self.verbose == 1:
                print("Training on: %s" % self.base_dir + ID)
            with h5py.File(self.base_dir + ID, 'r') as f:
                X[i] = np.array(f.get("x"))
                # remove the background class
                y[i] = np.moveaxis(np.array(f.get("y")), 3, 0)[1:]
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size: (index + 1) * self.batch_size]
        # Find list of IDs
        sample_list_temp = [self.sample_list[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(sample_list_temp)

        return X, y
