# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

import keras
# from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import time
import os
import h5py

class EmoClassifier():
    
    def __init__(self, pre_trained=None):
        if pre_trained is None:
            self.build_model()
            self.define_callbacks()
        else:
            self.model = load_model(pre_trained)
        

    def build_model(self):
        self.model = Sequential()
        self.model.add(Reshape((128,128,1), input_shape=(128,128)))
        self.model.add(Conv2D(32, (3,3), activation='relu'))
        self.model.add(Conv2D(32, (3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(Conv2D(64, (3,3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        # Final Classification Layer with Softmax activation
        self.model.add(Dense(4, activation='softmax'))


        # Loss Function and Metrics
        opt = optimizers.Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', 
                            optimizer=opt, 
                            metrics=['accuracy'])


    def define_callbacks(self):
        if not os.path.isdir('weights'):
            os.makedirs('weights')
        file_path = 'weights/emo_model_conv_' \
                        + time.strftime("%m%d_%H%M%S") \
                        + '.h5'

        self.callback_list = [
            ReduceLROnPlateau(
            	monitor='val_loss', 
            	factor=0.5,
                patience=5, 
                min_lr=0.00001,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_acc',
                patience=15,
                verbose=1,
                mode='max'
            ),
            ModelCheckpoint(
                filepath=file_path,
                monitor='val_acc',
                save_best_only='True',
                verbose=1,
                mode='max'
            )
        ]


    def fit(self, x_train, y_train,
            batch_size=32, 
            epochs=100,
            validation_data=None, class_weight=None):
        
        history = self.model.fit(
        				x=x_train, y=y_train,
        				batch_size=batch_size, epochs=epochs,
        				callbacks=self.callback_list,
        				validation_data=validation_data
        			)

        self.plot_metrics(history.history)
        

    def predict(self, x_test):
        return self.model.predict(x_test)


    def plot_metrics(self, history, show=False):
        plt.subplot(2, 1, 1)
        plt.title('Loss')
        plt.plot(history['loss'], '-o', label='train')
        plt.plot(history['val_loss'], '-o', label='val')
        plt.xlabel('epoch')
        plt.legend(loc='upper right')
        plt.subplot(2,1,2)
        plt.title('Accuracy')
        plt.plot(history['acc'], '-o', label='train')
        plt.plot(history['val_acc'], '-o', label='val')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.savefig('metrics.png')
        plt.gcf().set_size_inches(15, 12)
        if show:
            plt.show()
        else:
            plt.close()


def load_data(hdf5_path):
    with h5py.File(hdf5_path, 'r') as hf:
        x_train = hf.get('x_train')
        y_train = hf.get('y_train')
        x_val = hf.get('x_val')
        y_val = hf.get('y_val')
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_val = np.array(x_val)
        y_val = np.array(y_val)

    return x_train, y_train, x_val, y_val


if __name__ == "__main__":
    emo = EmoClassifier()
    emo.model.summary()

    x_train, y_train, x_val, y_val = load_data('dataset_spegram/emo_data.h5')
    x_train = (x_train-0.5)/0.5
    x_val = (x_val-0.5)/0.5
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    emo.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val))
