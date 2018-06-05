from __future__ import print_function

import numpy as np

import keras
from keras.layers import Dense, Dropout
from keras.layers import LSTM, TimeDistributed, Bidirectional
from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential

from sklearn.utils import class_weight as clw
from sklearn.utils import shuffle

import time
import os

class emoLSTM():
    
    def __init__(self):
        self.model = Sequential()
        self.model.add(TimeDistributed(Dense(512, activation='relu'), input_shape=(None, 40)))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(6, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'],
                            weighted_metrics=['accuracy'])
        
    def fit(self, x_train, y_train,
            batch_size=1, 
            epochs=25,
            validation_data=None, class_weight=None):
        
        steps_per_epoch = len(x_train)/batch_size
        x_val, y_val, val_sample_weight = validation_data
        validation_steps = len(x_val)/batch_size
        # number_of_batches = len(x_train)
        
        self.model.fit_generator(data_generator(x_train, y_train), 
                       steps_per_epoch=steps_per_epoch,
                       epochs=epochs, 
                       validation_data=val_generator(x_val, y_val, val_sample_weight),
                       validation_steps=validation_steps,
                       class_weight=class_weight)

        # score = self.model.evaluate_generator(val_generator(x_val, y_val, val_sample_weight), steps=validation_steps)
        # print ("Test Loss: {}, Test UA: {}, Test WA: {}".format(score[0], score[1], score[2]))
        
        model_name = 'emorec_model_'+time.strftime("%m%d_%H%M%S")
        save_dir = 'models'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name+'.h5')
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)
        

    def predict(self, x_test):
        return self.model.predict(x_test)


def data_generator(x_train, y_train):
    while True:
        x_train, y_train = shuffle(x_train, y_train)
        for x,y in zip(x_train, y_train):
            yield x[None,:], y[None,:]


def val_generator(x_val, y_val, val_sample_weight):
    while True:
        x_val, y_val, val_sample_weight = shuffle(x_val, y_val, val_sample_weight)
        for x,y,w in zip(x_val, y_val, val_sample_weight):
            # print (x[None,:].shape, y[None,:].shape, w.shape)
            yield x[None,:], y[None,:], w


if __name__ == "__main__":
    enn = emoLSTM()
    enn.model.summary()
    x_train = np.load('dataset/x_train.npy')
    y_train = np.load('dataset/y_train.npy')
    x_val = np.load('dataset/x_val.npy')
    y_val = np.load('dataset/y_val.npy')

    # compute class weights due to imbalanced data. 
    class_weight = clw.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weight = dict(enumerate(class_weight))
    val_sample_weight = np.array([class_weight[cls] for cls in y_val])
    val_sample_weight = val_sample_weight.reshape(-1,1)

    # convert training labels to one hot vectors.
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)

    enn.fit(x_train, y_train, 
            batch_size=1, 
            epochs=2,
            validation_data=(x_val, y_val, val_sample_weight), class_weight=class_weight)
