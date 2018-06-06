from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

import keras
from keras.layers import Dense, Dropout, Masking
from keras.layers import LSTM, TimeDistributed, Bidirectional
# from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer

from sklearn.utils import class_weight as clw
from sklearn.utils import shuffle

import time
import os


class MeanPool(Layer):
    def __init__(self, **kwargs):
        super(MeanPool, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def compute_mask(self, input, input_mask=None):
      # do not pass the mask to the next layers
      return None

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx())
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1])
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, [0,2,1])
            x = x * mask
        return K.sum(x, axis=1) / K.sum(mask, axis=1)

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return (input_shape[0], input_shape[2])


class emoLSTM():
    
    def __init__(self):
        self.model = Sequential()
        self.model.add(Masking(mask_value=0.0, input_shape=(None, 40)))
        self.model.add(TimeDistributed(Dense(512, activation='relu')))#, input_shape=(None, 40)))
        self.model.add(Dropout(0.5))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
        # self.model.add(GlobalAveragePooling1D())
        self.model.add(MeanPool())
        # self.model.add(Average())
        self.model.add(Dense(6, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'],
                            weighted_metrics=['accuracy'])
        
    def fit(self, x_train, y_train,
            batch_size=32, 
            epochs=25,
            validation_data=None, class_weight=None):
        
        x_val, y_val, val_sample_weight = validation_data
        steps_per_epoch = len(x_train) // batch_size
        # validation_steps = len(x_val) #// batch_size
        # number_of_batches = len(x_train)
        self.model.fit_generator(
                        bucket_generator(x_train, y_train, batch_size=batch_size), 
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, 
                        class_weight=class_weight,
                        validation_data=(pad_sequences(x_val), y_val, val_sample_weight)
                        )
                       #  validation_steps=validation_steps,
                       # )

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


# def data_generator(x_train, y_train):
#     while True:
#         x_train, y_train = shuffle(x_train, y_train)
#         for x,y in zip(x_train, y_train):
#             yield x[None,:], y[None,:]

def bucket_generator(x_train, y_train, batch_size=32):
    num_train = len(x_train)
    while True:
        counter = 0
        x_train, y_train = shuffle(x_train, y_train)
        while (counter+batch_size) < num_train:
            x_batch = pad_sequences(x_train[counter:counter+batch_size])
            y_batch = y_train[counter:counter+batch_size]
            # print (x_batch.shape, y_batch.shape)
            counter = counter + batch_size
            yield x_batch, y_batch

def pad_sequences(mini_batch):
    batch = np.copy(mini_batch)
    max_len = max([example.shape[0] for example in batch])

    for i,example in enumerate(batch):
        seq_len = example.shape[0]
        if seq_len != max_len:
            batch[i] = np.vstack([example, np.zeros((max_len - seq_len, 40), dtype=example.dtype)])
    return np.dstack(batch).transpose(2,0,1)

# def val_generator(x_val, y_val, val_sample_weight):
#     while True:
#         x_val, y_val, val_sample_weight = shuffle(x_val, y_val, val_sample_weight)
#         for x,y,w in zip(x_val, y_val, val_sample_weight):
#             # print (x[None,:].shape, y[None,:].shape, w.shape)
#             yield x[None,:], y[None,:], w


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
    # val_sample_weight = val_sample_weight.reshape(-1,1)

    # convert training labels to one hot vectors.
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)

    enn.fit(x_train[:1000], y_train[:1000], 
            batch_size=32, 
            epochs=2,
            validation_data=(x_val[:10], y_val[:10], val_sample_weight[:10]), 
            class_weight=class_weight)
