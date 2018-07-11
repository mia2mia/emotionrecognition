from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import keras
from keras.layers import Dense, Dropout, Masking
from keras.layers import LSTM, TimeDistributed, Bidirectional
# from keras.layers import GlobalAveragePooling1D
from keras.models import Sequential, load_model
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

from sklearn.utils import class_weight as clw
from sklearn.utils import shuffle

import time
import os
from math import ceil, floor, pow


class MeanPool(Layer):
    def __init__(self, **kwargs):
        super(MeanPool, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)


    def compute_mask(self, input, input_mask=None):
      # do not pass the mask to the next layers
      return None


    def call(self, x, mask=None):
        # x shape = (batch, timesteps, feats)
        # mask shape = (batch, timesteps)
        if mask is not None:
            # mask (batch, time)
            mask = K.cast(mask, K.floatx()) # cast to float type
            # mask (batch, x_dim, time)
            mask = K.repeat(mask, x.shape[-1]) # (batch, timesteps) --> (batch, feats, timesteps)
            # mask (batch, time, x_dim)
            mask = tf.transpose(mask, [0,2,1]) # (batch, feats, timesteps) --> (batch, timesteps, feats)
            x = x * mask
        return K.sum(x, axis=1) / K.sum(mask, axis=1)


    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return (input_shape[0], input_shape[2])

class emoLSTM():
    
    def __init__(self, pre_trained=None):
        if pre_trained is None:
            self.build_model()
        else:
            self.model = load_model(pre_trained)
        

    def build_model(self):
        self.model = Sequential()
        # Masking layer to ignore the 0 padded values
        self.model.add(Masking(mask_value=0.0, input_shape=(None, 40)))

        # Dense layers for LLD learning
        self.model.add(TimeDistributed(Dense(128, activation='relu')))#, input_shape=(None, 40)))
        self.model.add(Dropout(0.5))
        # self.model.add(TimeDistributed(Dense(512, activation='relu')))
        # self.model.add(Dropout(0.5))

        # Bidirectional LSTM Layer
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(Dropout(0.5))
        # Average Pooling Layer for combining all time steps' outputs
        # self.model.add(GlobalAveragePooling1D())
        self.model.add(MeanPool())
        
        # Final Classification Layer with Softmax activation
        self.model.add(Dense(4, activation='softmax'))

        # Loss Function and Metrics
        opt = optimizers.RMSprop(lr=0.0)
        self.model.compile(loss='categorical_crossentropy', 
                            optimizer=opt, 
                            metrics=['accuracy'],
                            weighted_metrics=['accuracy'])

        # Callbacks: Model snapshotting, early stopping, Learning Rate scheduler
        # self.loss_history = LossHistory()

        if not os.path.isdir('models'):
            os.makedirs('models')
        file_path = 'models/emorec_model_' \
                        + time.strftime("%m%d_%H%M%S") \
                        + ".val-acc-{val_weighted_acc:.4f}" \
                        + '.h5'

        self.callback_list = [
            LearningRateScheduler(step_decay, verbose=1),
            EarlyStopping(
                monitor='val_weighted_acc',
                patience=10,
                verbose=1,
                mode='max'
            ),
            ModelCheckpoint(
                filepath=file_path,
                monitor='val_weighted_acc',
                save_best_only='True',
                verbose=1,
                mode='max'
            )
        ]


    def fit(self, x_train, y_train,
            batch_size=32, 
            epochs=25,
            validation_data=None, class_weight=None):
        
        x_val, y_val, val_sample_weight = validation_data
        steps_per_epoch = ceil(len(x_train) / batch_size)
        # validation_steps = len(x_val) #// batch_size
        # number_of_batches = len(x_train)
        history = self.model.fit_generator(
                        generator=intel_bucket_generator(x_train, y_train, batch_size=batch_size), 
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs, 
                        class_weight=class_weight,
                        validation_data=(pad_sequences(x_val), y_val, val_sample_weight),
                        callbacks=self.callback_list) # validation_steps=validation_steps,

        self.history = history
        # print (self.loss_history.losses)  
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
        plt.plot(history['weighted_acc'], '-o', label='train')
        plt.plot(history['val_weighted_acc'], '-o', label='val')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.savefig('metrics.png')
        plt.gcf().set_size_inches(15, 12)
        if show:
            plt.show()
        else:
            plt.close()

    
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * pow(drop, floor((epoch)/epochs_drop))
    return lrate


def intel_bucket_generator(x_train, y_train, batch_size=64):
    num_train = len(x_train)
    seq_lens = [example.shape[0] for example in x_train]
    sort_indices = np.argsort(seq_lens)
    x_train = x_train[sort_indices]
    y_train = y_train[sort_indices]
    #check if it worked
    # print (x_train[0].shape[0], x_train[-1].shape[0])

    while True:
        counter = 0
        while counter < num_train:
            x_batch = pad_sequences(x_train[counter:counter+batch_size])
            y_batch = y_train[counter:counter+batch_size]
            x_batch, y_batch = shuffle(x_batch, y_batch)
            # print (x_batch.shape[1])
            counter = counter + batch_size
            yield x_batch, y_batch


def pad_sequences(mini_batch):
    batch = np.copy(mini_batch)
    max_len = max([example.shape[0] for example in batch])
    feat_len = batch[0].shape[1]
    for i,example in enumerate(batch):
        seq_len = example.shape[0]
        if seq_len != max_len:
            batch[i] = np.vstack([example, np.zeros((max_len - seq_len, feat_len), dtype=example.dtype)])
    return np.dstack(batch).transpose(2,0,1)


if __name__ == "__main__":
    enn = emoLSTM()
    enn.model.summary()
    x_train = np.load('dataset_four/x_train.npy')
    y_train = np.load('dataset_four/y_train.npy')
    x_val = np.load('dataset_four/x_val.npy')
    y_val = np.load('dataset_four/y_val.npy')

    # compute class weights due to imbalanced data. 
    class_weight = clw.compute_class_weight('balanced', np.unique(y_train), y_train)
    class_weight = dict(enumerate(class_weight))
    val_class_weight = clw.compute_class_weight('balanced', np.unique(y_val), y_val)
    val_class_weight = dict(enumerate(val_class_weight))
    val_sample_weight = np.array([val_class_weight[cls] for cls in y_val])
    # val_sample_weight = val_sample_weight.reshape(-1,1)
    # convert training labels to one hot vectors.
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)

    enn.fit(x_train, y_train, 
            batch_size=64, 
            epochs=100,
            validation_data=(x_val, y_val, val_sample_weight), 
            class_weight=class_weight)




# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#         self.lr = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.lr.append(step_decay(len(self.losses)))
#         print('lr:', step_decay(len(self.losses)))

# def bucket_generator(x_train, y_train, batch_size=32):
#     num_train = len(x_train)
#     while True:
#         counter = 0
#         x_train, y_train = shuffle(x_train, y_train)
#         while counter < num_train:
#             x_batch = pad_sequences(x_train[counter:counter+batch_size])
#             y_batch = y_train[counter:counter+batch_size]
#             # print (x_batch.shape, y_batch.shape)
#             counter = counter + batch_size
#             yield x_batch, y_batch
#             
"""deprecated batch size 1 data generator
# def data_generator(x_train, y_train):
#     while True:
#         x_train, y_train = shuffle(x_train, y_train)
#         for x,y in zip(x_train, y_train):
#             yield x[None,:], y[None,:]


# def val_generator(x_val, y_val, val_sample_weight):
#     while True:
#         x_val, y_val, val_sample_weight = shuffle(x_val, y_val, val_sample_weight)
#         for x,y,w in zip(x_val, y_val, val_sample_weight):
#             # print (x[None,:].shape, y[None,:].shape, w.shape)
#             yield x[None,:], y[None,:], w
"""