from __future__ import print_function
from __future__ import division

import numpy as np
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
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.utils import class_weight as clw
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score

import time
import os
from math import ceil
# import matplotlib.pyplot as plt


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
    
    def __init__(self, pre_trained=None):
        if pre_trained is None:
            self.build_model()
        else:
            self.model = load_model(pre_trained, custom_objects={'MeanPool': MeanPool})
        

    def build_model(self):
        self.model = Sequential()
        # Masking layer to ignore the 0 padded values
        self.model.add(Masking(mask_value=0.0, input_shape=(None, 61)))

        # Dense layers for LLD learning
        self.model.add(TimeDistributed(Dense(512, activation='relu')))#, input_shape=(None, 40)))
        self.model.add(Dropout(0.5))
        self.model.add(TimeDistributed(Dense(512, activation='relu')))
        self.model.add(Dropout(0.5))

        # Bidirectional LSTM Layer
        self.model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.5)))

        # Average Pooling Layer for combining all time steps' outputs
        # self.model.add(GlobalAveragePooling1D())
        self.model.add(MeanPool())
        # self.model.add(Average())
        
        # Final Classification Layer with Softmax activation
        self.model.add(Dense(6, activation='softmax'))

        # Loss Function and Metrics
        adam = optimizers.RMSprop(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', 
                            optimizer=adam, 
                            metrics=['accuracy'],
                            weighted_metrics=['accuracy'])

        # Model snapshotting and early stopping
        file_path = 'models/emorec_model_' \
                        + time.strftime("%m%d_%H%M%S") \
                        + ".{epoch:02d}-{val_weighted_acc:.4f}" \
                        + '.h5'

        self.callback_list = [
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
                        callbacks=self.callback_list
                        )
                       #  validation_steps=validation_steps,
                       # )
        self.plot_metrics(history.history)
        
    
    def evaluate(self, x_test, y_test, sample_weight=None):
        score = self.model.evaluate(x=pad_sequences(x_test), y=y_test, sample_weight=sample_weight)
        # print ("Test Loss: {}, Test UA: {}, Test WA: {}".format(score[0], score[1], score[2]))
        print (score)
        print (self.model.metrics_names)
        

    def predict(self, x_test):
        return self.model.predict(pad_sequences(x_test))

"""function to plot
    def plot_metrics(self, history):
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
        plt.show()
"""

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
    labels = {0:'ang', 1:'hap', 2:'exc', 3:'sad', 4:'fru', 5:'neu'}
    enn = emoLSTM('models/emorec_model_0611_040759.val-acc-0.4148.h5')
    enn.model.summary()

    # x_train = np.load('data/x_train.npy')
    # y_train = np.load('data/y_train.npy')
    x_val = np.load('data/x_val.npy')
    y_val = np.load('data/y_val.npy')

    # compute class weights due to imbalanced data. 
    # class_weight = clw.compute_class_weight('balanced', np.unique(y_train), y_train)
    # class_weight = dict(enumerate(class_weight))
    val_class_weight = clw.compute_class_weight('balanced', np.unique(y_val), y_val)
    val_class_weight = dict(enumerate(val_class_weight))
    val_sample_weight = np.array([val_class_weight[cls] for cls in y_val])
    # val_sample_weight = val_sample_weight.reshape(-1,1)

    # convert training labels to one hot vectors.
    # y_train = keras.utils.to_categorical(y_train)
    # y_val = keras.utils.to_categorical(y_val)

    # seq_lens = [example.shape[0] for example in x_val]
    # sort_indices = np.argsort(seq_lens)
    # x_val = x_val[sort_indices]
    # y_val = y_val[sort_indices]
    # val_sample_weight = val_sample_weight[sort_indices]

    # enn.evaluate(x_val, y_val, val_sample_weight)
    y_pred = np.argmax(enn.predict(x_val), axis=1)
    print (y_pred)
    print (y_pred.shape)
    confusion_mat = confusion_matrix(y_val, y_pred, sample_weight=val_sample_weight)
    print (confusion_mat)
    acc = accuracy_score(y_val, y_pred, sample_weight=val_sample_weight)
    print ("Accuracy = {}".format(acc))
    # wav_path = 'samples/Ses05F_impro03.wav'
    # logmel = extract_logmel(wav_path)
    # print (logmel.shape)
    # y_pred = enn.predict(logmel[None,:])
    # print ()
    # print("Predicted label: {}, {}".format(np.argmax(y_pred, axis=1), labels[np.argmax(y_pred, axis=1)[0]]))






















