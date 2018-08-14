from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import argparse
import speech_recognition as sr
# from keras.layers import GlobalAveragePooling1D
from keras.models import load_model
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
from audio_features import extract_features

from sklearn.utils import class_weight as clw
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score

import time
import os
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
    
    def __init__(self, pre_trained):
        self.model = load_model(pre_trained, custom_objects={'MeanPool': MeanPool})


    def evaluate(self, x_test, y_test, sample_weight=None):
        score = self.model.evaluate(x=pad_sequences(x_test), y=y_test, sample_weight=sample_weight)
        # print ("Test Loss: {}, Test UA: {}, Test WA: {}".format(score[0], score[1], score[2]))
        print (score)
        print (self.model.metrics_names)
        

    def predict(self, x_test):
        return self.model.predict(pad_sequences(x_test))


class AudioRec(object):

    def __init__(self):
        self.r = sr.Recognizer()
        self.src = sr.Microphone()
        with self.src as source:
            print("Calibrating microphone...")
            self.r.adjust_for_ambient_noise(source, duration=2)


    def listen(self, save_path):
        with self.src as source:
            print("Recording ...")
            # record for a maximum of 10s
            audio = self.r.listen(source, phrase_time_limit=10)
        # write audio to a WAV file
        with open(save_path, "wb") as f:
            f.write(audio.get_wav_data())



def pad_sequences(mini_batch):
    batch = np.copy(mini_batch)
    max_len = max([example.shape[0] for example in batch])
    feat_len = batch[0].shape[1]
    for i,example in enumerate(batch):
        seq_len = example.shape[0]
        if seq_len != max_len:
            batch[i] = np.vstack([example, np.zeros((max_len - seq_len, feat_len), dtype=example.dtype)])
    return np.dstack(batch).transpose(2,0,1)


def parse_args():
    """Returns dictionary containing CLI arguments"""
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weights", required=True, help="Path to the saved model")
    ap.add_argument("-r", "--realtime", type=str2bool, required=False, default=True, help="Whether to run Realtime")
    ap.add_argument("-a", "--audio", required=False, default=None, help="path to wav file")
    args = vars(ap.parse_args())

    if args["realtime"] and args["audio"] is not None:
        ap.error("wav file can only be specified when not running in realtime mode")
    elif not args["realtime"] and args["audio"] is None:
        ap.error("Must specify wav file when not running in realtime mode")

    return args


if __name__ == "__main__":
    args = parse_args()
    weights_path = args["weights"]
    realtime_mode = args["realtime"]
    wav_path = args["audio"]

    labels = {0:'ang', 1:'hap', 2:'exc', 3:'sad', 4:'fru', 5:'neu'}
    enn = emoLSTM(weights_path)
    enn.model.summary()

    if realtime_mode:
        ar = AudioRec()
        while True:
            ar.listen("microphone-results.wav")
            features = extract_features("microphone-results.wav")

            if features is not None:
                print ("Extracted features of shape: ", features.shape)
                y_pred = enn.predict(features[None,:])
                pred_class = labels[np.argmax(y_pred, axis=1)[0]]
                print ("Predicted emotion is: ", pred_class)
            time.sleep(0.1)

    else:
        wav_path = args["audio"]
        features = extract_features(wav_path)
        print ("Extracted features of shape: ", features.shape)
        y_pred = enn.predict(features[None,:])
        pred_class = labels[np.argmax(y_pred, axis=1)[0]]
        print("Predicted emotion is: ", pred_class)


    # x_train = np.load('data/x_train.npy')
    # y_train = np.load('data/y_train.npy')
    # x_val = np.load('data/x_val.npy')
    # y_val = np.load('data/y_val.npy')

    # compute class weights due to imbalanced data. 
    # class_weight = clw.compute_class_weight('balanced', np.unique(y_train), y_train)
    # class_weight = dict(enumerate(class_weight))
    # val_class_weight = clw.compute_class_weight('balanced', np.unique(y_val), y_val)
    # val_class_weight = dict(enumerate(val_class_weight))
    # val_sample_weight = np.array([val_class_weight[cls] for cls in y_val])
    # 
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
    # y_pred = np.argmax(enn.predict(x_val), axis=1)
    # print (y_pred)
    # print (y_pred.shape)
    # confusion_mat = confusion_matrix(y_val, y_pred, sample_weight=val_sample_weight)
    # print (confusion_mat)
    # acc = accuracy_score(y_val, y_pred, sample_weight=val_sample_weight)
    # print ("Accuracy = {}".format(acc))
    














