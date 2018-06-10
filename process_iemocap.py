import argparse
import os
from audio_features import *
from sklearn.utils import shuffle
import numpy as np


def parse_args():
    """Returns dictionary containing CLI arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--data-path", required=True, help="Path to the dataset")
    ap.add_argument("-o", "--out-path", required=True, help="Path to the output directory")
    args = vars(ap.parse_args())
    return args


def parse_dialog_file(dialog_file_path):
    global target_emotions, x, y
    """Function to parse the Evaluation file per dialog"""
    dialog_wav_base_path = os.path.splitext(dialog_file_path)[0]

    with open(dialog_file_path, 'r') as dialog_file:
        # skip the headers
        _ = dialog_file.readline()
        _ = dialog_file.readline()
        # read the line containing frame and emotion and skip till the next
        line = dialog_file.readline().strip()
        while line:
            wavfile, emotion = line.split('\t')[1:3]

            if emotion in target_emotions:
                wav_path = os.path.join(dialog_wav_base_path, wavfile+'.wav')
                print (wav_path, emotion)
                features = extract_features(wav_path)
                x.append(features)
                y.append(target_emotions[emotion])

            while line != '\n':
                line = dialog_file.readline()
            line = dialog_file.readline().strip()


if __name__ == '__main__':
    args = parse_args()
    data_path = args["data_path"]
    out_path = args["out_path"]

    if not os.path.exists(data_path):
        raise("Path to Dataset is incorrect or doesn't exist")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # sessions = ['Session1', 'Session2', 'Session3', 'Session4']
    sessions = ['Session5']
    target_emotions = {
                        'ang':0, 
                        'hap':1, 
                        'exc':2, 
                        'sad':3, 
                        'fru':4, 
                        'neu':5,
                        }

    # x_train = []
    # y_train = []
    x = []
    y = []

    for session in sessions:
        session_path = os.path.join(data_path, session)
        for dirpath, dnames, fnames in os.walk(session_path):
            for f in fnames:
                if f.endswith(".txt"):
                    dialog_file_path = os.path.join(dirpath, f)
                    parse_dialog_file(dialog_file_path)

    n_val = int(len(y)/2)
    x,y = shuffle(x,y)
    x_val = x[:n_val]
    y_val = y[:n_val]
    x_test = x[n_val:]
    y_test = y[n_val:]

    print len(x_val), len(y_val), len(x_test), len(y_test)

    np.save(os.path.join(out_path,'x_val.npy'), x_val)
    np.save(os.path.join(out_path,'y_val.npy'), y_val)
    np.save(os.path.join(out_path,'x_test.npy'), x_test)
    np.save(os.path.join(out_path,'y_test.npy'), y_test)
