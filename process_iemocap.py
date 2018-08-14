import argparse
import os
from audio_features import extract_logmel
from sklearn.utils import shuffle
import numpy as np


def parse_args():
    """Returns dictionary containing CLI arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--data-path", required=True, help="Path to the IEMOCAP processed dataset")
    ap.add_argument("-o", "--out-path", required=True, help="Path to the output directory")
    args = vars(ap.parse_args())
    return args


def parse_dialog_file(dialog_file_path, x, y):
    """Function to parse the Evaluation file per dialog
        Evaluation file structure:

        % [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]

        [6.2901 - 8.2357]   Ses01F_impro01_F000 neu [2.5000, 2.5000, 2.5000]
        C-E2:   Neutral;    ()
        C-E3:   Neutral;    ()
        C-E4:   Neutral;    ()
        C-F1:   Neutral;    (curious)
        A-E3:   val 3; act 2; dom  2;   ()
        A-E4:   val 2; act 3; dom  3;   (mildly aggravated but staying polite, attitude)
        A-F1:   val 3; act 2; dom  1;   ()

        [10.0100 - 11.3925] Ses01F_impro01_F001 neu [2.5000, 2.5000, 2.5000]
        ...

    """
    global target_emotions
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

                # extract features and append it to the list
                features = extract_logmel(wav_path) # shape (M, 40)
                if features is not None:
                    x.append(features)
                    y.append(target_emotions[emotion])
            # skip all lines till the blank line
            while line != '\n':
                line = dialog_file.readline()
            # read in the next line containing frame and emotion
            line = dialog_file.readline().strip()


def walk_through_sessions(x, y, data_path, out_path, sessions):
    """Function to iterate recursively over all session annotations,
        extract features, and append the features/labels to the x,y lists.
    """
    for session in sessions:
        session_path = os.path.join(data_path, session)
        for dirpath, dnames, fnames in os.walk(session_path):
            for f in fnames:
                if f.endswith(".txt"):
                    dialog_file_path = os.path.join(dirpath, f)
                    parse_dialog_file(dialog_file_path, x, y)


def compute_mean_std(x_train, y_train):
    """Function to compute mean and std of feature vectors from neutral emotions"""
    # warning: not used
    global mean, std, target_emotions
    neutral_examples = np.vstack(x_train[np.where(y_train==target_emotions['neu'])])
    mean = np.mean(neutral_examples, axis=0)
    std = np.std(neutral_examples, axis=0)
    print (mean.shape)
    print (mean)
    print (std.shape)
    print (std)


def normalize_data(x, mean, std):
    """function to normalize X by subtracting mean and dividing by std"""
    for i in range(len(x)):
        x[i] = (x[i] - mean) / std


def create_training_set(data_path, out_path, sessions):
    """wrapper function to create training set"""
    global mean, std
    print ("Processing training set...")
    x_train = []
    y_train = []
    walk_through_sessions(x_train, y_train, data_path, out_path, sessions)
    print (len(x_train), len(y_train))
    """Lines to perform external normalization. Not used.
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    compute_mean_std(x_train, y_train)
    print ("sample before normalization", x_train[0][0])
    normalize_data(x_train, mean, std)
    print ("sample after normalization", x_train[0][0])
    """
    np.save(os.path.join(out_path,'x_train.npy'), x_train)
    np.save(os.path.join(out_path,'y_train.npy'), y_train)


def create_testval_set(data_path, out_path, sessions):
    global mean, std
    print ("Processing test/val set...")
    x = []
    y = []
    walk_through_sessions(x, y, data_path, out_path, sessions)

    # normalize_data(x, mean, std)
    n_val = int(len(y)/2)
    x,y = shuffle(x,y)
    x_val = x[:n_val]
    y_val = y[:n_val]
    x_test = x[n_val:]
    y_test = y[n_val:]

    print (len(x_val), len(y_val), len(x_test), len(y_test))
    np.save(os.path.join(out_path,'x_val.npy'), x_val)
    np.save(os.path.join(out_path,'y_val.npy'), y_val)
    np.save(os.path.join(out_path,'x_test.npy'), x_test)
    np.save(os.path.join(out_path,'y_test.npy'), y_test)


if __name__ == '__main__':
    args = parse_args()
    data_path = args["data_path"]
    out_path = args["out_path"]

    if not os.path.exists(data_path):
        raise("Path to Dataset is incorrect or doesn't exist")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # sessions used for training and testing
    sessions_train = ['Session1', 'Session2', 'Session3', 'Session4']
    sessions_val = ['Session5']
    target_emotions = { 'ang':0, 'hap':1, 'sad':2, 'neu':3, }
    # target_emotions = {'ang':0, 'hap':1, 'exc':2, 'sad':3, 'fru':4, 'neu':5,}

    mean = None
    std = None

    create_training_set(data_path, out_path, sessions_train)
    create_testval_set(data_path, out_path, sessions_val)
