# speech-emotion-recognition
This repo contains the implementation of the paper [Automatic Speech Emotion Recognition using RNN with Local Attention](https://www.researchgate.net/profile/Seyedmahdad_Mirsamadi/publication/314756323_Automatic_Speech_Emotion_Recognition_Using_Recurrent_Neural_Networks_with_Local_Attention/links/59d9e6ddaca272e6096bc213/Automatic-Speech-Emotion-Recognition-Using-Recurrent-Neural-Networks-with-Local-Attention.pdf)


# Dependencies

This code is contingent upon the following dependencies:
* Python 2 (although most of the code is written to be both version 2/3 compliant)
* tensorflow >= 1.8.0
* Keras >= 2.1.6
* sklearn
* matplotlib
* numpy
* libmagic
* scipy
* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) (only needed if using short term features - only supported in Python 2) Description: https://github.com/tyiannak/pyAudioAnalysis/wiki/3.-Feature-Extraction

For realtime audio recording functionality:
* [`SpeechRecognition`](https://pypi.org/project/SpeechRecognition/) - Install using pip
* [`pyAudio`](https://people.csail.mit.edu/hubert/pyaudio/) - See installation instructions for your platform


The processed IEMOCAP data needs to be in a folder called IEMOCAP_PROCESSED. Download the dataset from: https://docs.lighting.com/:u:/g/personal/jin_yu_1_signify_com/EUWao7PytdpGp98D_XtQDSQBBABRb_iXJDtkKklmCTbLuQ?e=xUv4Fy


### Usage
This section provides detailed information about the code structure and how to train and test your models from scratch.

#### Code Structure

* DATASET_FOUR_4 - Logmel Feature extracted dataset from IEMOCAP for 4 emotions
* DATASET_ALT_LOGMEL_4 - logmel feature extracted using google audioset's logmel code for 4 emotions
* DATASET_LOGMEL_6 - logmel extracted feature set for 6 emotions
* DATASET_LOGMEL+ST_6 - logmel + short term features extracted using pyAudioAnalysis library for 6 emotions
* DATASET_VGGISH_4 - VGGish embeddings extracted for 4 emotions
* DATASET_SPECGRAM - spectrogram dataset extracted from 4 emotions
* MODELS - directory containing trained emotion models
* PYAUDIOANALYSIS - audio feature extraction utilities from pyaudioanalysis library
* `audio_features.py` - code containing audio feature extraction utilities such as logmel, MFCC, short time features, etc.
* `model_lstm.py` - main code used to train emotion recognition model. Has the architecture described in the paper mention above, except the attention part. Its a time distributed dense layer, followed by Bidirectional LSTM, followed by Mean Pooling and dense layer with softmax activation for emotional class.
* `load_model.py` - code used to test the model on realtime audio or recorded audio samples.
* `mel_features.py` - alternate logmel extraction code from google's audioset.
* `model_conv_specgram.py` - deprecated. code used to test emotion recognition model using spectrogram features.
* `model_vggish.py` - deprecated. code used to test emotion recognition model using extracted VGGish embeddings.
* `process_iemocap.py` - code used for parsing IEMOCAP annotations, loading audio, extracting features and saving the dataset.
* `test_model.ipynb` - jupyter notebook for interactive testing of realtime application

#### How to test the pre-trained model

Open `load_model.py` and change the feature extraction function to the correct type of features you want to extract under the `#TODO` line as one of the follows:
```python
from audio_features import extract_logmel as extract_feat # 40 dim
from audio_features import extract_features as extract_feat # 62 dim
from audio_features import extract_mfcc as extract_feat # 12 dim
```
Now, we can run the code by specifying the path to the trained model, whether we want to use realtime mode or not, and optional path to a wav file if we want to test it offline:
```sh
$ python load_model.py -w <MODEL_PATH> -r <1/0 or y/n> [-a <WAV_PATH>]
```

#### How to train from scratch

1. The first step is to do feature extraction from the processed IEMOCAP dataset. To do this, first make sure that the [IEMOCAP_PROCESSED](https://docs.lighting.com/:u:/g/personal/jin_yu_1_signify_com/EUWao7PytdpGp98D_XtQDSQBBABRb_iXJDtkKklmCTbLuQ?e=xUv4Fy) directory is in the root directory of your repo. 

2. Open `process_iemocap.py` and import the correct feature extraction function as before, such as for logmel:
```python
from audio_features import extract_logmel as extract_feat # 40 dim
```

3. Run the data preprocessing code as follows:
```sh
$ python process_iemocap.py -p <path to IEMOCAP_PROCESSED> -o <dataset-output-folder>
```
The dataset will then be created in the specified output folder.

4. Now we can start training using the `model.py` file. Change training parameters if necessary inside the code in the `build_model`, `define_callbacks` or `fit` functions inside the `emoLSTM` class. Then Run:
```sh
$ python model.py -p <DATASET_PATH>
```
The best model will be saved under MODELS directory. All models trained so far are in the folder `~/speech-emotion-recognition/models` on AWS instance `deep`.






