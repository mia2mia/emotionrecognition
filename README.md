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
* scipy
* [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) (only needed if using short term features - only supported in Python 2)

For realtime audio recording functionality:
* [`SpeechRecognition`](https://pypi.org/project/SpeechRecognition/) - Install using pip
* [`pyAudio`](https://people.csail.mit.edu/hubert/pyaudio/) - See installation instructions for your platform


The training data needs to be in a folder called dataset. Download the dataset from: https://docs.lighting.com/:u:/g/personal/deep_chakraborty_lighting_com/ESTNoXziobxIotdX_ZF4cFEBbN9bb-aljGNm9pErYfptNA?e=0mmoSk

The models will be saved in a folder called models.

Usage for now:
```shell
python model.py
```
