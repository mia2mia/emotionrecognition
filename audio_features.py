"""Code to extract audio features
References:

[1] Haytham Fayek's blog post on "Speech Processing for Machine Learning"
    url: http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html

[2] Theodoros Giannakopoulos' "pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis"
    url: https://github.com/tyiannak/pyAudioAnalysis
"""

import numpy as np # matrix math
from scipy.io import wavfile # reading the wavfile
from scipy.fftpack import dct
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction

def extract_logmel(path_file, 
                    frame_size=25e-3, 
                    frame_stride=10e-3, 
                    NFFT=512,
                    nfilt=40,
                    normalize=True):
    """Code to extract logmel features
        Inputs: path_file: The path to the audio file
                frame_size: frame size to use in milliseconds. default=25ms
                frame_stride: frames stride to use in milliseconds. default=10ms
                NFFT: n-point FFT. default=512
                nfilt: number of mel-filter banks to apply. default=40
                normalize: Whether to return normalized or unnormalized coefficients. default=True
        Outputs: filter_bank: nfilt Melfilter bank coefficients
    """
    sample_rate, signal = wavfile.read(path_file)
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples without
    # truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z)
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) \
                + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # hamming window
    frames *= np.hamming(frame_length)
    
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    if normalize:
        filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    
    return filter_banks


def extract_mfcc(path_file, 
                frame_size=25e-3, 
                frame_stride=10e-3, 
                NFFT=512,
                nfilt=40,
                num_ceps=12,
                normalize=True):
    """Code to extract MFCC that internally extracts logmel features
    and then applies DCT to it
    Additional inputs: num_ceps: Number of cepstral coefficients to return. Default=12 
                Outputs: mfcc: Normalized Mel frequency cepstral coefficients
    """
    filter_banks = extract_logmel(path_file, 
                                    frame_size=25e-3, 
                                    frame_stride=10e-3, 
                                    NFFT=512,
                                    nfilt=40,
                                    normalize=False)

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    
    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    if normalize:
        mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc


def extract_features(path_file, 
                    frame_size=25e-3, 
                    frame_stride=10e-3):
    """Function to combine logmel and frame level ST features"""
    [sample_rate, signal] = audioBasicIO.readAudioFile(path_file)
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    # signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    st_features = audioFeatureExtraction.stFeatureExtraction(signal, 
                                                            sample_rate, 
                                                            frame_length, 
                                                            frame_step)
    filter_banks = extract_logmel(path_file, 
                                frame_size=25e-3, 
                                frame_stride=10e-3,
                                normalize=False)

    st_features = np.transpose(st_features) # transpose to make frame_count as x-axis
    st_features = np.delete(st_features, np.s_[8:21], axis=1) # delete the MFCCs
    if st_features.shape[0] - filter_banks.shape[0] == 1:
        st_features = st_features[:-1, :]
    # print (st_features.shape[0], filter_banks.shape[0])
    features = np.c_[st_features, filter_banks]
    features -= (np.mean(features, axis=0) + 1e-8)
    return features
