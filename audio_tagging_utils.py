from glob import glob
import numpy as np
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import figure
from tqdm import tqdm
import os 

def append_ext(fn):
    return fn.replace('.wav', '.jpg')

def create_spectrogram(filename,name, folder):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = os.path.join('images', folder, f'{name}.jpg')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S
    
def create_images(input_dir, output_dir):
    Data_dir=np.array(glob(os.path.join(input_dir, '*')))
    
    with tqdm(total= Data_dir.shape[0], position=0, leave=True) as pbar:
        for file in tqdm(Data_dir, position=0, leave=True):
            filename,name = file, file.split(os.sep)[-1].split('.')[0]
            create_spectrogram(filename,name, output_dir)
            pbar.update()
            
def create_mfcc_array(df, input_dir, sr, max_len, n_mfcc):
    Data_dir=np.array(glob(os.path.join(input_dir, '*')))
    
    mfcc_vectors = []
    labels = []
    
    with tqdm(total= len(df), position=0, leave=True) as pbar:
        pbar.set_description('Creating mfcc of %d .wav files' % len(df))
        for i, row in tqdm(df.iterrows(), position=0, leave=True):
            wavfile = row['fname']
            mfcc = wav2mfcc(os.path.join(input_dir, wavfile), n_mfcc= n_mfcc, sr= sr, max_len= max_len)
            mfcc_vectors.append(mfcc)
            labels.append(row['label'])
            pbar.update()
            
    return mfcc_vectors, labels

def create_wav_array(df, input_dir, sr, max_len):
    Data_dir=np.array(glob(os.path.join(input_dir, '*')))
    
    wav_vectors = []
    labels = []
    
    with tqdm(total= len(df), position=0, leave=True) as pbar:
        pbar.set_description('Creating np.array description of %d .wav files' % len(df))
        for i, row in tqdm(df.iterrows(), position=0, leave=True):
            wavfile = row['fname']
            wave = wav(os.path.join(input_dir, wavfile), sr= sr, max_len= max_len)
            wav_vectors.append(wave)
            labels.append(row['label'])
            pbar.update()
            
    return wav_vectors, labels
            
def wav2mfcc(file_path, n_mfcc, sr, max_len):
    wave = wav(file_path, sr, max_len)
    
    mfcc = librosa.feature.mfcc(wave, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.expand_dims(mfcc, axis=-1)
    
    return mfcc

def wav(file_path, sr, max_len):
    wave, _ = librosa.load(file_path, sr=sr, res_type='kaiser_fast')
    true_length = sr * max_len
    
    if len(wave) > true_length:
        max_offset = len(wave) - true_length
        offset = np.random.randint(max_offset)
        wave = wave[offset:(true_length+offset)]
    else:
        if true_length > len(wave):
            max_offset = true_length - len(wave)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        wave = np.pad(wave, (offset, true_length - len(wave) - offset), "constant")
        
    return wave

def mfcc_input_sizes(n_mfcc, sr, max_len):
    return (n_mfcc, 1 + int(np.floor((sr*max_len)/512)), 1)

def wav_input_sizes(sr, max_len):
    return (sr*max_len, 1)
    