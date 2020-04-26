from glob import glob
import numpy as np
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
from tqdm import tqdm

def append_ext(fn):
    return fn.replace('.wav', '.jpg')

def create_spectrogram(filename,name, folder):
    plt.interactive(False)
    sample_rate = 16000
    clip, _ = librosa.load(filename, sr=16000, res_type='kaiser_fast')
    input_length = 2*sample_rate

    if len(clip) > input_length:
        max_offset = len(clip) - input_length
        offset = np.random.randint(max_offset)
        clip = clip[offset:(input_length+offset)]
    else:
        if input_length > len(clip):
            max_offset = input_length - len(clip)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        clip = np.pad(clip, (offset, input_length - len(clip) - offset), "constant")

    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = 'images\\' + folder + '\\' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def create_spectrogram_no_prepro(filename,name, folder):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = 'images\\' + folder + '\\' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S
    
# create_images('train', 'train_no_preprocessing')
def create_images_no_prepro(input_dir, output_dir):
    Data_dir=np.array(glob(f"{input_dir}\\*"))
    
    with tqdm(total= Data_dir.shape[0], position=0, leave=True) as pbar:
        for file in tqdm(Data_dir, position=0, leave=True):
            filename,name = file, file.split('\\')[-1].split('.')[0]
            create_spectrogram_no_prepro(filename,name, output_dir)
            pbar.update()
            
def create_images_prepro(input_dir, output_dir):
    Data_dir=np.array(glob(f"{input_dir}\\*"))
    
    with tqdm(total= Data_dir.shape[0], position=0, leave=True) as pbar:
        for file in tqdm(Data_dir, position=0, leave=True):
            filename,name = file, file.split('\\')[-1].split('.')[0]
            create_spectrogram(filename,name, output_dir)
            pbar.update()