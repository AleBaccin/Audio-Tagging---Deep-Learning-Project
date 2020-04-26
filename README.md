# Audio-Tagging---Deep-Learning-Project
Deep Learning Project for automatic audio tagging.

## Use on Google Colab

As a shared Drive folder was created, the jupiter Notebook can be run as it is on Google Colab. Note that the pieces of code in which the images get generated from the .wav files should not be run(⚠️), as this will overwrite the images already present on Drive, if one want to generate custom images to try a new approach a temporary folder should be used.

```python

#Already present on the top of the notebook.
!mkdir /content/temp
!mkdir /content/temp/test
!mkdir /content/temp/train

def create_spectrogram(filename,name, folder):
    # ...
    #change this line 
    filename  = '.../images/' + folder + '/' + name + '.jpg'
    # ...

#Generation
Data_dir=np.array(glob(".../train/*"))

with tqdm(total=Data_dir.shape[0], position=0, leave=True) as pbar:
    for file in tqdm(Data_dir, position=0, leave=True):
        filename,name = file, file.split('/')[-1].split('.')[0]
        create_spectrogram(filename,name, 'train')
        pbar.update()
```

## Installation on propietary machine

If you would like to use the python notebook on your own pc you should first installa anaconda, so that you are able to create a virtual enviroment. Then you can install all the missing requirements, which should be:

```python

conda create -n venv python=3.7 anaconda
conda activate venv
conda install tensorflow-gpu
conda install -c conda-forge librosa
conda install numba==0.48.0 (This is to avoid a Warning)
```

### Using a CUDA enabled GPU

1. Install Anaconda (https://www.anaconda.com/distribution/)
2. Download Visual Studio (Wathever recent version, don't know why this is needed)
3. Install Kuda Toolkit (Need to register an Nvidia account - https://developer.nvidia.com/cuda-toolkit)
4. Download CuDNN https://developer.nvidia.com/rdp/cudnn-archive#a-collapse714-9
    > By default, % CUDA_Installation_directory % points to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0
    
    > The cuDNN library contains three files: \bin\cudnn64_7.dll (the version number may be different), \include\cudnn.h and \lib\x64\cudnn.lib. You should copy them to the following locations:
    - % CUDA_Installation_directory%\bin\cudnn64_7.dll
    - % CUDA_Installation_directory %\include\cudnn.h
    - % CUDA_Installation_directory %\lib\x64\cudnn.lib
5. Add System Env Variables 
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
6. Restart PC
7. Install Tensorflow:
    - conda create -n venv python=3.7 anaconda
    - conda activate venv
    - conda install tensorflow-gpu
    - conda install -c conda-forge librosa
    - conda install numba==0.48.0 (This is to avoid a Warning)
   
More Info: https://medium.com/@ab9.bhatia/set-up-gpu-accelerated-tensorflow-keras-on-windows-10-with-anaconda-e71bfa9506d1

## What is the started project doing?

Check this for info:
- https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4
- https://towardsdatascience.com/ensembling-convnets-using-keras-237d429157eb
- https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
- https://www.kaggle.com/CVxTz/audio-data-augmentation

## Notes

⚠️ IMPORTANT: The gitignore will also prevent you from uploading the dataset to github, if you work locally, make sure to always respect this structure:

C:.
├───images
│   ├───meta
│   ├───test
│   └───train
├───runs
├───meta
├───test
└───train

⚠️ IMPORTANT: Remember to always create your branch and do not ever push to master, all the pieces will be merged before submission.


