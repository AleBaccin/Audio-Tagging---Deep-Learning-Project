# Audio-Tagging---Deep-Learning-Project
Deep Learning Project for automatic audio tagging.

⚠️ IMPORTANT: The folder in which the notebook will run requires the following structure. Where 'meta' is the directory in which train.csv and test.csv reside, 'train' and 'test' folders store the respective .wav files and 'meta/train' and 'meta/test' store the respective .jpg images, in 'runs', the necessary file for each model's run will be stored. The 'runs', 'images/test', 'images/train' are generate by the Notebook, while meta, test and train need to befined by the user.

.

├───images

│   ├───test

│   └───train

├───runs

├───meta

├───test

└───train

## What are all these files for?

The actualy Notebook where the full project resides is: audio_tagging.ipynb, the 'optimise' Notebook contains the work we did to optimse the 1d cnn.

## Installation

If you would like to use the python notebook on your own pc it is advised to first install anaconda. Here are the requirements:

```python

conda create -n venv python=3.7 anaconda
conda activate venv
conda install tensorflow-gpu
conda install -c conda-forge librosa
conda install numba==0.48.0
conda install scikit-learn
```

Alternatively, the requirements file could be used.

```
conda install --file requirements.txt
```

Librosa is still to be installed manually as it uses che conda-forge channel.

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