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
    filename  = '/content/drive/Shared drives/Deep Learning Gang 👺/images/' + folder + '/' + name + '.jpg'
    # ...

#Generation
Data_dir=np.array(glob("/content/drive/Shared drives/Deep Learning Gang 👺/wavs_train/*"))

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
conda install keras
conda install -c conda-forge librosa

```

### Using a CUDA enabled GPU

1. Install Anaconda (https://www.anaconda.com/distribution/)
2. Download Visual Studio (Wathever recent version, don't know why this is needed)
3. Install Kuda Toolkit (Need to register an Nvidia account - https://developer.nvidia.com/cuda-toolkit)
4. Download CuDNN https://developer.nvidia.com/rdp/cudnn-archive#a-collapse714-9
    > By default, % CUDA_Installation_directory % points to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X
    
    > The cuDNN library contains three files: \bin\cudnn64_7.dll (the version number may be different), \include\cudnn.h and \lib\x64\cudnn.lib. You should copy them to the following locations:
    - %CUDA_Installation_directory%\bin\cudnn64_7.dll
    - % CUDA_Installation_directory %\include\cudnn.h
    - % CUDA_Installation_directory %\lib\x64\cudnn.lib
5. Add System Env Variables 
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\libnvvp
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin
    - C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\lib\x64
6. Restart PC
7. Install Tensorflow:
    - conda create -n venv python=3.7 anaconda
    - conda activate venv
    - conda install keras
    - conda install -c conda-forge librosa
8. We need to install tensorflow-gpu so:
    - conda uninstall tensorflow (installed by keras)
    - conda install tensorflow-gpu
    - conda reinstall keras (to validate the reinstallation of tensorflow)
   
More Info: https://medium.com/@ab9.bhatia/set-up-gpu-accelerated-tensorflow-keras-on-windows-10-with-anaconda-e71bfa9506d1

## What is the started project doing?

Check this for info:
https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4

## Notes

⚠️ IMPORTANT: The Network runs for only 1 epoch, this is to quickly check if the whole program works.

⚠️ IMPORTANT: The gitignore will also prevent you from uploading the dataset to github, if you work locally, make sure to always respect this structure:

C:.

├───images

│   ├───meta

│   ├───test

│   └───train

├───meta

├───test

└───train

⚠️ IMPORTANT: Remember to always create your branch and do not ever push to master, all the pieces will be merged before submission.

If you install your cuda libraries and set up everything but when you run your model you get this error:

```
tensorflow.python.framework.errors_impl.UnknownError:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed 
above.
         [[node conv2d_1/convolution (defined at C:\tools\Anaconda3\envs\venv\lib\site-packages\keras\backend\tensorflow_backend.py:3009) ]] [Op:__inference_keras_scratch_graph_1779]
```
Uncomment this:

```python

#WARNING: USE ONLY WHEN RUNNING WITH GPUs WITH CUDA ENABLED
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
```


