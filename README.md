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

To enable the usage of the CPU from Keras, follow this guide: https://medium.com/@ab9.bhatia/set-up-gpu-accelerated-tensorflow-keras-on-windows-10-with-anaconda-e71bfa9506d1

## What is the started project doing?

Check this for info:
https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4

## Notes

⚠️ IMPORTANT: Remember to always create your branch and do not ever push to master, all the pieces will be merged before submission.

What is this piece of code? Worry not, this is needed in case you have a cuda enabled gpu that keras can use, unless you install all the packages needed to make it work with the gpu, this can be left commented out.

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


