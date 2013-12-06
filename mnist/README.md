MNIST DATASET
=============

This folder contains MNIST dataset files and several processing scripts.

Here, a *basic dataset* is one with entries such as Xtrain, Ytrain, Xtest, Ytest.
A *split dataset* has entries such as TrainClass, TrainSet, ValidClass, ValidSet,
TestClass, TestSet.

`preprocess.py` - performs normalization of the basic dataset

`lower_resolution.m` - takes the basic dataset and makes a new one with low
resolution images

`split_dataset.m` - splits train set into train and validation. Takes any basic 
dataset and outputs the split dataset.

`create_xor.m` - creates xor dataset (needs preprocessing)

`create_mini_xor.m` - creates mini xor dataset

`mp_X-Y_data.mat` - a basic dataset that is not preprocessed

`mp_X-Y_data_prepr.mat` - a basic preprocessed dataset 

`mp_X-Y_data_split.mat` - a prerocessed split dataset

`mp_X-Y_lowres_data.mat` - a low resolution basic dataset, not preprocessed

`xor_data.mat` - a basic dataset for xor problem (empty Xtest/Ytest)

`xor_prepr_data.mat` - a basic preprocessed dataset for xor problem (empty Xtest/Ytest)

`mini_xor_data.mat` - a basic dataset for xor problem (preprocessed) with only 4 datapoints
