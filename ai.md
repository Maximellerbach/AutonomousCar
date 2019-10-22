# AI approach

The main goal is to get an image, process it and take decision from it. 
To do that, I use a CNN (convolutional neural network)
you will find few architectures in train_model/convolution/architectures.py

To train my model I use something called behavior cloning, the goal of my model is to clone my driving style to drive itself autonomously.
To achieve that, I gather images while driving my car then train my model on those images.
To have best results, I also use data augmentation to push to the limit my model and thus have better result in real life.
