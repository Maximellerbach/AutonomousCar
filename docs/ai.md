# Deeplearning
I'm using the Tensorflow framework to create, train and evaluate models.
On top of Tensorflow, I'm using Keras which simplifies the syntax.
You can see [here](/custom_modules/custom_modules/architectures.py) some models architecture that I use.

## Model training
to train my models, I use a custom training loop using a generator.
This generator allows me to use some custom functions to augment my datas.

## Model testing
To evaluate my models, I use some functions to visualize the prediction of the model on a given dataset,
this is work in progress.
