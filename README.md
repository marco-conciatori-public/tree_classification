# Tree Classification
**Identify tree species from aerial images corresponding to 2m x 2m patches on the ground.**

The environment needs to be set up with the following packages:
- python 3
- pytorch (gpu version if hardware supports it)
- torchvision
- openCV
- albumentations
- numpy
- pathlib
- matplotlib (optional, for confusion matrix display)
- sklearn (optional, for confusion matrix display)

The scripts to be run are in the main folder, they are:
- **train_custom_model**: prepares the data, then creates, trains, saves and tests a custom model, and finally displays the results.
- **finetune_pretrained_model**: downloads a pre-trained model (from torchvision models) and prepares the data with the 
standard procedures plus the specific preprocessing required by the chosen model. Then fine-tunes the NN on the
dataset, evaluates it on the test set and displays the results.
- **load_and_test_model**: loads a saved model (from name and ID) and just tests it on the test set. Can be used for both
custom and pre-trained fine-tuned models, can also display the confusion matrix of the test.
- **load_and_use_model**: loads a saved model (from name and ID) and uses it to predict the species of a given image. Can
be used for both custom and pre-trained fine-tuned models.

Saved models are stored in the folder "output/models". They have names chosen by the user (if it is not chosen, the
default is the type of Neural Network used). Then, they are assigned a sequential ID to avoid overwriting. Each of
them as also a meta_data file associated, which contains the parameters used to create the model and the results
obtained during training/fine-tuning and evaluation. The meda_data file is saved as a json file, so that it can be read
directly by humans.

Parameters for the scripts can be modified in the file "config.py". There are also some script-specific parameters
that can be modified directly in the script.

The data preprocessing expects data in data/step_1, loads the images and resizes them and saves them in
data/step_2. If specified in the config file, data augmentation is applied, and transformations specific to
pre-trained models are executed. The resulting images are organized in a pytorch dataset, which is split into train,
validation and test sets. The datasets are converted into pytorch dataloaders, which are saved in
data/step_3/augmentation_ _X_. _X_=1 means no augmentation; _X_=2 and _X_=3 mean respectively double and triple
the original data.
