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
- pkgutil
- importlib
- matplotlib
- sklearn

The scripts to be run are in the main folder, they are:
- **main**: prepares the data, then creates, trains, saves and tests a custom model, and finally displays the results.
- **finetune_pretrained_model**: downloads a pre-trained model (from torchvision models) and prepares the data with the 
standard procedures plus the specific preprocessing required by the chosen model. Then fine-tunes the NN on the
dataset, evaluates it on the test set and displays the results.
- **load_and_test_model**: loads a saved model (from name and ID) and just tests it on the test set. Can be used for both
custom and pre-trained fine-tuned models.
- **load_and_use_model**: loads a saved model (from name and ID) and uses it to predict the species of a given image. Can
be used for both custom and pre-trained fine-tuned models.

Saved models are stored in the folder "output/models". They have names chosen by the user (if it is not chosen, the
default is the type of Neural Network used). Then, they are assigned a sequential ID to avoid overwriting.

Parameters for the scripts can be modified in the file "config.py".

The data preprocessing expects data in data/step_1_data, loads the images and resizes them and saves them in
data/step_2_data. If specified in the config file, data augmentation is applied, and transformations specific to
pre-trained models are executed. The resulting images are organized in a pytorch dataset, which is split into train,
validation and test sets. The datasets are converted into pytorch dataloaders, which are saved in
data/step_3_data/augmentation_ _X_. _X_=1 means no augmentation; _X_=2 and _X_=3 mean respectively double and triple
the original data. When a script is run, if appropriate step_3 or step_2 data are present, they are not re-created.

