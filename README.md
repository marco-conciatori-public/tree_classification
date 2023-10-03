# Tree Classification
**Identify tree species from aerial images corresponding to 2m x 2m patches on the ground.**

The environment needs to be set up with the following packages:
- python 3
- pytorch (gpu version if hardware supports it)
- torchvision
- openCV
- numpy
- pyyaml
- pathlib
- matplotlib (optional, for confusion matrix display)

The scripts that can be run are in the main folder, the list is as follows:
- **train_custom_model**: prepares the data, then creates, trains, saves and tests a custom model, and finally displays
the results.
- **finetune_pretrained_model**: downloads a pre-trained model (from torchvision models) and prepares the data with the 
standard procedures plus the specific preprocessing required by the chosen model. Then fine-tunes the NN on the
dataset, evaluates it on the test set and displays the results.
- **load_and_test_model**: loads a saved model (from name and ID) and just tests it on the test set. Can be used for
both custom and pre-trained fine-tuned models, can also display the confusion matrix of the test.
- **load_and_use_model**: loads a saved model (from name and ID) and uses it to predict the species of a given set of
images. Can be used for both custom and pre-trained fine-tuned models.
- **best_parameters_grid_search**: for each hyperparameter, a list of values can be provided. Performs a grid search
on them (trains and tests models for each possible combination), and outputs the evaluation of each combination of
parameters. Also shows the time required by each combination. The results are saved in a json file in 
"output/parameter_search".
- **resume_interromped_grid_search**: resumes the grid search for best hyperparameters that was interrupted.
- **show_difficult_cases**: use a trained/fine-tuned model on a set of images. Then it displays worst _n_ cases (where
the model was most confident in a wrong classification) along with the model's output, one by one so that users can
assess the model's behaviour in the small subset of wrongly classified images.
- **show_confusion_matrix**: loads a saved model's meta data (from name and ID) and displays the confusion matrix
of the test that was performed on it after it was trained/fine-tuned. All the information is already stored in the
meta data, so there is no need to redo the test, or even load the model.
- **show_parameter_search_results**: loads a saved parameter search from "output/parameter_search" (distinguished by
ID) and displays the results of the grid search for best hyperparameters that was performed. If only one parameter was
searched, it displays a graph of the results. If two parameters were searched, it displays a 3D graph of the results.
If more than two parameters were searched, it prints on screen the top _n_ best combinations of parameters.
- **analyze_orthomosaic**: automatically apply the model to a whole orthomosaic and creates an heatmap of the 
species distribution. **Not fully implemented yet**.


Saved models are stored in the folder "output/models". They have names chosen by the user (if it is not chosen, the
default is the type of Neural Network used). Then, they are assigned a sequential ID to avoid overwriting. Each of
them as also a meta_data file associated, which contains the parameters used to create the model and the results
obtained during training/fine-tuning and evaluation. The meda_data file is saved as a json file, so that it can be read
directly by humans.

Parameters for the scripts can be modified in the file "config.yaml" or provided via command line argument. There are
also some script-specific parameters that can be modified directly inside some of the scripts.

The data preprocessing expects data in "data/step_1", loads the images and resizes them. If specified in the config
file, shuffling, balancing, and augmentation are applied. Transformations specific to pre-trained models are executed.
The resulting images are organized into datasets, which are converted into pytorch dataloaders, and then saved in
data/step_3/augmentation_ _X_. _X_=1 means no augmentation; _X_=2 and _X_=3 mean respectively double and triple
the original data.
